// Expert down GEMV on real Q4_K weights: llama.cpp layout versus one lane group per row.
// The dot product and reduction follow ggml-cuda's mul_mat_vec_q and vec_dot_q4_K_q8_1.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#define CK(x)                                                                 \
  do {                                                                        \
    cudaError_t e_ = (x);                                                     \
    if (e_ != cudaSuccess) {                                                  \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

struct block_q4_K { half2 dm; uint8_t scales[12]; uint8_t qs[128]; };
struct block_q8_1 { half2 ds; int8_t qs[32]; };
static_assert(sizeof(block_q4_K) == 144 && sizeof(block_q8_1) == 36, "layout");

static const int K = 8, N_EXP = 128, EXPERT_KB = 6144;  // 768 x 2048 weights is 6144 K-blocks in either orientation

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// STAGE 0: weight loads. 1: + activation loads. 2: + scale loads and decode. 3: + dot product and float math.
template <int STAGE>
static __device__ __forceinline__ float dot(const block_q4_K *__restrict__ vb, const block_q8_1 *__restrict__ y, int kbx,
                                            int iqs, uint32_t &acc) {
  const block_q4_K *b = vb + kbx;
  const int off = 2 * ((iqs / 2) / 4);
  const int *q4 = (const int *)(b->qs + 16 * off + 4 * ((iqs / 2) % 4));
  const int v0 = q4[0], v1 = q4[4];
  if constexpr (STAGE == 0) {
    acc ^= (uint32_t)v0 ^ (uint32_t)v1;
    return 0.0f;
  } else {
    int u[4];
    float d8[2];
    for (int i = 0; i < 2; ++i) {
      const block_q8_1 *bq = y + off + i;
      d8[i] = __low2float(bq->ds);
      const int *q8 = (const int *)bq->qs + ((iqs / 2) % 4);
      u[2 * i] = q8[0];
      u[2 * i + 1] = q8[4];
    }
    if constexpr (STAGE == 1) {
      acc ^= (uint32_t)v0 ^ (uint32_t)v1 ^ (uint32_t)u[0] ^ (uint32_t)u[1] ^ (uint32_t)u[2] ^ (uint32_t)u[3];
      acc += (uint32_t)(d8[0] + d8[1]);
      return 0.0f;
    } else {
      const uint16_t *sc16 = (const uint16_t *)b->scales;
      const int j = off / 2, jm = j & 1;
      const uint32_t s0 = sc16[jm], s2 = sc16[jm + 2], s4 = sc16[jm + 4];
      const uint32_t hi = (uint32_t) - (int32_t)(j >= 2);
      uint16_t aux[2];
      aux[0] = (uint16_t)(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
      aux[1] = (uint16_t)(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
      const uint8_t *sc = (const uint8_t *)aux, *m = sc + 2;
      if constexpr (STAGE == 2) {
        acc ^= (uint32_t)v0 ^ (uint32_t)v1 ^ (uint32_t)u[0] ^ (uint32_t)u[1] ^ (uint32_t)u[2] ^ (uint32_t)u[3];
        acc += (uint32_t)(d8[0] + d8[1]) + aux[0] + aux[1] + (uint32_t)__low2float(b->dm);
        return 0.0f;
      } else {
        float sd = 0.0f, sm = 0.0f;
        for (int i = 0; i < 2; ++i) {
          const int v0i = (v0 >> (4 * i)) & 0x0F0F0F0F, v1i = (v1 >> (4 * i)) & 0x0F0F0F0F;
          const int d1 = __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i], 0));
          const int d2 = __dp4a(0x01010101, u[2 * i + 1], __dp4a(0x01010101, u[2 * i], 0));
          sd += d8[i] * (d1 * sc[i]);
          sm += d8[i] * (d2 * m[i]);
        }
        const float2 dm = __half22float2(b->dm);
        return dm.x * sd - dm.y * sm;
      }
    }
  }
}

static __device__ __forceinline__ float reduce32(float v) {
  v += __shfl_xor_sync(0xffffffff, v, 16);
  v += __shfl_xor_sync(0xffffffff, v, 8);
  v += __shfl_xor_sync(0xffffffff, v, 4);
  v += __shfl_xor_sync(0xffffffff, v, 2);
  v += __shfl_xor_sync(0xffffffff, v, 1);
  return v;
}

static __device__ __forceinline__ float silu_d(float x) { return x / (1.0f + expf(-x)); }

static uint16_t half_bits(const half2 &h, int i) {
  uint16_t b[2];
  memcpy(b, &h, 4);
  return b[i];
}

static float half_to_float(uint16_t v) {
  const int e = (v >> 10) & 0x1f, m = v & 0x3ff;
  const float s = (v & 0x8000) ? -1.0f : 1.0f;
  if (e == 0) return s * std::ldexp((float)m, -24);
  if (e == 31) return s * INFINITY;
  return s * std::ldexp((float)(m + 1024), e - 25);
}

// Dense reference for one expert matrix with the same quantized x.
static std::vector<float> host_gemv(const block_q4_K *w, int expert, const block_q8_1 *yq, int kbr, int nrows) {
  const int ncols = kbr * 256;
  std::vector<float> xq(ncols);
  for (int b = 0; b < ncols / 32; ++b) {
    const float d = half_to_float(half_bits(yq[b].ds, 0));
    for (int i = 0; i < 32; ++i) xq[b * 32 + i] = d * yq[b].qs[i];
  }
  std::vector<float> out(nrows);
  for (int r = 0; r < nrows; ++r) {
    double sum = 0;
    for (int kb = 0; kb < kbr; ++kb) {
      const block_q4_K &q = w[expert * EXPERT_KB + r * kbr + kb];
      const float d = half_to_float(half_bits(q.dm, 0)), dmin = half_to_float(half_bits(q.dm, 1));
      for (int s = 0; s < 8; ++s) {
        int sc, mn;
        if (s < 4) {
          sc = q.scales[s] & 63;
          mn = q.scales[s + 4] & 63;
        } else {
          sc = (q.scales[s + 4] & 0xF) | ((q.scales[s - 4] & 0xC0) >> 2);
          mn = (q.scales[s + 4] >> 4) | ((q.scales[s] & 0xC0) >> 2);
        }
        const int base = 16 * (s & ~1);
        for (int m = 0; m < 32; ++m) {
          const uint8_t byte = q.qs[base + m];
          const int nib = (s & 1) ? (byte >> 4) : (byte & 0xF);
          sum += (double)(d * sc * nib - dmin * mn) * xq[kb * 256 + s * 32 + m];
        }
      }
    }
    out[r] = (float)sum;
  }
  return out;
}

struct Env {
  std::vector<block_q4_K *> a, b;      // a: up or down tensors. b: gate tensors.
  std::vector<block_q4_K *> a_h, b_h;
  block_q8_1 *y = nullptr;
  std::vector<block_q8_1> y_h;
  int32_t *ids = nullptr;
  float *dst = nullptr;
  std::mt19937 rng{42};
  double peak = 180.0;
};


static const int KBR = 3, NROWS = 2048, Y_STRIDE = 24;

// llama.cpp layout: 16 threads per super-block, warps split the row, partial sums meet in shared memory.
template <int NW, int R>
__launch_bounds__(NW * 32, 1)
static __global__ void down_ref(const block_q4_K *__restrict__ vx, const block_q8_1 *__restrict__ y0, const int32_t *__restrict__ ids,
                                float *__restrict__ dst) {
  const int tid = 32 * threadIdx.y + threadIdx.x;
  const int row0 = R * blockIdx.x;
  const int kb0 = ids[blockIdx.y] * EXPERT_KB + row0 * KBR;
  const block_q8_1 *y = y0 + blockIdx.y * Y_STRIDE;
  float tmp[R] = {0.0f};
  uint32_t acc = 0;
  for (int kbx = tid / 16; kbx < KBR; kbx += 2 * NW) {
#pragma unroll
    for (int i = 0; i < R; ++i) tmp[i] += dot<3>(vx, &y[kbx * 8], kb0 + i * KBR + kbx, 2 * (tid % 16), acc);
  }
  __shared__ float sh[NW > 1 ? NW - 1 : 1][R][32];
  if constexpr (NW > 1) {
    if (threadIdx.y > 0) {
#pragma unroll
      for (int i = 0; i < R; ++i) sh[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
  }
#pragma unroll
  for (int i = 0; i < R; ++i) {
    float a = tmp[i];
#pragma unroll
    for (int l = 0; l < NW - 1; ++l) a += sh[l][i][threadIdx.x];
    a = reduce32(a);
    if (threadIdx.x == i) dst[blockIdx.y * NROWS + row0 + i] = a;
  }
}

// One group of 16 lanes owns a row and walks all its super-blocks. RPG rows per group run in turn.
// No shared memory and no __syncthreads; the sum stays inside the group.
template <int NW, int RPG>
__launch_bounds__(NW * 32, 1)
static __global__ void down_lanes(const block_q4_K *__restrict__ vx, const block_q8_1 *__restrict__ y0, const int32_t *__restrict__ ids,
                                  float *__restrict__ dst) {
  const int lane = threadIdx.x % 16;
  const int group = threadIdx.y * 2 + threadIdx.x / 16;
  const int row0 = (blockIdx.x * NW * 2 + group) * RPG;
  const int kb0 = ids[blockIdx.y] * EXPERT_KB + row0 * KBR;
  const block_q8_1 *y = y0 + blockIdx.y * Y_STRIDE;
  float tmp[RPG] = {0.0f};
  uint32_t acc = 0;
#pragma unroll
  for (int kbx = 0; kbx < KBR; ++kbx) {
#pragma unroll
    for (int i = 0; i < RPG; ++i) tmp[i] += dot<3>(vx, &y[kbx * 8], kb0 + i * KBR + kbx, 2 * lane, acc);
  }
#pragma unroll
  for (int i = 0; i < RPG; ++i) {
    float a = tmp[i];
    a += __shfl_xor_sync(0xffffffff, a, 8);
    a += __shfl_xor_sync(0xffffffff, a, 4);
    a += __shfl_xor_sync(0xffffffff, a, 2);
    a += __shfl_xor_sync(0xffffffff, a, 1);
    if (lane == 0) dst[blockIdx.y * NROWS + row0 + i] = a;
  }
}

struct DownEnv {
  std::vector<block_q4_K *> a, a_h;
  block_q8_1 *y = nullptr;
  std::vector<block_q8_1> y_h;
  int32_t *ids = nullptr;
  float *dst = nullptr;
  std::mt19937 rng{42};
};

template <typename F>
static double run(DownEnv &E, F launch, int rows_per_block, int threads, const char *label, double base_us) {
  const int nodes = (int)E.a.size();
  cudaStream_t st;
  CK(cudaStreamCreate(&st));
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  for (int m = 0; m < nodes; ++m) launch(E.a[m], E.ids + m * K, st);
  cudaGraph_t g = nullptr;
  cudaGraphExec_t ge = nullptr;
  CK(cudaStreamEndCapture(st, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  std::vector<int32_t> ids_h(nodes * K);
  std::vector<double> ts;
  for (int r = 0; r < 44; ++r) {
    // top-k picks distinct experts, so sample without replacement
    for (int m = 0; m < nodes; ++m) {
      std::vector<int32_t> perm(N_EXP);
      for (int e = 0; e < N_EXP; ++e) perm[e] = e;
      for (int s = 0; s < K; ++s) std::swap(perm[s], perm[s + E.rng() % (N_EXP - s)]);
      std::copy(perm.begin(), perm.begin() + K, ids_h.begin() + m * K);
    }
    CK(cudaMemcpy(E.ids, ids_h.data(), ids_h.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    double t0 = now_s();
    CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));
    if (r >= 4) ts.push_back(now_s() - t0);
  }
  std::sort(ts.begin(), ts.end());
  const double us = ts[ts.size() / 2] / nodes * 1e6;
  const double bytes = (double)NROWS * K * KBR * 144;
  std::vector<float> out(K * NROWS);
  const int last = nodes - 1;
  CK(cudaMemcpy(out.data(), E.dst, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
  double e2 = 0, n2 = 0;
  for (int slot : {0, 7}) {
    std::vector<float> want = host_gemv(E.a_h[last], ids_h[last * K + slot], E.y_h.data() + slot * Y_STRIDE, KBR, NROWS);
    for (int i = 0; i < NROWS; ++i) {
      const double d = (double)out[slot * NROWS + i] - want[i];
      e2 += d * d;
      n2 += (double)want[i] * want[i];
    }
  }
  char gain[32] = "";
  if (base_us > 0) snprintf(gain, sizeof gain, "%+6.1f%%", (base_us / us - 1) * 100);
  printf("%-34s %5d B/block %4d thr %7.1f us %6.1f GB/s %5.1f%% %8s err %.0e\n", label, rows_per_block * KBR * 144, threads, us,
         bytes / us / 1e3, bytes / us / 1e3 / 180 * 100, gain, std::sqrt(e2 / std::max(n2, 1e-30)));
  fflush(stdout);
  CK(cudaGraphExecDestroy(ge));
  CK(cudaGraphDestroy(g));
  CK(cudaStreamDestroy(st));
  return us;
}

template <int NW, int R>
static double ref(DownEnv &E, const char *label, double b) {
  return run(E, [&](const block_q4_K *w, const int32_t *ids, cudaStream_t st) {
    down_ref<NW, R><<<dim3(NROWS / R, K), dim3(32, NW), 0, st>>>(w, E.y, ids, E.dst);
  }, R, NW * 32, label, b);
}

template <int NW, int RPG>
static double lanes(DownEnv &E, const char *label, double b) {
  return run(E, [&](const block_q4_K *w, const int32_t *ids, cudaStream_t st) {
    down_lanes<NW, RPG><<<dim3(NROWS / (NW * 2 * RPG), K), dim3(32, NW), 0, st>>>(w, E.y, ids, E.dst);
  }, NW * 2 * RPG, NW * 32, label, b);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s MODEL.gguf INDEX.tsv\n", argv[0]);
    return 1;
  }
  int fd = open(argv[1], O_RDONLY);
  std::ifstream idx(argv[2]);
  DownEnv E;
  for (std::string line; std::getline(idx, line) && E.a.size() < 16;) {
    std::istringstream is(line);
    std::string name, dims;
    size_t off, size;
    int type;
    is >> name >> off >> size >> type >> dims;
    if (name.find("ffn_down_exps") == std::string::npos || type != 12) continue;
    block_q4_K *h = (block_q4_K *)malloc(size), *d = nullptr;
    for (size_t done = 0; done < size;) {
      ssize_t n = pread(fd, (char *)h + done, size - done, off + done);
      if (n <= 0) { perror("pread"); return 1; }
      done += n;
    }
    CK(cudaMalloc(&d, size));
    CK(cudaMemcpy(d, h, size, cudaMemcpyHostToDevice));
    E.a.push_back(d);
    E.a_h.push_back(h);
  }
  close(fd);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::mt19937 yr(7);
  E.y_h.resize(K * Y_STRIDE);
  for (auto &b : E.y_h) {
    float xs[32], amax = 0;
    for (float &x : xs) { x = nd(yr); amax = std::max(amax, std::fabs(x)); }
    int e;
    const float mant = std::frexp(amax / 127.0f, &e);
    const uint16_t hv = (uint16_t)(((e + 14) << 10) | ((int)std::round((mant * 2.0f - 1.0f) * 1024.0f) & 0x3ff));
    const float dq = half_to_float(hv);
    const uint16_t hb[2] = {hv, hv};
    memcpy(&b.ds, hb, 4);
    for (int i = 0; i < 32; ++i) b.qs[i] = (int8_t)std::lround(std::max(-127.0f, std::min(127.0f, xs[i] / dq)));
  }
  CK(cudaMalloc(&E.y, E.y_h.size() * sizeof(block_q8_1)));
  CK(cudaMemcpy(E.y, E.y_h.data(), E.y_h.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&E.ids, E.a.size() * K * sizeof(int32_t)));
  CK(cudaMalloc(&E.dst, K * NROWS * sizeof(float)));
  printf("expert down, Q4_K, %zu layers, 2048 rows x 768 cols x 8 experts, 7.08 MB per launch\n\n", E.a.size());

  for (int rep = 0; rep < 2; ++rep) {
    const double b = ref<2, 4>(E, "llama.cpp layout, 2 warps x 4 rows", 0);
    ref<4, 4>(E, "llama.cpp layout, 4 warps x 4 rows", b);
    ref<1, 8>(E, "llama.cpp layout, 1 warp x 8 rows", b);
    lanes<1, 1>(E, "lane groups, 1 warp, 1 row/group", b);
    lanes<2, 1>(E, "lane groups, 2 warps, 1 row/group", b);
    lanes<4, 1>(E, "lane groups, 4 warps, 1 row/group", b);
    lanes<8, 1>(E, "lane groups, 8 warps, 1 row/group", b);
    lanes<1, 2>(E, "lane groups, 1 warp, 2 rows/group", b);
    lanes<2, 2>(E, "lane groups, 2 warps, 2 rows/group", b);
    lanes<4, 2>(E, "lane groups, 4 warps, 2 rows/group", b);
    lanes<2, 4>(E, "lane groups, 2 warps, 4 rows/group", b);
    puts("");
  }
  return 0;
}

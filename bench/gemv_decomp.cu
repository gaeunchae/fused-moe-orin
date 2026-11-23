// Decompose llama.cpp's Q4_K expert GEMV on real weights: add one cost at a time, then sweep the block shape.
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

static const int N_FF = 768, N_EMBD = 2048, KB_PER_ROW = N_EMBD / 256, K = 8, N_EXP = 128;
static const int EXPERT_KB = N_FF * KB_PER_ROW;

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static __device__ __forceinline__ int dp4a(int a, int b, int c) { return __dp4a(a, b, c); }

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
          const int d1 = dp4a(v1i, u[2 * i + 1], dp4a(v0i, u[2 * i], 0));
          const int d2 = dp4a(0x01010101, u[2 * i + 1], dp4a(0x01010101, u[2 * i], 0));
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

// STAGE 4 adds shared-memory partial sums, warp reduction, activation and the result write.
template <int STAGE, int NMAT, int NW, int R>
__launch_bounds__(NW * 32, 1)
static __global__ void gemv(const block_q4_K *__restrict__ vup, const block_q4_K *__restrict__ vgate,
                            const block_q8_1 *__restrict__ y, const int32_t *__restrict__ ids, float *__restrict__ dst) {
  const int tid = 32 * threadIdx.y + threadIdx.x;
  const int row0 = R * blockIdx.x;
  const int cx = ids[blockIdx.y];
  const int kb0 = cx * EXPERT_KB + row0 * KB_PER_ROW;
  float tmp[R] = {0.0f}, tmp_g[R] = {0.0f};
  uint32_t acc = 0;
  for (int kbx = tid / 16; kbx < KB_PER_ROW; kbx += 2 * NW) {
    const int kby = kbx * 8, kqs = 2 * (tid % 16);
#pragma unroll
    for (int i = 0; i < R; ++i) {
      tmp[i] += dot<STAGE>(vup, &y[kby], kb0 + i * KB_PER_ROW + kbx, kqs, acc);
      if constexpr (NMAT == 2) tmp_g[i] += dot<STAGE>(vgate, &y[kby], kb0 + i * KB_PER_ROW + kbx, kqs, acc);
    }
  }
  if constexpr (STAGE < 4) {
    if (threadIdx.x == 0 && threadIdx.y == 0) dst[0] = (float)acc + tmp[0] + tmp_g[0];
    return;
  } else {
    __shared__ float sh[NW > 1 ? NW - 1 : 1][R][32];
    __shared__ float sh_g[(NMAT == 2 && NW > 1) ? NW - 1 : 1][R][32];
    if constexpr (NW > 1) {
      if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
          sh[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
          if constexpr (NMAT == 2) sh_g[threadIdx.y - 1][i][threadIdx.x] = tmp_g[i];
        }
      }
      __syncthreads();
      if (threadIdx.y > 0) return;
    }
    float *d = dst + blockIdx.y * N_FF + row0;
#pragma unroll
    for (int i = 0; i < R; ++i) {
      float a = tmp[i], g = tmp_g[i];
#pragma unroll
      for (int l = 0; l < NW - 1; ++l) {
        a += sh[l][i][threadIdx.x];
        if constexpr (NMAT == 2) g += sh_g[l][i][threadIdx.x];
      }
      a = reduce32(a);
      if constexpr (NMAT == 2) g = reduce32(g);
      if (threadIdx.x == i) d[i] = NMAT == 2 ? a * silu_d(g) : a;
    }
  }
}

struct Env {
  std::vector<block_q4_K *> up, gate;
  std::vector<block_q4_K *> up_h, gate_h;
  block_q8_1 *y = nullptr;
  std::vector<block_q8_1> y_h;
  int32_t *ids = nullptr;
  float *dst = nullptr;
  std::mt19937 rng{42};
  double peak = 180.0;
};

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

// Dense reference for one expert matrix: y = W x with the same quantized x.
static std::vector<float> host_gemv(const block_q4_K *w, int expert, const std::vector<block_q8_1> &yq) {
  std::vector<float> xq(N_EMBD);
  for (int b = 0; b < N_EMBD / 32; ++b) {
    const float d = half_to_float(half_bits(yq[b].ds, 0));
    for (int i = 0; i < 32; ++i) xq[b * 32 + i] = d * yq[b].qs[i];
  }
  std::vector<float> out(N_FF);
  for (int r = 0; r < N_FF; ++r) {
    double sum = 0;
    for (int kb = 0; kb < KB_PER_ROW; ++kb) {
      const block_q4_K &q = w[expert * EXPERT_KB + r * KB_PER_ROW + kb];
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

template <int STAGE, int NMAT, int NW, int R>
static double run(Env &E, const char *label, bool verify = false) {
  const int nodes = (int)E.up.size();
  cudaStream_t st;
  CK(cudaStreamCreate(&st));
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  for (int m = 0; m < nodes; ++m)
    gemv<STAGE, NMAT, NW, R><<<dim3(N_FF / R, K), dim3(32, NW), 0, st>>>(E.up[m], E.gate[m], E.y, E.ids + m * K, E.dst);
  cudaGraph_t g = nullptr;
  cudaGraphExec_t ge = nullptr;
  CK(cudaStreamEndCapture(st, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  std::vector<int32_t> ids_h(nodes * K);
  std::uniform_int_distribution<int> pick(0, N_EXP - 1);
  std::vector<double> ts;
  for (int r = 0; r < 24; ++r) {
    for (auto &x : ids_h) x = pick(E.rng);
    CK(cudaMemcpy(E.ids, ids_h.data(), ids_h.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    double t0 = now_s();
    CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));
    if (r >= 4) ts.push_back(now_s() - t0);
  }
  std::sort(ts.begin(), ts.end());
  const double us = ts[ts.size() / 2] / nodes * 1e6;
  const double bytes = (double)NMAT * N_FF * K * KB_PER_ROW * 144;
  const double weights = (double)NMAT * N_FF * K * N_EMBD;
  char err[64] = "";
  if (verify) {
    std::vector<float> out(K * N_FF);
    const int last = nodes - 1;
    CK(cudaMemcpy(out.data(), E.dst, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double e2 = 0, n2 = 0;
    for (int slot : {0, 7}) {
      std::vector<float> up = host_gemv(E.up_h[last], ids_h[last * K + slot], E.y_h);
      std::vector<float> ga;
      if (NMAT == 2) ga = host_gemv(E.gate_h[last], ids_h[last * K + slot], E.y_h);
      for (int i = 0; i < N_FF; ++i) {
        const float want = NMAT == 2 ? up[i] * (ga[i] / (1.0f + std::exp(-ga[i]))) : up[i];
        const double diff = (double)out[slot * N_FF + i] - want;
        e2 += diff * diff;
        n2 += (double)want * want;
      }
    }
    snprintf(err, sizeof err, "  rel.err %.1e", std::sqrt(e2 / std::max(n2, 1e-30)));
  }
  printf("%-42s %8.1f us %7.1f GB/s %6.1f%% %7.1f Gw/s%s\n", label, us, bytes / us / 1e3, bytes / us / 1e3 / E.peak * 100,
         weights / us / 1e3, err);
  fflush(stdout);
  CK(cudaGraphExecDestroy(ge));
  CK(cudaGraphDestroy(g));
  CK(cudaStreamDestroy(st));
  return us;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s MODEL.gguf INDEX.tsv [n_layers]\n", argv[0]);
    return 1;
  }
  const int n_layers = argc > 3 ? atoi(argv[3]) : 16;
  Env E;
  int fd = open(argv[1], O_RDONLY);
  std::ifstream idx(argv[2]);
  struct Item { size_t off, size; int type; };
  std::vector<Item> ups(48), gates(48);
  for (std::string line; std::getline(idx, line);) {
    std::istringstream is(line);
    std::string name, dims;
    Item it;
    is >> name >> it.off >> it.size >> it.type >> dims;
    int il = atoi(name.c_str() + 4);
    if (name.find("ffn_up_exps") != std::string::npos) ups[il] = it;
    if (name.find("ffn_gate_exps") != std::string::npos) gates[il] = it;
  }
  auto load = [&](const Item &it, std::vector<block_q4_K *> &dev, std::vector<block_q4_K *> &host) {
    if (it.type != 12) { printf("unexpected tensor type %d\n", it.type); exit(1); }
    block_q4_K *h = (block_q4_K *)malloc(it.size), *d = nullptr;
    size_t done = 0;
    while (done < it.size) {
      ssize_t n = pread(fd, (char *)h + done, it.size - done, it.off + done);
      if (n <= 0) { perror("pread"); exit(1); }
      done += n;
    }
    CK(cudaMalloc(&d, it.size));
    CK(cudaMemcpy(d, h, it.size, cudaMemcpyHostToDevice));
    dev.push_back(d);
    host.push_back(h);
  };
  for (int il = 0; il < n_layers; ++il) {
    load(ups[il], E.up, E.up_h);
    load(gates[il], E.gate, E.gate_h);
  }
  close(fd);

  E.y_h.resize(N_EMBD / 32);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (auto &b : E.y_h) {
    float xs[32], amax = 0;
    for (float &x : xs) { x = nd(E.rng); amax = std::max(amax, std::fabs(x)); }
    const float d = amax / 127.0f;
    int e;
    const float mant = std::frexp(d, &e);
    uint16_t hv = (uint16_t)(((e + 14) << 10) | ((int)std::round((mant * 2.0f - 1.0f) * 1024.0f) & 0x3ff));
    const float dq = half_to_float(hv);
    uint16_t hb[2] = {hv, hv};
    memcpy(&b.ds, hb, 4);
    for (int i = 0; i < 32; ++i) b.qs[i] = (int8_t)std::lround(std::max(-127.0f, std::min(127.0f, xs[i] / dq)));
  }
  CK(cudaMalloc(&E.y, E.y_h.size() * sizeof(block_q8_1)));
  CK(cudaMemcpy(E.y, E.y_h.data(), E.y_h.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&E.ids, n_layers * K * sizeof(int32_t)));
  CK(cudaMalloc(&E.dst, K * N_FF * sizeof(float)));

  printf("real Q4_K weights, %d layers, x = random N(0,1), graph nodes, 8 experts per launch\n", n_layers);
  printf("target: llama.cpp gate+up node traced at 156.6 us (90 GB/s)\n\n");
  printf("%-42s %11s %12s %7s %12s\n", "variant", "per launch", "weights", "peak", "weights/s");

  puts("-- Stage by stage, gate+up, 128 threads, 1 row per block (llama.cpp's shape) --");
  run<0, 2, 4, 1>(E, "S0 weight loads only");
  run<1, 2, 4, 1>(E, "S1 + activation loads");
  run<2, 2, 4, 1>(E, "S2 + scale loads and decode");
  run<3, 2, 4, 1>(E, "S3 + dot product and float math");
  run<4, 2, 4, 1>(E, "S4 + reduction, SiLU, write (= llama.cpp)", true);

  puts("-- Same, one matrix --");
  run<0, 1, 4, 1>(E, "S0 weight loads only");
  run<1, 1, 4, 1>(E, "S1 + activation loads");
  run<2, 1, 4, 1>(E, "S2 + scale loads and decode");
  run<3, 1, 4, 1>(E, "S3 + dot product and float math");
  run<4, 1, 4, 1>(E, "S4 + reduction and write", true);

  puts("-- Block shape sweep, full kernel, gate+up (warps per block x rows per block) --");
#define SW(NW, R) run<4, 2, NW, R>(E, "warps=" #NW " rows=" #R, true)
  SW(1, 1); SW(1, 2); SW(1, 4); SW(1, 8); SW(1, 16);
  SW(2, 1); SW(2, 2); SW(2, 4); SW(2, 8); SW(2, 16);
  SW(4, 1); SW(4, 2); SW(4, 4); SW(4, 8); SW(4, 16);
  SW(8, 1); SW(8, 2); SW(8, 4); SW(8, 8); SW(8, 16);

  puts("-- Same block shapes before reduction (S3), gate+up --");
#define S3(NW, R) run<3, 2, NW, R>(E, "S3 warps=" #NW " rows=" #R)
  S3(1, 1); S3(1, 4); S3(1, 16); S3(2, 4); S3(2, 16); S3(4, 4); S3(4, 16);
  return 0;
}

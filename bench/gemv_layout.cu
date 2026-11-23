// Gate+up GEMV with weights placed as in llama.cpp (one buffer, same relative offsets) versus separate allocations.
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
#include <map>
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


static __device__ __forceinline__ void prefetch_l2(const void *p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

// PF 0: as llama.cpp. 1: prefetch the next iteration of the same row (llama.cpp's GB10 path, distance 1).
// 2: unroll the K loop so all weight loads issue before the math. 3: prefetch the rows of the block
// pf_blocks later in launch order into L2. 4: 2 and 3 together.
template <int NMAT, int NW, int R, int KBR, int NROWS, int PF>
__launch_bounds__(NW * 32, 1)
static __global__ void gemv(const block_q4_K *__restrict__ vx, const block_q4_K *__restrict__ vgate,
                            const block_q8_1 *__restrict__ y0, int y_stride, const int32_t *__restrict__ ids,
                            float *__restrict__ dst, int pf_blocks) {
  const int tid = 32 * threadIdx.y + threadIdx.x;
  const int row0 = R * blockIdx.x;
  const int cx = ids[blockIdx.y];
  const int kb0 = cx * EXPERT_KB + row0 * KBR;
  const block_q8_1 *y = y0 + blockIdx.y * y_stride;

  if constexpr (PF == 3 || PF == 4) {
    const int nbx = gridDim.x;
    const int t = blockIdx.y * nbx + blockIdx.x + pf_blocks;
    if (t < nbx * (int)gridDim.y) {
      const int ty = t / nbx, tx = t - ty * nbx;
      const size_t off = ((size_t)ids[ty] * EXPERT_KB + (size_t)R * tx * KBR) * sizeof(block_q4_K);
      constexpr int bytes = R * KBR * sizeof(block_q4_K);
      const int first = (int)(off & 127);
      const int lines = (first + bytes + 127) / 128;
#pragma unroll 1
      for (int l = tid; l < NMAT * lines; l += NW * 32) {
        const char *base = (const char *)((NMAT == 2 && l >= lines) ? vgate : vx);
        const int li = l >= lines ? l - lines : l;
        prefetch_l2(base + (off & ~(size_t)127) + (size_t)li * 128);
      }
    }
  }

  float tmp[R] = {0.0f}, tmp_g[R] = {0.0f};
  uint32_t acc = 0;
  if constexpr (PF == 2 || PF == 4) {
    constexpr int ITERS = (KBR + 2 * NW - 1) / (2 * NW);
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      const int kbx = tid / 16 + it * 2 * NW;
      if (kbx < KBR) {
        const int kby = kbx * 8, kqs = 2 * (tid % 16);
#pragma unroll
        for (int i = 0; i < R; ++i) {
          tmp[i] += dot<3>(vx, &y[kby], kb0 + i * KBR + kbx, kqs, acc);
          if constexpr (NMAT == 2) tmp_g[i] += dot<3>(vgate, &y[kby], kb0 + i * KBR + kbx, kqs, acc);
        }
      }
    }
  } else {
    for (int kbx = tid / 16; kbx < KBR; kbx += 2 * NW) {
      const int kby = kbx * 8, kqs = 2 * (tid % 16);
      if constexpr (PF == 1) {
        if (kbx + 2 * NW < KBR) {
#pragma unroll
          for (int i = 0; i < R; ++i) {
            prefetch_l2(vx + kb0 + i * KBR + kbx + 2 * NW);
            if constexpr (NMAT == 2) prefetch_l2(vgate + kb0 + i * KBR + kbx + 2 * NW);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < R; ++i) {
        tmp[i] += dot<3>(vx, &y[kby], kb0 + i * KBR + kbx, kqs, acc);
        if constexpr (NMAT == 2) tmp_g[i] += dot<3>(vgate, &y[kby], kb0 + i * KBR + kbx, kqs, acc);
      }
    }
  }

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
  float *d = dst + blockIdx.y * NROWS + row0;
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

template <int NMAT, int NW, int R, int KBR, int NROWS, int PF>
static double run(Env &E, int y_stride, int pf_blocks, const char *label, double base_us) {
  const int nodes = (int)E.a.size();
  cudaStream_t st;
  CK(cudaStreamCreate(&st));
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  for (int m = 0; m < nodes; ++m)
    gemv<NMAT, NW, R, KBR, NROWS, PF><<<dim3(NROWS / R, K), dim3(32, NW), 0, st>>>(
        E.a[m], NMAT == 2 ? E.b[m] : E.a[m], E.y, y_stride, E.ids + m * K, E.dst, pf_blocks);
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
  const double spread = (ts[ts.size() * 9 / 10] - ts[ts.size() / 10]) / nodes * 1e6;
  const double bytes = (double)NMAT * NROWS * K * KBR * 144;
  std::vector<float> out(K * NROWS);
  const int last = nodes - 1;
  CK(cudaMemcpy(out.data(), E.dst, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
  double e2 = 0, n2 = 0;
  for (int slot : {0, 7}) {
    const block_q8_1 *yq = E.y_h.data() + slot * y_stride;
    std::vector<float> x = host_gemv(E.a_h[last], ids_h[last * K + slot], yq, KBR, NROWS), ga;
    if (NMAT == 2) ga = host_gemv(E.b_h[last], ids_h[last * K + slot], yq, KBR, NROWS);
    for (int i = 0; i < NROWS; ++i) {
      const float want = NMAT == 2 ? x[i] * (ga[i] / (1.0f + std::exp(-ga[i]))) : x[i];
      const double diff = (double)out[slot * NROWS + i] - want;
      e2 += diff * diff;
      n2 += (double)want * want;
    }
  }
  char gain[32] = "";
  if (base_us > 0) snprintf(gain, sizeof gain, "%+6.1f%%", (base_us / us - 1) * 100);
  printf("NW=%d R=%d %-26s pf=%-4d %7.1f us (p10-p90 %4.1f) %6.1f GB/s %5.1f%% %8s err %.0e\n", NW, R, label, pf_blocks, us, spread,
         bytes / us / 1e3, bytes / us / 1e3 / E.peak * 100, gain, std::sqrt(e2 / std::max(n2, 1e-30)));
  fflush(stdout);
  CK(cudaGraphExecDestroy(ge));
  CK(cudaGraphDestroy(g));
  CK(cudaStreamDestroy(st));
  return us;
}

struct Item { size_t off = 0, size = 0; int type = -1; };

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s MODEL.gguf INDEX.tsv ADDRESSES.tsv\n", argv[0]);
    return 1;
  }
  const int n_layers = argc > 4 ? atoi(argv[4]) : 16;
  int fd = open(argv[1], O_RDONLY);
  std::map<std::string, Item> file;
  {
    std::ifstream idx(argv[2]);
    for (std::string line; std::getline(idx, line);) {
      std::istringstream is(line);
      std::string name, dims;
      Item it;
      is >> name >> it.off >> it.size >> it.type >> dims;
      file[name] = it;
    }
  }
  std::map<std::string, unsigned long long> devaddr;
  {
    std::ifstream ad(argv[3]);
    for (std::string line; std::getline(ad, line);) {
      std::istringstream is(line);
      std::string name, type;
      unsigned long long a;
      size_t sz;
      is >> name >> a >> sz >> type;
      devaddr[name] = a;
    }
  }
  auto read_host = [&](const Item &it) {
    block_q4_K *h = (block_q4_K *)malloc(it.size);
    for (size_t done = 0; done < it.size;) {
      ssize_t n = pread(fd, (char *)h + done, it.size - done, it.off + done);
      if (n <= 0) { perror("pread"); exit(1); }
      done += n;
    }
    return h;
  };
  auto gname = [](int il) { return "blk." + std::to_string(il) + ".ffn_gate_exps.weight"; };
  auto uname = [](int il) { return "blk." + std::to_string(il) + ".ffn_up_exps.weight"; };

  std::vector<block_q4_K *> up_h, gate_h;
  for (int il = 0; il < n_layers; ++il) {
    up_h.push_back(read_host(file[uname(il)]));
    gate_h.push_back(read_host(file[gname(il)]));
  }
  close(fd);
  const size_t tsize = file[uname(0)].size;

  // One device buffer spanning layers 0..15 as llama.cpp lays them out, with room to move the up tensors.
  unsigned long long lo = ~0ull, hi = 0;
  for (int il = 0; il < n_layers; ++il) {
    lo = std::min({lo, devaddr[gname(il)], devaddr[uname(il)]});
    hi = std::max({hi, devaddr[gname(il)] + tsize, devaddr[uname(il)] + tsize});
  }
  const size_t lo_mod = lo % (2u << 20);
  const size_t span = hi - lo + lo_mod + (8u << 20);
  char *big = nullptr;
  CK(cudaMalloc(&big, span));
  printf("one buffer of %.2f GB, base aligned like llama.cpp's within 2 MiB (offset %zu)\n", span / 1e9, lo_mod);

  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::mt19937 yr(7);
  std::vector<block_q8_1> yh(K * 64);
  for (auto &b : yh) {
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
  Env U;
  U.y_h = yh;
  CK(cudaMalloc(&U.y, yh.size() * sizeof(block_q8_1)));
  CK(cudaMemcpy(U.y, yh.data(), yh.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&U.ids, n_layers * K * sizeof(int32_t)));
  CK(cudaMalloc(&U.dst, K * 2048 * sizeof(float)));
  U.a_h = up_h;
  U.b_h = gate_h;

  // Separate allocations, as the earlier harness did.
  std::vector<block_q4_K *> sep_up, sep_gate;
  for (int il = 0; il < std::min(n_layers, 16); ++il) {
    block_q4_K *a = nullptr, *b = nullptr;
    CK(cudaMalloc(&a, tsize));
    CK(cudaMalloc(&b, tsize));
    CK(cudaMemcpy(a, up_h[il], tsize, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b, gate_h[il], tsize, cudaMemcpyHostToDevice));
    sep_up.push_back(a);
    sep_gate.push_back(b);
  }

  // Place the tensors in the big buffer; up moves by `shift` bytes (multiple of 256) from its llama.cpp offset.
  auto place = [&](long shift) {
    U.a.clear();
    U.b.clear();
    for (int il = 0; il < n_layers; ++il) {
      char *g = big + lo_mod + (devaddr[gname(il)] - lo);
      char *u = big + lo_mod + (long)(devaddr[uname(il)] - lo) + shift;
      CK(cudaMemcpy(g, gate_h[il], tsize, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(u, up_h[il], tsize, cudaMemcpyHostToDevice));
      U.a.push_back((block_q4_K *)u);
      U.b.push_back((block_q4_K *)g);
    }
  };

  puts("\n=== gate+up, 4 warps x 2 rows, Q4_K, 14.16 MB per launch ===");
  for (int rep = 0; rep < 2; ++rep) {
    U.a = sep_up;
    U.b = sep_gate;
    double b = run<2, 4, 2, 8, 768, 3>(U, 0, 64, "separate allocs (16 layers), prefetch 64", 0);
    run<2, 4, 2, 8, 768, 0>(U, 0, 0, "separate allocs (16 layers), no prefetch", b);
    for (long shift : n_layers > 16 ? std::vector<long>{0L} : std::vector<long>{0L, -256L, 256L, 768L, 1792L, 3840L, 65280L, 1048320L}) {
      place(shift);
      char label[64];
      snprintf(label, sizeof label, "llama.cpp layout, %d layers, up %+ld B", n_layers, shift);
      run<2, 4, 2, 8, 768, 3>(U, 0, 64, label, b);
    }
    place(0);
    run<2, 4, 2, 8, 768, 0>(U, 0, 0, "llama.cpp layout, no prefetch", b);
    puts("");
  }
  return 0;
}

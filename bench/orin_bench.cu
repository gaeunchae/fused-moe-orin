// Orin micro-benchmark: GPU read bandwidth vs. bytes-per-launch, and launch cost.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#define CK(x)                                                                 \
  do {                                                                        \
    cudaError_t e_ = (x);                                                     \
    if (e_ != cudaSuccess) {                                                  \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

__global__ void fill_kernel(uint4 *p, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t a = (uint32_t)(i * 2654435761u) ^ 0x9e3779b9u;
  p[i] = make_uint4(a, a * 2654435761u, ~a, a ^ 0x5bd1e995u);
}

// Each block streams a contiguous tile with 128-bit loads, one vector per thread per iteration.
__global__ void read_kernel(const uint4 *__restrict__ p, size_t n_vec, int iters, uint32_t *out) {
  const size_t tile = (size_t)blockIdx.x * (size_t)blockDim.x * (size_t)iters;
  uint32_t acc = 0;
  for (int it = 0; it < iters; ++it) {
    const size_t idx = tile + (size_t)it * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
      const uint4 v = __ldg(p + idx);
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  if (threadIdx.x == 0) out[blockIdx.x & 1023] = acc;
}

__global__ void empty_kernel() {}

// Reads `total` bytes in launches of `chunk` bytes each, queued without per-launch sync.
static double read_gbps(const uint4 *buf, uint32_t *out, size_t buf_bytes, size_t total, size_t chunk,
                        int threads, int iters, int rounds, double *mean_out) {
  const size_t span_bytes = (size_t)threads * iters * sizeof(uint4);
  const int blocks = (int)std::max<size_t>(1, chunk / span_bytes);
  const size_t real_chunk = (size_t)blocks * span_bytes;
  const size_t launches = std::max<size_t>(1, total / real_chunk);
  double best = 0, sum = 0;
  for (int r = 0; r < rounds; ++r) {
    size_t off = 0;
    double t0 = now_s();
    for (size_t l = 0; l < launches; ++l) {
      if (off + real_chunk > buf_bytes) off = 0;
      read_kernel<<<blocks, threads>>>(buf + off / sizeof(uint4), real_chunk / sizeof(uint4), iters, out);
      off += real_chunk;
    }
    CK(cudaDeviceSynchronize());
    double t = now_s() - t0;
    double g = launches * real_chunk / t / 1e9;
    best = std::max(best, g);
    sum += g;
  }
  if (mean_out) *mean_out = sum / rounds;
  return best;
}

static void launch_cost(const char *name, int blocks, int threads, int n) {
  double best_sync = 1e30, best_q = 1e30, best_g = 1e30;
  for (int r = 0; r < 5; ++r) {
    double t0 = now_s();
    for (int i = 0; i < n / 5; ++i) {
      empty_kernel<<<blocks, threads>>>();
      CK(cudaDeviceSynchronize());
    }
    best_sync = std::min(best_sync, (now_s() - t0) / (n / 5) * 1e6);
    t0 = now_s();
    for (int i = 0; i < n; ++i) empty_kernel<<<blocks, threads>>>();
    CK(cudaDeviceSynchronize());
    best_q = std::min(best_q, (now_s() - t0) / n * 1e6);
  }
  // 1152 nodes = the proposal's GEMV count per token.
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  cudaGraph_t g = nullptr;
  cudaGraphExec_t ge = nullptr;
  cudaError_t ce = cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
  if (ce == cudaSuccess) {
    for (int i = 0; i < 1152; ++i) empty_kernel<<<blocks, threads, 0, s>>>();
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    for (int r = 0; r < 20; ++r) {
      double t0 = now_s();
      CK(cudaGraphLaunch(ge, s));
      CK(cudaStreamSynchronize(s));
      best_g = std::min(best_g, (now_s() - t0) / 1152 * 1e6);
    }
    printf("%-22s launch+sync %8.2f us | queued %7.2f us | graph node %7.2f us\n", name, best_sync, best_q, best_g);
    CK(cudaGraphExecDestroy(ge));
    CK(cudaGraphDestroy(g));
  } else {
    printf("%-22s launch+sync %8.2f us | queued %7.2f us | graph capture failed: %s\n", name, best_sync, best_q,
           cudaGetErrorString(ce));
  }
  CK(cudaStreamDestroy(s));
}

int main() {
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));
  printf("device=%s cc=%d.%d SMs=%d integrated=%d l2=%d KiB\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount, prop.integrated, prop.l2CacheSize / 1024);

  const size_t buf_bytes = 6ull << 30;
  uint4 *buf = nullptr;
  uint32_t *out = nullptr;
  CK(cudaMalloc(&buf, buf_bytes));
  CK(cudaMalloc(&out, 4096));
  const size_t n_vec = buf_bytes / sizeof(uint4);
  double t0 = now_s();
  fill_kernel<<<(int)(n_vec / 256 + 1), 256>>>(buf, n_vec);
  CK(cudaDeviceSynchronize());
  printf("fill: %.1f GB/s (write)\n", buf_bytes / (now_s() - t0) / 1e9);

  // Warm up so devfreq reaches its top GPU/EMC rates before measuring.
  double mean = 0;
  read_gbps(buf, out, buf_bytes, buf_bytes, buf_bytes, 256, 16, 8, &mean);

  puts("\n== Peak sequential read, one launch over 6 GiB ==");
  for (int th : {64, 128, 256, 512, 1024}) {
    for (int it : {4, 16, 64}) {
      double g = read_gbps(buf, out, buf_bytes, buf_bytes, buf_bytes, th, it, 6, &mean);
      printf("threads=%4d iters=%3d  best %6.1f GB/s  mean %6.1f GB/s\n", th, it, g, mean);
    }
  }

  puts("\n== Read bandwidth vs. bytes per launch (no per-launch sync) ==");
  for (size_t kib : {size_t(2500 / 3), size_t(2500), size_t(20000), size_t(200000), size_t(1800000)}) {
    double g = read_gbps(buf, out, buf_bytes, 6ull << 30, kib * 1000, 256, 16, 5, &mean);
    printf("%8zu kB/launch  best %6.1f GB/s  mean %6.1f GB/s\n", kib, g, mean);
  }

  puts("\n== Empty-kernel cost ==");
  launch_cost("1 block x 1 thread", 1, 1, 5000);
  launch_cost("256 blocks x 256 thr", 256, 256, 5000);

  CK(cudaFree(buf));
  CK(cudaFree(out));
  return 0;
}

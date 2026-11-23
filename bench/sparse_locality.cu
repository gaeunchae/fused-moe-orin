// Orin micro-benchmark: effective read bandwidth when only active channels of expert matrices are read.
// A channel is one contiguous 1024 B run (2048 weights at 4 bits). An expert matrix holds 768 channels.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <random>
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

static const int CH_BYTES = 1024;
static const int THREADS = 64;  // 64 threads x 16 B = one channel per iteration
static const int CH_PER_EXPERT = 768;

__global__ void fill_kernel(uint4 *p, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t a = (uint32_t)(i * 2654435761u) ^ 0x9e3779b9u;
  p[i] = make_uint4(a, a * 2654435761u, ~a, a ^ 0x5bd1e995u);
}

// Each block reads ch_per_block channels named by the index list.
__global__ void sparse_read(const uint4 *__restrict__ p, const uint32_t *__restrict__ ids, int n_ids, int ch_per_block,
                            uint32_t *out) {
  uint32_t acc = 0;
  for (int it = 0; it < ch_per_block; ++it) {
    const int k = blockIdx.x * ch_per_block + it;
    if (k >= n_ids) break;
    const uint32_t ch = __ldg(ids + k);
    const uint4 v = __ldg(p + (size_t)ch * (CH_BYTES / sizeof(uint4)) + threadIdx.x);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (threadIdx.x == 0) out[blockIdx.x & 1023] = acc;
}

// run == 0 means one contiguous prefix per expert. Otherwise active channels come in contiguous runs of `run`.
static std::vector<uint32_t> make_ids(size_t n_experts, double sparsity, int run, bool shuffle, std::mt19937 &rng,
                                      std::vector<size_t> *expert_end) {
  std::vector<uint32_t> ids;
  const int active = (int)(CH_PER_EXPERT * (1.0 - sparsity) + 0.5);
  ids.reserve(n_experts * active);
  std::vector<int> slots;
  for (size_t e = 0; e < n_experts; ++e) {
    const uint32_t base = (uint32_t)(e * CH_PER_EXPERT);
    if (run == 0 || active == CH_PER_EXPERT) {
      for (int c = 0; c < active; ++c) ids.push_back(base + c);
    } else {
      const int n_slots = CH_PER_EXPERT / run, n_runs = std::max(1, active / run);
      slots.resize(n_slots);
      for (int i = 0; i < n_slots; ++i) slots[i] = i;
      for (int i = 0; i < n_runs; ++i) std::swap(slots[i], slots[i + rng() % (n_slots - i)]);
      std::sort(slots.begin(), slots.begin() + n_runs);
      for (int i = 0; i < n_runs; ++i)
        for (int c = 0; c < run; ++c) ids.push_back(base + slots[i] * run + c);
    }
    if (shuffle) std::shuffle(ids.end() - std::min<size_t>(ids.size(), active), ids.end(), rng);
    if (expert_end) expert_end->push_back(ids.size());
  }
  return ids;
}

// Same reads, but the grid is fixed and the active count comes from memory, as a captured graph node requires.
__global__ void sparse_read_fixed(const uint4 *__restrict__ p, const uint32_t *__restrict__ ids, const int *__restrict__ count,
                                  int ch_per_block, uint32_t *out) {
  const int first = blockIdx.x * ch_per_block;
  const int n_ids = __ldg(count);
  if (first >= n_ids) return;
  uint32_t acc = 0;
  for (int it = 0; it < ch_per_block; ++it) {
    const int k = first + it;
    if (k >= n_ids) break;
    const uint32_t ch = __ldg(ids + k);
    const uint4 v = __ldg(p + (size_t)ch * (CH_BYTES / sizeof(uint4)) + threadIdx.x);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (threadIdx.x == 0) out[blockIdx.x & 1023] = acc;
}

struct Result { double best, mean; size_t nbytes; };

// All experts in one launch: no launch cost, only memory behaviour.
static Result one_launch(const uint4 *buf, uint32_t *out, uint32_t *ids_dev, const std::vector<uint32_t> &ids, int cpb, int rounds) {
  CK(cudaMemcpy(ids_dev, ids.data(), ids.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  const int blocks = (int)((ids.size() + cpb - 1) / cpb);
  Result r{0, 0, ids.size() * CH_BYTES};
  for (int i = 0; i < rounds; ++i) {
    double t0 = now_s();
    sparse_read<<<blocks, THREADS>>>(buf, ids_dev, (int)ids.size(), cpb, out);
    CK(cudaDeviceSynchronize());
    double g = r.nbytes / (now_s() - t0) / 1e9;
    r.best = std::max(r.best, g);
    r.mean += g / rounds;
  }
  return r;
}

// One launch per k-expert matrix, as a per-GEMV kernel would run.
static Result per_matrix(const uint4 *buf, uint32_t *out, uint32_t *ids_dev, const std::vector<uint32_t> &ids,
                         const std::vector<size_t> &ends, int k, int cpb, int rounds) {
  CK(cudaMemcpy(ids_dev, ids.data(), ids.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  Result r{0, 0, ids.size() * CH_BYTES};
  for (int i = 0; i < rounds; ++i) {
    double t0 = now_s();
    size_t start = 0;
    for (size_t e = k - 1; e < ends.size(); e += k) {
      const int n = (int)(ends[e] - start);
      sparse_read<<<(n + cpb - 1) / cpb, THREADS>>>(buf, ids_dev + start, n, cpb, out);
      start = ends[e];
    }
    CK(cudaDeviceSynchronize());
    double g = r.nbytes / (now_s() - t0) / 1e9;
    r.best = std::max(r.best, g);
    r.mean += g / rounds;
  }
  return r;
}

int main() {
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));
  printf("device=%s SMs=%d l2=%d KiB, channel=%d B, expert=%d channels\n", prop.name, prop.multiProcessorCount,
         prop.l2CacheSize / 1024, CH_BYTES, CH_PER_EXPERT);

  const size_t buf_bytes = 6ull << 30;
  uint4 *buf = nullptr;
  uint32_t *out = nullptr, *ids_dev = nullptr;
  CK(cudaMalloc(&buf, buf_bytes));
  CK(cudaMalloc(&out, 4096));
  const size_t n_vec = buf_bytes / sizeof(uint4);
  fill_kernel<<<(int)(n_vec / 256 + 1), 256>>>(buf, n_vec);
  CK(cudaDeviceSynchronize());

  const size_t n_experts = buf_bytes / (CH_PER_EXPERT * CH_BYTES);
  CK(cudaMalloc(&ids_dev, n_experts * CH_PER_EXPERT * sizeof(uint32_t)));
  std::mt19937 rng(1234);
  const int rounds = 6;

  puts("\n== Dense baseline through the index list: block-size effect ==");
  auto dense = make_ids(n_experts, 0.0, 0, false, rng, nullptr);
  for (int cpb : {1, 2, 4, 8, 16}) {
    Result r = one_launch(buf, out, ids_dev, dense, cpb, rounds);
    printf("channels/block=%2d  best %6.1f GB/s  mean %6.1f GB/s\n", cpb, r.best, r.mean);
  }

  puts("\n== Sparse reads, all in one launch (memory behaviour only), 4 channels/block ==");
  printf("%-10s %-22s %8s %8s\n", "sparsity", "pattern", "GB read", "GB/s");
  for (double s : {0.5, 0.7, 0.85, 0.95}) {
    for (int run : {1, 2, 4, 8, 16, 32, 64, 0}) {
      if (run && run > (int)(CH_PER_EXPERT * (1 - s))) continue;
      auto ids = make_ids(n_experts, s, run, false, rng, nullptr);
      Result r = one_launch(buf, out, ids_dev, ids, 4, rounds);
      char name[32];
      if (run) snprintf(name, sizeof name, "run=%d channels", run); else snprintf(name, sizeof name, "one prefix");
      printf("%-10.2f %-22s %8.2f %8.1f\n", s, name, r.nbytes / 1e9, r.best);
    }
    auto ids = make_ids(n_experts, s, 1, true, rng, nullptr);
    Result r = one_launch(buf, out, ids_dev, ids, 4, rounds);
    printf("%-10.2f %-22s %8.2f %8.1f\n", s, "run=1, shuffled", r.nbytes / 1e9, r.best);
  }

  puts("\n== One launch per 8-expert matrix (launch cost included), 4 channels/block ==");
  printf("%-10s %-14s %10s %8s\n", "sparsity", "pattern", "MB/launch", "GB/s");
  for (double s : {0.0, 0.5, 0.7, 0.85, 0.95}) {
    for (int run : {1, 16, 0}) {
      if (run && s == 0.0 && run != 1) continue;
      std::vector<size_t> ends;
      auto ids = make_ids(n_experts, s, run, false, rng, &ends);
      Result r = per_matrix(buf, out, ids_dev, ids, ends, 8, 4, rounds);
      char name[32];
      if (run) snprintf(name, sizeof name, "run=%d", run); else snprintf(name, sizeof name, "prefix");
      printf("%-10.2f %-14s %10.2f %8.1f\n", s, name, (ends[7] * (double)CH_BYTES) / 1e6, r.best);
    }
  }

  puts("\n== Knee: one launch, 4 channels/block ==");
  printf("%-10s %8s %8s %8s %8s\n", "sparsity", "run=1", "run=4", "run=16", "prefix");
  for (double s : {0.80, 0.85, 0.875, 0.90, 0.925, 0.95}) {
    printf("%-10.3f", s);
    for (int run : {1, 4, 16, 0}) {
      auto ids = make_ids(n_experts, s, run, false, rng, nullptr);
      printf(" %8.1f", one_launch(buf, out, ids_dev, ids, 4, rounds).best);
    }
    puts("");
  }

  // Captured graph over 8-expert matrices with a fixed grid: active lists sit at fixed slots, counts in device memory.
  puts("\n== Captured CUDA graph, fixed grid per 8-expert matrix, run=1, 4 channels/block ==");
  printf("%-10s %10s %8s %14s\n", "sparsity", "MB/node", "GB/s", "us/node idle");
  const int k = 8, slot = k * CH_PER_EXPERT, n_nodes = (int)(n_experts / k);
  const int fixed_blocks = slot / 4;
  int *counts = nullptr;
  CK(cudaMalloc(&counts, n_nodes * sizeof(int)));
  cudaStream_t st;
  CK(cudaStreamCreate(&st));
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  for (int m = 0; m < n_nodes; ++m)
    sparse_read_fixed<<<fixed_blocks, THREADS, 0, st>>>(buf, ids_dev + (size_t)m * slot, counts + m, 4, out);
  cudaGraph_t g = nullptr;
  cudaGraphExec_t ge = nullptr;
  CK(cudaStreamEndCapture(st, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  std::vector<int> c_host(n_nodes);
  std::vector<uint32_t> slotted((size_t)n_nodes * slot);
  for (double s : {1.0, 0.95, 0.85, 0.70, 0.50, 0.0}) {
    std::vector<size_t> ends;
    auto ids = make_ids(n_experts, std::min(s, 0.9999), 1, false, rng, &ends);
    if (s == 1.0) ids.clear();
    size_t total = 0;
    for (int m = 0; m < n_nodes; ++m) {
      const size_t a = s == 1.0 ? 0 : (m ? ends[m * k - 1] : 0), b = s == 1.0 ? 0 : ends[(m + 1) * k - 1];
      c_host[m] = (int)(b - a);
      total += b - a;
      if (b > a) std::copy(ids.begin() + a, ids.begin() + b, slotted.begin() + (size_t)m * slot);
    }
    CK(cudaMemcpy(ids_dev, slotted.data(), slotted.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(counts, c_host.data(), c_host.size() * sizeof(int), cudaMemcpyHostToDevice));
    double best_t = 1e30;
    for (int r = 0; r < rounds; ++r) {
      double t0 = now_s();
      CK(cudaGraphLaunch(ge, st));
      CK(cudaStreamSynchronize(st));
      best_t = std::min(best_t, now_s() - t0);
    }
    printf("%-10.2f %10.2f %8.1f %14.2f\n", s, total * (double)CH_BYTES / n_nodes / 1e6,
           total * (double)CH_BYTES / best_t / 1e9, best_t / n_nodes * 1e6);
  }
  CK(cudaGraphExecDestroy(ge));
  CK(cudaGraphDestroy(g));
  CK(cudaStreamDestroy(st));
  CK(cudaFree(counts));

  CK(cudaFree(buf));
  CK(cudaFree(out));
  CK(cudaFree(ids_dev));
  return 0;
}

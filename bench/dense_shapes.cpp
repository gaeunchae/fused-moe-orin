// Time single-token dense GEMV (MUL_MAT, as in attention and dense FFN layers) through ggml's CUDA backend over matrix
// sizes, to separate two causes of slow block shapes: rows covered in several passes (too few warps) and grids too
// small to fill the GPU (too few blocks). Link against a build's ggml libraries, as mmid_formats.
// Variant "plain" is one matrix per node; "fused" is gate and up with SwiGLU, run by the CUDA backend as one GEMV.
// Each graph reads at least 64 MB, far more than L2, so weights come from DRAM. Small kernels take 1-2 us, so the
// wall-clock column (which includes the input quantization) is only a sanity check: run under nsys and use
// scripts/dense_extract.py for kernel times. Usage: dense_shapes [TYPE|all] [plain|fused] [COLS] [ROWS]
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const size_t GRAPH_BYTES = 64ull << 20;
static const int    MAX_NODES = 512, WARMUP = 3, RUNS = 15;

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    const double peak = getenv("PEAK_GBS") ? atof(getenv("PEAK_GBS")) : 180.0;  // measured peak read, GB/s
    const std::vector<ggml_type> types = {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0};
    const std::vector<int> cols_list = {2048, 4096, 12288};
    const std::vector<int> rows_list = {256, 512, 1024, 2048, 4096, 8192, 16384};
    const std::string only      = argc > 1 && std::string(argv[1]) != "all" ? argv[1] : "";
    const bool        fused     = argc > 2 && std::string(argv[2]) == "fused";
    const int         only_cols = argc > 3 ? atoi(argv[3]) : 0;
    const int         only_rows = argc > 4 ? atoi(argv[4]) : 0;
    const int         n_mats    = fused ? 2 : 1;

    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) {
        fprintf(stderr, "no CUDA backend\n");
        return 1;
    }
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // nodes = launches per graph; dense_extract.py splits the trace by these counts, in this order
    printf("%-6s %5s %6s %6s %5s %8s %8s %6s  %s\n", "type", "cols", "rows", "B/row", "nodes", "us/node", "GB/s", "%peak",
           fused ? "fused" : "plain");
    for (ggml_type type : types) {
        if (!only.empty() && only != ggml_type_name(type)) {
            continue;
        }
        for (int cols : cols_list) {
            for (int rows : rows_list) {
                if ((only_cols && cols != only_cols) || (only_rows && rows != only_rows)) {
                    continue;
                }
                const size_t row_bytes = ggml_row_size(type, cols);
                const size_t mat_bytes = row_bytes * rows;
                const int    n_nodes   = (int) std::min<size_t>(MAX_NODES, std::max<size_t>(8, (GRAPH_BYTES + n_mats * mat_bytes - 1) / (n_mats * mat_bytes)));

                const size_t graph_size = 8 * (size_t) n_nodes;
                ggml_init_params ip = {ggml_tensor_overhead() * (6 * (size_t) n_nodes + 4) + ggml_graph_overhead_custom(graph_size, false), nullptr, true};
                ggml_context * ctx = ggml_init(ip);
                ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, 1);
                std::vector<ggml_tensor *> as;
                ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);
                for (int i = 0; i < n_nodes; ++i) {
                    ggml_tensor * up  = ggml_new_tensor_2d(ctx, type, cols, rows);
                    ggml_tensor * out = ggml_mul_mat(ctx, up, x);
                    as.push_back(up);
                    if (fused) {
                        ggml_tensor * gate = ggml_new_tensor_2d(ctx, type, cols, rows);
                        as.push_back(gate);
                        out = ggml_swiglu_split(ctx, ggml_mul_mat(ctx, gate, x), out);
                    }
                    ggml_build_forward_expand(gf, out);
                }
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

                // Quantize a tile of 64 rows of random weights and repeat it: timing does not depend on the values.
                const int tile = std::min(rows, 64);
                std::vector<float> f((size_t) cols * tile);
                for (float & v : f) {
                    v = nd(rng);
                }
                std::vector<uint8_t> q(row_bytes * tile), full(mat_bytes);
                ggml_quantize_chunk(type, f.data(), q.data(), 0, tile, cols, nullptr);
                for (size_t off = 0; off < mat_bytes; off += q.size()) {
                    memcpy(full.data() + off, q.data(), std::min(q.size(), mat_bytes - off));
                }
                for (ggml_tensor * a : as) {
                    ggml_backend_tensor_set(a, full.data(), 0, mat_bytes);
                }
                std::vector<float> xv(cols);
                for (float & v : xv) {
                    v = nd(rng);
                }
                ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));

                for (int w = 0; w < WARMUP; ++w) {
                    ggml_backend_graph_compute(be, gf);
                }
                std::vector<double> ts;
                for (int r = 0; r < RUNS; ++r) {
                    const double t0 = now_s();
                    ggml_backend_graph_compute(be, gf);
                    ts.push_back(now_s() - t0);
                }
                std::sort(ts.begin(), ts.end());
                const double us    = ts[ts.size() / 2] / n_nodes * 1e6;
                const double bytes = (double) n_mats * mat_bytes;
                printf("%-6s %5d %6d %6zu %5d %8.2f %8.1f %5.1f%%\n", ggml_type_name(type), cols, rows, row_bytes, n_nodes, us,
                       bytes / us / 1e3, bytes / us / 1e3 / peak * 100);
                fflush(stdout);

                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
            }
        }
    }
    ggml_backend_free(be);
    return 0;
}

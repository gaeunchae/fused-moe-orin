// Time single-token MUL_MAT_ID (the MoE expert GEMV) through ggml's CUDA backend for many weight formats and row lengths.
// Link it against a build's ggml libraries to compare builds: the kernels are the ones llama.cpp runs at decode.
// Variant "plain" is one matrix per node with one input row per expert slot, as in down. Variant "fused" is up and
// gate with a shared input and SwiGLU, which the CUDA backend runs as one GEMV reading two matrices, as in gate+up.
// Each graph holds N_NODES nodes with their own weights, so a node never finds its experts in L2 from an earlier node.
// Times are wall clock per node and include the small Q8_1 quantization of the input vector (about 2 us).
// Usage: mmid_formats [TYPE|all] [plain|fused] [COLS]
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static const int N_EXPERT = 16, N_USED = 8, N_NODES = 16, ROWS = 2048;

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    // measured peak read bandwidth, GB/s: 180 on this Orin; PEAK_GBS sets it on other GPUs
    const double peak = getenv("PEAK_GBS") ? atof(getenv("PEAK_GBS")) : 180.0;
    const std::vector<ggml_type> types = {
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_MXFP4, GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS,
    };
    const std::vector<int> cols_list = {512, 768, 1024, 2048, 4096};
    const std::string only  = argc > 1 && std::string(argv[1]) != "all" ? argv[1] : "";  // type name filter, e.g. q4_K
    const bool        fused = argc > 2 && std::string(argv[2]) == "fused";
    const int         only_cols = argc > 3 ? atoi(argv[3]) : 0;
    const int         n_mats    = fused ? 2 : 1;

    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) {
        fprintf(stderr, "no CUDA backend\n");
        return 1;
    }
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    printf("%-7s %5s %7s %8s %8s %6s  %s\n", "type", "cols", "B/row", "us/node", "GB/s", "%peak", fused ? "fused" : "plain");
    for (ggml_type type : types) {
        if (!only.empty() && only != ggml_type_name(type)) {
            continue;
        }
        for (int cols : cols_list) {
            if (only_cols && cols != only_cols) {
                continue;
            }
            ggml_init_params ip = {ggml_tensor_overhead() * (6 * N_NODES + 4) + ggml_graph_overhead(), nullptr, true};
            ggml_context * ctx = ggml_init(ip);
            // down: one input row per expert slot. gate+up: one input row shared by all slots.
            ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cols, fused ? 1 : N_USED, 1);
            std::vector<ggml_tensor *> as, ids(N_NODES);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            for (int i = 0; i < N_NODES; ++i) {
                ids[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, 1);
                ggml_tensor * up = ggml_new_tensor_3d(ctx, type, cols, ROWS, N_EXPERT);
                ggml_tensor * out = ggml_mul_mat_id(ctx, up, b, ids[i]);
                as.push_back(up);
                if (fused) {
                    ggml_tensor * gate = ggml_new_tensor_3d(ctx, type, cols, ROWS, N_EXPERT);
                    as.push_back(gate);
                    out = ggml_swiglu_split(ctx, ggml_mul_mat_id(ctx, gate, b, ids[i]), out);
                }
                ggml_build_forward_expand(gf, out);
            }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

            // Quantize one expert of random weights and copy it into every expert: timing does not depend on the values.
            const size_t expert_bytes = ggml_row_size(type, cols) * ROWS;
            std::vector<float> f((size_t) cols * ROWS);
            for (float & x : f) {
                x = nd(rng);
            }
            std::vector<uint8_t> q(expert_bytes);
            ggml_quantize_chunk(type, f.data(), q.data(), 0, ROWS, cols, nullptr);
            std::vector<float> xb((size_t) cols * b->ne[1]);
            for (float & x : xb) {
                x = nd(rng);
            }
            ggml_backend_tensor_set(b, xb.data(), 0, xb.size() * sizeof(float));
            for (ggml_tensor * a : as) {
                for (int e = 0; e < N_EXPERT; ++e) {
                    ggml_backend_tensor_set(a, q.data(), e * expert_bytes, expert_bytes);
                }
            }
            std::vector<int32_t> perm(N_EXPERT);
            for (int i = 0; i < N_NODES; ++i) {
                std::iota(perm.begin(), perm.end(), 0);  // distinct experts per node, as top-k routing picks
                std::shuffle(perm.begin(), perm.end(), rng);
                ggml_backend_tensor_set(ids[i], perm.data(), 0, N_USED * sizeof(int32_t));
            }

            for (int w = 0; w < 3; ++w) {
                ggml_backend_graph_compute(be, gf);
            }
            std::vector<double> ts;
            for (int r = 0; r < 15; ++r) {
                const double t0 = now_s();
                ggml_backend_graph_compute(be, gf);
                ts.push_back(now_s() - t0);
            }
            std::sort(ts.begin(), ts.end());
            const double us    = ts[ts.size() / 2] / N_NODES * 1e6;
            const double bytes = (double) n_mats * N_USED * expert_bytes;
            printf("%-7s %5d %7zu %8.1f %8.1f %5.1f%%\n", ggml_type_name(type), cols, ggml_row_size(type, cols), us,
                   bytes / us / 1e3, bytes / us / 1e3 / peak * 100);
            fflush(stdout);

            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    }
    ggml_backend_free(be);
    return 0;
}

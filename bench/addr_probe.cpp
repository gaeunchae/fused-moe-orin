// Print the device address of every expert weight tensor that a MUL_MAT_ID node reads in one decode step.
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <set>
#include <string>

static std::set<std::string> seen;

static bool on_tensor(ggml_tensor * t, bool ask, void *) {
    if (ask && t->op == GGML_OP_MUL_MAT_ID && t->src[0] && !seen.count(t->src[0]->name)) {
        seen.insert(t->src[0]->name);
        printf("%s\t%llu\t%zu\t%s\n", t->src[0]->name, (unsigned long long) (uintptr_t) t->src[0]->data, ggml_nbytes(t->src[0]),
               ggml_type_name(t->src[0]->type));
    }
    return false;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.cb_eval = on_tensor;
    llama_context * ctx = llama_init_from_model(model, cp);
    llama_token tok = 1;
    llama_decode(ctx, llama_batch_get_one(&tok, 1));
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

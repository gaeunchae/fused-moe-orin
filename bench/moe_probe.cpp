// Probe MoE channel activations in llama.cpp through the eval callback.
//   dump: greedy-decode each prompt, write gate/up/ids/weights per layer plus reference tokens and logits.
//   mask: teacher-force the reference tokens, zero h = SiLU(gate)*up on selected channels, compare logits.
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static const int K = 8;
static const int N_FF = 768;
static const int N_LAYER = 48;

enum class Score { none, gate, up, h };

struct Probe {
  bool capture = false;  // false during prefill so fusion stays intact
  bool dump = false;
  bool ask_all = false;   // split the graph at the same nodes as dump mode
  bool no_split = false;  // leave the graph untouched
  FILE *save = nullptr;   // write mask-mode logits here
  Score score = Score::none;
  double sparsity = 0.0;
  int step = 0;
  FILE *out = nullptr;
  std::vector<float> gate[N_LAYER], up[N_LAYER], w[N_LAYER];
  std::vector<int32_t> ids[N_LAYER];
  long zeroed = 0, total = 0;
};

static float silu(float x) { return x / (1.0f + std::exp(-x)); }

static bool name_is(const char *name, const char *prefix, int *il) {
  const size_t n = strlen(prefix);
  if (strncmp(name, prefix, n) != 0 || name[n] != '-') return false;
  *il = atoi(name + n + 1);
  return *il >= 0 && *il < N_LAYER;
}

static bool on_tensor(ggml_tensor *t, bool ask, void *ud) {
  Probe *p = (Probe *)ud;
  if (!p->capture || p->no_split) return ask ? false : true;
  int il = -1;
  const bool is_gate = name_is(t->name, "ffn_moe_gate", &il);
  const bool is_up = !is_gate && name_is(t->name, "ffn_moe_up", &il);
  const bool is_swiglu = !is_gate && !is_up && name_is(t->name, "ffn_moe_swiglu", &il);
  const bool is_topk = !is_gate && !is_up && !is_swiglu && name_is(t->name, "ffn_moe_topk", &il);
  const bool is_w = !is_gate && !is_up && !is_swiglu && !is_topk && name_is(t->name, "ffn_moe_weights_norm", &il);
  const bool want = is_swiglu || is_gate || is_up || ((p->dump || p->ask_all) && (is_topk || is_w));
  if (ask) return want;
  if (!want) return true;

  const size_t nb = ggml_nbytes(t);
  if (is_gate || is_up) {
    GGML_ASSERT(nb == K * N_FF * sizeof(float));
    auto &v = is_gate ? p->gate[il] : p->up[il];
    v.resize(K * N_FF);
    ggml_backend_tensor_get(t, v.data(), 0, nb);
  } else if (is_topk) {
    p->ids[il].resize(K);
    ggml_backend_tensor_get(t, p->ids[il].data(), 0, K * sizeof(int32_t));
  } else if (is_w) {
    p->w[il].resize(K);
    ggml_backend_tensor_get(t, p->w[il].data(), 0, K * sizeof(float));
  } else if (is_swiglu) {
    if (p->dump) {
      int32_t hdr[2] = {p->step, il};
      fwrite(hdr, sizeof hdr, 1, p->out);
      fwrite(p->ids[il].data(), sizeof(int32_t), K, p->out);
      fwrite(p->w[il].data(), sizeof(float), K, p->out);
      fwrite(p->gate[il].data(), sizeof(float), K * N_FF, p->out);
      fwrite(p->up[il].data(), sizeof(float), K * N_FF, p->out);
    }
    if (p->score != Score::none && p->sparsity > 0.0) {
      GGML_ASSERT(nb == K * N_FF * sizeof(float));
      std::vector<float> h(K * N_FF);
      ggml_backend_tensor_get(t, h.data(), 0, nb);
      const int keep = N_FF - (int)(N_FF * p->sparsity + 0.5);
      std::vector<std::pair<float, int>> s(N_FF);
      for (int e = 0; e < K; ++e) {
        for (int c = 0; c < N_FF; ++c) {
          const int i = e * N_FF + c;
          float v = 0;
          if (p->score == Score::gate) v = std::fabs(silu(p->gate[il][i]));
          if (p->score == Score::up) v = std::fabs(p->up[il][i]);
          if (p->score == Score::h) v = std::fabs(h[i]);
          s[c] = {v, c};
        }
        std::nth_element(s.begin(), s.begin() + keep, s.end(), [](auto &a, auto &b) { return a.first > b.first; });
        for (int j = keep; j < N_FF; ++j) h[e * N_FF + s[j].second] = 0.0f;
        p->zeroed += N_FF - keep;
        p->total += N_FF;
      }
      ggml_backend_tensor_set(t, h.data(), 0, nb);
    }
  }
  return true;
}

static std::vector<llama_token> tokenize(const llama_vocab *v, const std::string &s) {
  std::vector<llama_token> toks(s.size() + 16);
  int n = llama_tokenize(v, s.c_str(), (int)s.size(), toks.data(), (int)toks.size(), false, true);
  GGML_ASSERT(n >= 0);
  toks.resize(n);
  return toks;
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s MODEL PROMPTS OUTDIR dump|mask N_GEN [score:sparsity,...] [max_prompts]\n", argv[0]);
    return 1;
  }
  const std::string model_path = argv[1], prompts_path = argv[2], outdir = argv[3], mode = argv[4];
  const int n_gen = atoi(argv[5]);
  const std::string cfgs = argc > 6 ? argv[6] : "";
  const int max_prompts = argc > 7 ? atoi(argv[7]) : 1 << 30;

  std::vector<std::pair<std::string, std::string>> prompts;
  std::ifstream pf(prompts_path);
  for (std::string line; std::getline(pf, line);) {
    const size_t tab = line.find('\t');
    std::string text = line.substr(tab + 1), un;
    for (size_t i = 0; i < text.size(); ++i) {
      if (text[i] == '\\' && i + 1 < text.size() && text[i + 1] == 'n') { un += '\n'; ++i; }
      else un += text[i];
    }
    prompts.push_back({line.substr(0, tab), un});
  }
  if ((int)prompts.size() > max_prompts) prompts.resize(max_prompts);

  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 99;
  llama_model *model = llama_model_load_from_file(model_path.c_str(), mp);
  GGML_ASSERT(model);
  const llama_vocab *vocab = llama_model_get_vocab(model);
  const int n_vocab = llama_vocab_n_tokens(vocab);

  Probe probe;
  probe.ask_all = getenv("PROBE_ASK_ALL") != nullptr;
  probe.no_split = getenv("PROBE_NO_SPLIT") != nullptr;
  if (const char *path = getenv("PROBE_SAVE")) probe.save = fopen(path, "wb");
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = 4096;
  cp.n_batch = 2048;
  cp.n_ubatch = 512;
  cp.cb_eval = on_tensor;
  cp.cb_eval_user_data = &probe;
  llama_context *ctx = llama_init_from_model(model, cp);
  GGML_ASSERT(ctx);

  auto start_prompt = [&](const std::string &text) {
    llama_memory_clear(llama_get_memory(ctx), true);
    std::string chat = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<llama_token> toks = tokenize(vocab, chat);
    if (toks.size() > 1024) toks.resize(1024);
    probe.capture = false;
    GGML_ASSERT(llama_decode(ctx, llama_batch_get_one(toks.data(), (int)toks.size())) == 0);
    return toks.size();
  };

  if (mode == "dump") {
    probe.dump = true;
    probe.out = fopen((outdir + "/activations.bin").c_str(), "wb");
    FILE *flog = fopen((outdir + "/ref_logits.bin").c_str(), "wb");
    std::ofstream ftok(outdir + "/ref_tokens.txt");
    int step = 0;
    for (size_t pi = 0; pi < prompts.size(); ++pi) {
      size_t np = start_prompt(prompts[pi].second);
      ftok << pi << ' ' << prompts[pi].first << ' ' << np;
      float *logits = llama_get_logits_ith(ctx, -1);
      for (int g = 0; g < n_gen; ++g) {
        llama_token tok = (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
        if (llama_vocab_is_eog(vocab, tok)) break;
        ftok << ' ' << tok;
        probe.capture = true;
        probe.step = step++;
        GGML_ASSERT(llama_decode(ctx, llama_batch_get_one(&tok, 1)) == 0);
        logits = llama_get_logits_ith(ctx, -1);
        fwrite(logits, sizeof(float), n_vocab, flog);
      }
      ftok << '\n';
      fprintf(stderr, "dump prompt %zu/%zu steps=%d\n", pi + 1, prompts.size(), step);
    }
    fclose(probe.out);
    fclose(flog);
  } else {
    // Reference tokens and logits come from the dump run.
    std::vector<std::vector<llama_token>> ref;
    std::vector<std::string> dom;
    std::ifstream ftok(outdir + "/ref_tokens.txt");
    for (std::string line; std::getline(ftok, line);) {
      std::istringstream is(line);
      int pi; std::string d; size_t np; is >> pi >> d >> np;
      std::vector<llama_token> v; llama_token x;
      while (is >> x) v.push_back(x);
      ref.push_back(v); dom.push_back(d);
    }
    std::vector<size_t> skip_steps(ref.size() + 1, 0);
    for (size_t i = 0; i < ref.size(); ++i) skip_steps[i + 1] = skip_steps[i] + ref[i].size();
    FILE *flog = fopen((outdir + "/ref_logits.bin").c_str(), "rb");
    GGML_ASSERT(flog);
    std::ofstream res(outdir + "/mask_results.jsonl", std::ios::app);
    std::vector<float> rl(n_vocab);
    std::vector<double> pr(n_vocab), qr(n_vocab);
    std::istringstream cs(cfgs);
    for (std::string c; std::getline(cs, c, ',');) {
      const size_t col = c.find(':');
      const std::string sn = c.substr(0, col);
      probe.score = sn == "gate" ? Score::gate : sn == "up" ? Score::up : sn == "h" ? Score::h : Score::none;
      probe.sparsity = atof(c.c_str() + col + 1);
      probe.zeroed = probe.total = 0;
      probe.dump = false;
      long n = 0, top1 = 0, top5 = 0;
      double kl_sum = 0;
      std::vector<double> kls;
      std::map<std::string, std::pair<long, long>> by_dom;
      for (size_t pi = 0; pi < prompts.size() && pi < ref.size(); ++pi) {
        start_prompt(prompts[pi].second);
        for (size_t g = 0; g < ref[pi].size(); ++g) {
          probe.capture = true;
          llama_token tok = ref[pi][g];
          GGML_ASSERT(llama_decode(ctx, llama_batch_get_one(&tok, 1)) == 0);
          const float *q = llama_get_logits_ith(ctx, -1);
          if (probe.save) fwrite(q, sizeof(float), n_vocab, probe.save);
          fseek(flog, (long)((skip_steps[pi] + g) * (size_t)n_vocab * sizeof(float)), SEEK_SET);
          GGML_ASSERT(fread(rl.data(), sizeof(float), n_vocab, flog) == (size_t)n_vocab);
          double ms_p = -1e300, ms_q = -1e300, zp = 0, zq = 0;
          for (int i = 0; i < n_vocab; ++i) { ms_p = std::max(ms_p, (double)rl[i]); ms_q = std::max(ms_q, (double)q[i]); }
          for (int i = 0; i < n_vocab; ++i) {
            pr[i] = std::exp((double)rl[i] - ms_p); zp += pr[i];
            qr[i] = std::exp((double)q[i] - ms_q); zq += qr[i];
          }
          double kl = 0;
          for (int i = 0; i < n_vocab; ++i) {
            const double a = pr[i] / zp, b = qr[i] / zq;
            if (a > 1e-30) kl += a * (std::log(a) - std::log(std::max(b, 1e-300)));
          }
          const int arg_p = (int)(std::max_element(rl.begin(), rl.end()) - rl.begin());
          const int arg_q = (int)(std::max_element(q, q + n_vocab) - q);
          std::vector<int> idx(n_vocab);
          for (int i = 0; i < n_vocab; ++i) idx[i] = i;
          std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) { return q[a] > q[b]; });
          const bool in5 = std::find(idx.begin(), idx.begin() + 5, arg_p) != idx.begin() + 5;
          ++n; top1 += arg_p == arg_q; top5 += in5; kl_sum += kl; kls.push_back(kl);
          by_dom[dom[pi]].first++; by_dom[dom[pi]].second += arg_p == arg_q;
        }
        fprintf(stderr, "%s prompt %zu/%zu\n", c.c_str(), pi + 1, prompts.size());
      }
      std::sort(kls.begin(), kls.end());
      std::ostringstream js;
      js << "{\"score\":\"" << sn << "\",\"sparsity\":" << probe.sparsity << ",\"tokens\":" << n
         << ",\"top1\":" << (double)top1 / n << ",\"top5\":" << (double)top5 / n << ",\"kl_mean\":" << kl_sum / n
         << ",\"kl_p50\":" << kls[n / 2] << ",\"kl_p95\":" << kls[(size_t)(n * 0.95)]
         << ",\"achieved\":" << (probe.total ? (double)probe.zeroed / probe.total : 0.0) << ",\"by_domain\":{";
      bool first = true;
      for (auto &kv : by_dom) {
        js << (first ? "" : ",") << "\"" << kv.first << "\":" << (double)kv.second.second / kv.second.first;
        first = false;
      }
      js << "}}";
      res << js.str() << std::endl;
      printf("%s\n", js.str().c_str());
      fflush(stdout);
    }
    fclose(flog);
  }
  if (probe.save) fclose(probe.save);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}

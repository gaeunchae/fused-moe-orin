| Engine | GB read per token | Decode tok/s, context ~0 | Decode tok/s, context 4096 | % of 180 GB/s (context ~0) | vs ours (context ~0 / 4096) |
|---|---:|---:|---:|---:|---:|
| llama.cpp master e300919 (Q4_K_M) | 1.92 | 48.0 (47.8-48.1) | 42.9 (42.8-43.0) | 51.2% | -21.8% / -19.6% |
| vLLM 0.20.0 (GPTQ INT4, Marlin MoE) | 2.06 | 59.3 (59.3-59.4) | 52.2 (52.2-52.2) | 67.9% | -3.3% / -2.3% |
| TensorRT Edge-LLM 0.6.0 (GPTQ INT4) | 2.06 | 59.8 (59.7-59.8) | 51.6 (51.6-51.8) | 68.4% | -2.6% / -3.3% |
| llama.cpp + our patch (Q4_K_M) | 1.92 | 61.4 (61.4-61.4) | 53.4 (53.3-53.5) | 65.4% | +0.0% / +0.0% |

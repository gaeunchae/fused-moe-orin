# Orin MoE 디코드 최적화: 결과 정리

2026-11-22 기준. 측정 절차와 도구는 [PROFILING.md](PROFILING.md)에 있다.

## 1. 요약

Jetson AGX Orin에서 batch-1 MoE 디코드의 병목은 DRAM 대역폭이 아니었다. GEMV kernel이 짧은 expert 행을 처리하는 **block 처리 속도**가 병목이었다. block 모양을 바이트 기준으로 정하자 Qwen3-30B-A3B가 43.4 tok/s에서 61.8 tok/s로 42% 빨라졌다.

- 원본 43.4 tok/s는 이론 상한 94 tok/s의 46%다. 가중치는 낭비 없이 정확히 한 번씩 읽지만, GEMV는 대역폭의 절반 정도만 쓴다.
- 원인은 block 하나가 맡는 바이트다. expert 행이 432~1,152 B로 작다. 원래 모양인 "block 하나에 행 하나"로는 SM이 메모리 요청을 동시에 충분히 내지 못한다.
- 같은 규칙이 다른 모델 3개와 양자화 형식 13개에서도 성립했다.
  - 모델별로 Qwen3.5-35B-A3B +35%, OLMoE-1B-7B +40%, dense Qwen3-8B +15%다.
  - 형식별 GEMV는 원본 대비 중앙값 1.28배(행렬 하나), 1.64배(gate+up 합친 kernel)다.
- **최신 llama.cpp master 대비로도 10~28% 빠르다.** master에는 2026-11-20에 Orin 전용 GEMV 설정이 들어왔지만, 합친 gate+up kernel과 짧은 행은 개선되지 않았다(7.1절). 최종 결과는 Qwen3-30B-A3B 기준으로 master 48.3에서 우리 61.6 tok/s다.
- llama.cpp 테스트 4,805개(최신 master에서는 5,411개)를 통과했다. logits는 원본과 top-1 98.8%로 일치하고, perplexity는 측정 오차 안에서 같다.

## 2. 문제와 측정 환경

| 항목 | 값 |
|---|---|
| 하드웨어 | Jetson AGX Orin 64 GB, sm_87, SM 16개, L2 4 MB, JetPack R36.5, CUDA 12.6 |
| 클럭 | `jetson_clocks`로 고정: GPU 1300.5 MHz, EMC 3199 MHz |
| 측정한 최대 읽기 대역폭 | 180 GB/s |
| SM 한도 | block 16개, warp 48개, 레지스터 65,536개 |
| 주 모델 | Qwen3-30B-A3B Q4_K_M, 48층, expert 128개 중 8개 사용, expert 행렬 768 × 2048 |
| 토큰당 읽는 가중치 | 1.92 GB이므로 바닥 시간은 10.7 ms, 상한은 94 tok/s |
| 기준 llama.cpp | `13ec8f6` (2026-11-04 master) |
| 속도 측정 | `llama-bench -p 0 -n 64 -r 5`, 컨텍스트 0 / 1024 / 4096, 5회 반복의 표준편차 0.1~0.3% |

## 3. 병목 프로파일링

원본은 토큰 하나에서 데이터 전송 바닥 시간 10.7 ms 위로 14.2 ms를 더 썼다. 그중 10.3 ms가 GEMV kernel 본체였다(nsys kernel 단위 추적).

| 바닥 위 시간 (ms/토큰) | 원본 | 최종 (전용 kernel 포함) |
|---|---:|---:|
| 양자화 GEMV | 10.3 | 2.7 |
| 작은 연산 (양자화, norm, rope, top-k 등) | 1.9 | 1.9 |
| attention 고정 비용 | 0.9 | 1.0 |
| 호스트 | 1.0 | 0.9 |
| 합계 | 14.2 | 6.4 |

- **launch 비용은 주원인이 아니었다.** 원본 llama.cpp는 이미 CUDA Graph를 쓰고, top-k와 gate+up과 가중합도 합쳐 실행한다(토큰당 kernel 1,110개).
- **ncu 결과, 모든 GEMV는 필요한 가중치 바이트만 DRAM에서 읽는다(비율 1.00~1.02).** 바이트 수가 문제가 아니라, 같은 바이트를 읽는 속도가 문제였다.
- **원본 expert GEMV의 대역폭 효율은 47~50%다.** gate+up 50%, down Q4_K 49%, down Q6_K 47%다.

## 4. 핵심 관찰: 짧은 행은 block 처리 속도에 막힌다

### 4.1 행 길이가 아니라 block당 바이트가 속도를 정한다

같은 Q4_K 가중치 7.08 MB를 행 길이만 바꿔 자르고, block 모양 75가지를 3번씩 쟀다(`bench/gemv_shape.cu` 섹션 D, 실행 간 차이 중앙값 0.4%).

| 행 길이 (열) | 해당 텐서 | 원래 모양 (4 warp, 행 1개) | 가장 좋은 모양 |
|---:|---|---:|---:|
| 512 | Qwen3.5 down | 29.6 GB/s | 139.8 GB/s |
| 768 | Qwen3 down | 36.2 | 136.0 |
| 1024 | OLMoE down | 55.2 | 142.1 |
| 2048 | gate/up | 86.0 | 142.8 |
| 4096 | 큰 dense 행 | 126.6 | 141.2 |

행 길이와 block 크기가 달라도 block당 바이트가 같으면 속도가 같다. 그리고 2.3 KB 근처에서 포화한다.

| block당 바이트 | 행 512, warp 1개 | 행 1024, warp 2개 |
|---:|---:|---:|
| 576 B | 82.6 GB/s | 81.8 GB/s |
| 1,152 B | 119.9 | 121.8 |
| 2,304 B | 140.1 | 139.4 |

### 4.2 SM 수를 줄이는 실험: 작은 block에서는 SM이 병목이다

green context(CUDA 12.4+)로 kernel이 쓰는 SM만 16, 12, 8, 6, 4개로 줄였다. 클럭과 DRAM은 그대로 두었다. 실제로 쓰인 SM 수는 매번 세어서 확인했다.

| block당 바이트 | SM 16개 | 12개 | 8개 | 6개 | 4개 |
|---|---:|---:|---:|---:|---:|
| 288 B: 전체 GB/s | 49.9 | 32.9 | 25.2 | 18.7 | 12.7 |
| 288 B: SM 하나당 GB/s | **3.1** | 2.7 | **3.2** | 3.1 | **3.2** |
| 4,608 B: 전체 GB/s | 140.2 | 131.4 | 96.9 | 76.3 | 53.0 |
| 4,608 B: SM 하나당 GB/s | 8.8 | 11.0 | 12.1 | 12.7 | 13.3 |

- **작은 block(288 B):** SM이 몇 개든 SM 하나당 3.1 GB/s로 같다. DRAM이 한계라면 이렇게 SM 수에 비례하지 않는다. 그러므로 SM 쪽의 block 처리 속도가 병목이다.
- **큰 block(4.6 KB):** SM 16개에서는 SM 하나당 속도가 오히려 낮다. 공유하는 DRAM이 먼저 한계에 닿기 때문이다.
- **SM 하나에도 상한이 있다(약 13~18 GB/s).** SM 16개의 처리 능력은 210~290 GB/s로, DRAM 180 GB/s의 1.2~1.6배밖에 안 된다. 그래서 block 오버헤드로 SM 시간을 조금만 낭비해도 바로 대역폭 손실로 이어진다.

"엣지 GPU라서"보다는 **"SM 수에 비해 메모리 대역폭이 큰 GPU에서"**라고 쓰는 게 정확하다. 아래는 측정이 아니라 스펙으로 계산한 값이다.

| GPU | SM당 block 한도 | DRAM 바이트 / SM / 클럭 |
|---|---:|---:|
| RTX 4090 | 24 | 약 3 |
| RTX 3090 | 16 | 약 7 |
| Orin AGX | 16 | 약 9~10 |
| H100 SXM | 32 | 약 13 |

## 5. 일반화 검증

### 5.1 실제 모델 4개

원본 kernel의 효율은 모델과 상관없이 행 길이 순서대로 정렬된다. 수정본은 행 길이와 관계없이 76~89%다(nsys kernel 시간, 180 GB/s 대비).

| 행 길이 | kernel | 원본 | 수정본 |
|---:|---|---:|---:|
| 512 | Qwen3.5 expert down Q4_K / Q6_K | 38% / 36% | 76% / 83% |
| 768 | Qwen3 expert down Q4_K / Q6_K | 49% / 47% | 83% / 86% |
| 1024 | OLMoE expert down Q4_K / Q6_K | 61% / 59% | 85% / 88% |
| 12288 | Qwen3-8B dense down Q6_K | 87% | 89% |
| 2048 | gate+up (Qwen3 / OLMoE / Qwen3.5) | 50% / 51% / 49% | 83% / 84% / 69% |

- Qwen3.5는 down의 개선 폭이 가장 컸다(36%에서 83%). 하지만 전체 속도 개선률은 +35%로 가장 작다. 어휘가 248k라서 출력층이 GPU 시간의 13%를 차지하고, 선형 attention 층도 30개 있기 때문이다.
- Qwen3.5의 gate+up은 shared expert를 9번째 expert로 합쳐서 계산한다. 그래서 gate+up 전용 kernel이 적용되지 않는다(69%).

### 5.2 양자화 형식 13개

llama.cpp의 실제 MoE GEMV(MUL_MAT_ID)를 형식 13개 × 행 길이 5개로 쟀다(`bench/mmid_formats.cpp`). 아래는 원본에서 최종 규칙으로의 변화이고, 대역폭 효율(%)이다.

| 형식 | 행렬 하나 (down 모양): 512 / 1024 / 2048 / 4096 | gate+up 합친 kernel: 512 / 1024 / 2048 / 4096 |
|---|---|---|
| Q8_0 | 86→86 / **48→91** / 59→93 / 91→94 | 43→70 / 55→82 / 72→90 / 89→93 |
| Q4_0 | 52→79 / 86→87 / **52→91** / 73→92 | 25→47 / 44→71 / 55→82 / 76→88 |
| Q4_K | 43→70 / 73→80 / 49→84 / 73→85 | 17→44 / 32→67 / 54→77 / 68→82 |
| Q6_K | 42→80 / 69→85 / 68→88 / 84→89 | 20→59 / 37→68 / 63→82 / 80→88 |
| MXFP4 | 34→58 / 56→72 / 55→84 / 69→88 | 16→40 / 29→62 / 52→73 / 69→81 |
| IQ4_XS | 25→49 / 48→85 / 72→90 / 71→94 | 12→38 / 24→66 / 44→82 / 84→93 |
| Q2_K | 21→38 / 41→71 / 68→85 / 72→90 | 10→25 / 20→48 / 36→66 / 65→80 |
| Q3_K | 15→38 / 30→68 / 50→69 / 56→72 | 8→30 / 14→54 / 27→58 / 57→66 |

- **65개 조합 모두에서 원본보다 빠르다.** 행렬 하나는 중앙값 1.28배(최대 2.54배), 합친 kernel은 중앙값 1.64배(최대 3.97배)다. 측정 프로그램의 시간에는 입력 양자화(약 2 µs)가 포함돼 있다.
- **원본 규칙에는 "절벽"이 있다.** 굵게 표시한 칸처럼 행이 더 긴데 효율이 절반으로 떨어진다. 원본은 "warp 4개가 한 번에 행을 덮는가"로 block이 맡을 행 수(4개 또는 1개)를 정한다. 그래서 그 경계에서 block당 바이트가 1/4로 줄어든다.
- **저비트 형식일수록 문제가 크다.** 행 하나의 바이트가 더 작기 때문이다. 엣지 사용자가 메모리에 맞추려고 고르는 형식에서 가장 심하다.

## 6. 최적화 방법

| 방법 | 바꾼 것 | 근거 | 효과 (Qwen3, 컨텍스트 0) |
|---|---|---|---|
| byte-balanced v1 | Q4_K·Q6_K의 일반 GEMV를 warp 2개, 행 2개(짧은 행은 4개)로 | block당 바이트가 너무 작음 | 43.4 → 58.1 tok/s |
| gate+up 전용 kernel | warp 4개, 행 2개. 64 block 뒤에 실행될 block의 행을 L2로 미리 읽음 | ncu: 메모리 대기 66%, occupancy 상한 67% | 약 +3% |
| down 전용 kernel | thread 16개가 행 하나를 맡음. shared memory와 `__syncthreads` 없음 | ncu: 명령 발행 한계, 레지스터 상한 | 약 +3% |
| 바이트 기준 규칙 (최종) | 형식 목록 없이 모든 형식의 block 모양을 정함 | 5.2절 | bb_v1과 같거나 +3% |

### 6.1 바이트 기준 규칙과 세 번의 수정

최종 규칙은 토큰 1개 GEMV에 적용한다(`GGML_CUDA_MMVQ_BYTES=8192`).

1. **warp 수:** 행 하나에 thread 32개 이하가 필요하면 1개, 그보다 많으면 2개다.
2. **행 수:** block이 읽는 바이트가 8 KB에 이를 때까지 1, 2, 4로 늘린다. gate와 up을 함께 읽으면 두 행렬을 모두 센다.
3. **누적값 상한:** thread 하나가 들고 있는 부분합은 4개 이하다. 합친 kernel 템플릿은 gate가 없어도 행마다 2개로 센다.

이 규칙은 세 번 고쳐서 나왔다.

1. **v1 (목표 4 KB, warp는 행을 한 번에 덮을 만큼):** 형식 측정에서는 중앙값 1.28배였다. 하지만 실제 모델에서는 bb_v1보다 4~12% 느렸다. nsys로 보니 gate+up 합친 kernel이 108 µs에서 150 µs로 느려졌다. 규칙을 만들 때 쓴 측정이 **행렬 하나짜리 kernel만** 쟀기 때문이다.
2. **v2 (합친 kernel 측정 추가):** 모양 12가지 × 두 변형 × 65개 조합을 쟀다. 원인은 레지스터였다. 합친 kernel의 4×4 모양은 thread당 레지스터가 96개라서 SM에 block이 5개만 올라간다. 2×2는 64개라서 16개가 올라간다. 그래서 누적값 상한을 넣고 매개변수를 다시 골랐다(최고 모양 대비 기하평균 96.6%). 그래도 실제 모델 Qwen3에서 −2.2%가 남았다.
3. **v3 (최종):** 남은 손해는 두 kernel에서 나왔다.
   - gate 없이 bias만 합친 attention GEMV: 합친 템플릿은 gate가 없어도 gate용 레지스터를 잡는다. 그래서 누적값을 **템플릿 기준**으로 세도록 바꿨다.
   - Q6_K 출력층: warp를 최대 2개로 제한해서 해결했다.

   그 결과 최고 모양 대비 중앙값 99.6%(행렬 하나), 99.1%(합친 kernel)가 됐고, 어느 모델에서도 bb_v1보다 느리지 않다.

## 7. 최종 결과

tok/s, 컨텍스트 0. 원본은 2026-11-19, 나머지는 2026-11-21에 같은 조건으로 측정했다. 원본은 세 번 측정해서 43.36 / 43.39 / 43.43이 나왔다.

| 모델 | 원본 | bb_v1 | bb_all | 새 규칙 | 새 규칙 + 전용 kernel | 원본 대비 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 43.39 | 58.05 | **61.77** | 58.27 | 61.45 | +42% |
| Qwen3.5-35B-A3B | 39.56 | 51.05 | 53.29 | 52.65 | **53.46** | +35% |
| OLMoE-1B-7B | 115.2 | 148.2 | **161.1** | 153.1 | 161.0 | +40% |
| Qwen3-8B (dense) | 26.44 | **30.46** | 30.45 | 30.41 | 30.43 | +15% |

- **컨텍스트 4096에서의 개선률:** 새 규칙 + 전용 kernel 기준으로 Qwen3 +37%, Qwen3.5 +34%, OLMoE +27%, Qwen3-8B +14%다. OLMoE는 KV cache가 커서 컨텍스트가 길수록 attention 비중이 커진다.
- **새 규칙 + 전용 kernel과 bb_all의 차이:** −0.5%~+0.3%라서 같은 수준으로 본다.
- **정확성:**
  - test-backend-ops의 MUL_MAT 2,490개, MUL_MAT_ID 937개, MUL_MAT_VEC_FUSION 1,378개를 모두 통과했다(규칙 목표 2 KB, 4 KB, 8 KB).
  - logits(28개 프롬프트, 1,792 토큰): 최종 설정은 원본 대비 top-1 98.83%, KL 0.00227이고, bb_all과는 비트 단위까지 같다. 원본끼리 실행 경로만 달라도 98.60%, 0.0020이다. 차이는 부동소수점 덧셈 순서에서 온다.
  - perplexity(wikitext-2, 512토큰 20개, `-ub 1`로 토큰 하나씩 처리해서 바뀐 경로를 지나게 함)는 측정 오차 안에서 같다. Qwen3 9.845 → 9.865(±0.43), Qwen3.5 6.709 → 6.715(±0.23), OLMoE 12.185 → 12.173(±0.55).

### 7.1 최신 master(e300919) 대비

2026-11-20 llama.cpp에 **Orin 전용 GEMV 설정**이 들어왔다(`ea84d70`, #31268, Orin Nano에서 측정). 토큰 1개 GEMV에서 warp 하나가 행 하나를 맡게 해서, 9개 형식의 block이 4 warp × 4행이 된다. 우리 patch를 이 master로 옮기고(byte-balanced v1은 빼고, 규칙과 전용 kernel을 ORIN 설정 표에서도 켜지게 함) 다시 쟀다. 측정은 2026-11-22에 했다.

| 모델 (tok/s, 컨텍스트 0) | 13ec8f6 원본 | 최신 master | master + 규칙 | master + 규칙 + 전용 kernel | master 대비 |
|---|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 43.39 | 48.27 | 58.41 | **61.64** | +28% |
| Qwen3.5-35B-A3B | 39.56 | 45.80 | 53.02 | **53.75** | +17% |
| OLMoE-1B-7B | 115.2 | 129.2 | 154.8 | **163.1** | +26% |
| Qwen3-8B (dense) | 26.44 | 27.76 | 30.45 | **30.49** | +10% |

- **컨텍스트 4096에서 master 대비:** +24%, +17%, +18%, +9%다.
- **테스트:** master에서 MUL_MAT 2,724개, MUL_MAT_ID 1,083개, MUL_MAT_FUSION 1,604개(총 5,411개)를 통과했다. 규칙만 켠 설정과 전용 kernel까지 켠 설정 모두에서다.
- **형식 측정 (master 대비):** 행렬 하나는 중앙값 1.22배이고 가장 나빠도 0.99배다. 절벽은 master가 이미 고쳤다. 합친 kernel은 65개 조합 모두 빨라서 중앙값 1.64배, 최소 1.02배다.

master 대비 이득은 upstream 설정이 남겨 둔 두 곳에서 나온다(nsys kernel 시간).

| kernel | master | 우리 규칙 | 차이 |
|---|---|---|---:|
| Qwen3 gate+up (합친 kernel) | 4×4, 148.8 µs | 2×2, 107.7 µs | −28% |
| Qwen3 expert down Q6_K / Q4_K (짧은 행) | 120.0 / 79.0 µs | 79.6 / 55.7 µs | −34% / −29% |
| Qwen3-8B gate+up (합친 kernel) | 4×4, 472.8 µs | 2×2, 381.1 µs | −19% |
| 행렬 하나·bias만 합친 긴 행 (attention, dense down, 출력층) | | | −6% ~ +2% |

- **합친 kernel:** upstream의 4×4는 thread당 레지스터가 96개라서 SM에 block이 5개만 올라간다(6.1절의 v2에서 찾은 문제와 같다).
- **짧은 행:** upstream은 small_k 경로(4 warp × 4행)를 바꾸지 않았다.
- **긴 행렬 하나:** upstream과 우리 모두 "block당 바이트를 늘린다"는 같은 방향이라 결과가 비슷하다.

### 7.2 RTX 3090에서의 재현 (2026-11-23)

같은 master(e300919)와 patch를 RTX 3090에서 쟀다. 측정 환경은 SM 82개, 측정한 최대 읽기 대역폭 893 GB/s, CUDA 12.8이고, 클럭은 측정 내내 1695 MHz였다. 결과는 `/media/vlsi/NVMESSD/fused-moe/profiling/rtx3090/`에 있다. 3090과 Orin은 SM당 block·warp·레지스터 한도가 같고, SM 하나가 감당하는 대역폭도 거의 같다(3090 10.9, Orin 11.3 GB/s).

**kernel 단위에서는 같은 현상이 나온다.**

| 행 길이 (Q4_K, 같은 7.08 MB) | 원래 모양 | 가장 좋은 모양 |
|---|---:|---:|
| 512 | 246 GB/s (최대의 28%) | 665 GB/s (74%) |
| 1024 | 417 (47%) | 669 (75%) |
| 2048 | 611 (68%) | 672 (75%) |
| 4096 | 664 (74%) | 664 (74%) |

- **Orin보다는 덜하다.** 예를 들어 행 2048에서 원래 모양이 Orin은 48%, 3090은 68%다. 3090의 SM이 클럭당 끌어와야 하는 바이트가 6.7로 Orin의 8.7보다 작다는 예측과 맞는다.
- **SM 수 실험:** 288 B block에서 SM 하나당 속도가 SM 4~20개 사이에서 5.3~5.5 GB/s로 일정하다. SM 쪽이 병목이라는 뜻이다. SM 82개를 다 쓰면 4.3으로 22% 떨어지는데, 원인은 확인하지 않았다(Orin에는 없던 현상이다).
- **형식 측정 (master 대비):** 행렬 하나는 중앙값 1.03배(최대 1.92배, 최악 0.97배), 합친 kernel은 중앙값 1.32배(최대 3.00배)다. 모양 sweep에서 우리 규칙은 최고 모양 대비 98.7%(행렬 하나), 100%(합친 kernel)다. 같은 기준으로 master는 96.2%, 76.0%다. MoE GEMV에서는 Orin에서 정한 매개변수가 3090에서도 그대로 맞는다.

**전체 속도 이득은 작다 (tok/s, 컨텍스트 0).** 두 번 측정했다. 1차는 `run.sh`, 2차는 SM-aware 실험 `run_exp.sh`다. %는 같은 회차 처음에 잰 master 대비다.

| 모델 | master (1차) | 규칙 + 전용 kernel (1차) | SM-aware 규칙만 (2차) | SM-aware + 전용 kernel (2차) |
|---|---:|---:|---:|---:|
| Qwen3-30B-A3B | 195.7 | 200.7 (+2.5%) | 201.5 (+2.9%) | 202.5 (+3.4%) |
| Qwen3.5-35B-A3B | 173.4 | 177.2 (+2.2%) | 179.3 (+3.4%) | 179.4 (+3.4%) |
| OLMoE-1B-7B | 545.3 | 562.8 (+3.2%) | 560.1 (+2.4%) | 563.3 (+3.0%) |
| Qwen3-8B (dense) | 141.5 | 139.7 (−1.3%) | 141.2 (−0.3%) | 141.0 (−0.4%) |

2차의 master는 처음 195.9 / 173.5 / 546.9 / 141.6, 끝에서 195.9 / 173.2 / 546.4 / 140.2였다. Qwen3-8B는 1% 흔들렸다.

- **이득이 작은 이유:** 3090은 토큰 하나가 약 5 ms다. 그래서 작은 연산, flash attention, kernel launch 같은 고정 비용이 차지하는 비중이 크다. GEMV kernel 자체는 확실히 빨라졌다(Qwen3 gate+up −9%, down Q6_K −20%, Q4_K −15%).
- **dense의 −1.3%를 설명하는 후보는 두 가지이고, 아직 둘을 떼어 내지 못했다.**
  - **나눠 처리 (H1):** 4096열 이상의 긴 행을 warp 2개가 여러 번 나눠 처리한다.
  - **GPU를 못 채움 (H2):** 행 1024개짜리 행렬(attention k/v)에 block당 4행을 주면 block이 256개뿐이라, SM 82개를 다 채우지 못한다. 이 kernel은 4.9 µs에서 5.5 µs가 됐다. Orin은 SM이 16개라 같은 256개로도 SM당 16개가 된다.
  - **손잡이를 하나씩 켠 결과(아래)는 H1 쪽을 가리킨다.** warp 4만 켜면 −1.3%가 −0.2%가 되지만, block 조건만 켜면 −1.0%에 그친다. 다만 kernel 시간을 손잡이별로 따로 잰 자료가 없어서 확정하지 못했다(9절의 분리 실험).
- **SM-aware 변형:** 바이트 규칙 위에 손잡이 두 개를 얹었다(`patches/sm_knobs_on_top_of_bytes_rule.patch`, `mmvq.cu`만 바꿈).
  - `GGML_CUDA_MMVQ_MAXNW=4`: 행 하나가 lane 64개를 넘게 필요하면 warp를 2개 대신 4개로 쓴다.
  - `GGML_CUDA_MMVQ_MINBLK=n`: 규칙대로 행 수를 정한 뒤, grid의 block 수(행 방향 block 수 × (채널 수 + shared expert 여부) × 샘플 수)가 n × SM 수보다 적으면 행 수를 절반씩 줄인다. SM 수는 `cudaDeviceProp.multiProcessorCount`다. n=4이면 3090은 328개, Orin은 64개다. warp 수는 바꾸지 않는다.
  - 바이트 규칙과 같은 13개 형식, GENERIC/ORIN 설정 표에서만 동작한다. MoE 전용 kernel에는 영향이 없다. `GGML_CUDA_MMVQ_SHAPE`로 모양을 강제하면 그게 먼저다.
  - 테스트 5,411개를 두 설정(8 KB·4 warp·4 block, 2 KB·4 warp·8 block)에서 통과했다.
- **손잡이를 하나씩 켠 결과** (2차 측정, 컨텍스트 0, 같은 회차 처음의 master 대비. 반복 측정 표준편차는 0.2~0.7%, master는 회차 안에서 최대 1% 흔들렸다). 값 4와 8은 원리에서 정한 게 아니라 후보로 넣어 본 값이고, warp 4는 미리 컴파일된 조합(1, 2, 4) 중 남은 선택지다.

| 설정 | Qwen3-8B | Qwen3-30B | Qwen3.5 | OLMoE |
|---|---:|---:|---:|---:|
| 규칙 그대로 (r8) | −1.3% | +2.0% | +2.5% | +2.3% |
| + warp 4만 | −0.2% | +2.7% | +3.2% | +2.3% |
| + SM당 block 4만 | −1.0% | +2.2% | +3.1% | +2.2% |
| + 둘 다 | −0.3% | +2.9% | +3.4% | +2.4% |
| + warp 4, SM당 block 8 | −0.3% | +2.9% | +3.5% | +2.7% |

- **dense 손해는 대부분 줄었지만 다 없어지지는 않았다.**
  - 디코드는 −1.3%에서 −0.3%가 됐다. 다만 같은 회차에서 master가 처음과 끝에 1% 차이 나서, 디코드 숫자만으로는 확정할 수 없다.
  - nsys GPU kernel 시간은 master 대비 +1.3%에서 +0.4%가 됐다(234.1 → 237.1 → 235.1 ms). 0.4%가 남아 있다.
  - 남은 손해는 8 KB 목표 때문에 행 길이 4096 행렬이 block당 4행으로 묶이는 kernel에서 나온다(13.5 µs, master 13.1 µs). 3090 모양 sweep에서도 목표 4 KB가 최고 모양 대비 99.9%로, 8 KB(99.3%)보다 조금 나았다.
- **MoE 이득:** 이 변형으로 규칙만 쓰면 +2.4~3.4%, 전용 kernel까지 켜면 +3.0~3.4%다.
- **아직 열린 질문:** 규칙에 GPU마다 다른 입력이 필요한지, 필요하다면 무엇인지(SM 수에 따른 block 수 조건인지, warp 상한인지)는 분리 실험 뒤에 정한다.

### 7.3 분리 실험: 설계와 예측 (2026-11-23, 측정 전에 기록)

> **범위 결정 (2026-11-23):** 7.3~7.5절은 dense 모델(Qwen3-8B)의 3090 손해를 다룬다. 프로젝트 주제가 MoE이므로 보고서와 발표에서는 이 내용을 빼고, 보고서 한계 항목에 dense 손해 한 줄만 남긴다. 기록은 여기 그대로 둔다.

7.2절의 dense 손해에 대해 두 원인 후보와 하나의 부가 질문을 떼어 내서 확인한다. Orin과 3090에서 똑같이 잰다.

| 가설 | 내용 | 확인 방법 |
|---|---|---|
| H1 (나눠 처리) | 행 하나에 lane이 64개 넘게 필요할 때, warp 2개로 여러 번 나눠 처리하면 warp 4개보다 느리다 | block 수가 같은 두 모양 비교 (2×4 대 4×4) |
| H2 (GPU를 못 채움) | block 수가 한 번에 GPU를 채우는 양(SM 수 × SM당 동시 block 수)보다 적으면 느려진다. 손해는 "몇 번에 걸쳐 채우는가(wave 수)"로 정해진다 | 나눠 처리 횟수가 같은 두 모양 비교 (2×1 대 2×4), 행 수를 바꿔 wave 수에 따른 효율 |
| H3 (레지스터) | 합친 kernel 템플릿에서는 warp 4가 레지스터 때문에 손해다. 그렇다면 warp 상한은 GPU가 아니라 kernel 종류로 정할 수 있다 | 합친 변형에서 2×2 대 4×2 |

**도구:**
- `bench/dense_shapes.cpp`: dense MUL_MAT, 행 256~16384 × 열 2048·4096·12288, Q4_K·Q6_K·Q8_0, 행렬 하나와 합친 것
- `scripts/dense_sweep.sh`: nsys로 kernel 시간과 실제 실행된 warp·block 수를 기록하고 `dense_extract.py`로 정리
- `scripts/model_ablation.sh`: 실제 모델 4개. 손잡이 하나씩, `MINBLK` 1~16. 설정을 번갈아 3바퀴 돈다

**예측:**
- **P1:** 행렬 하나·4096열 이상에서 block 수가 같으면 warp 4가 warp 2보다 빠르다. 3090에서 5~15%, Orin에서는 ±3% 안이다.
- **P2:** 모양을 고정하면 효율은 block 수와 함께 오르다가 약 2 wave 이상에서 평평해진다. 가로축을 wave 수로 두면 Orin과 3090의 곡선이 10%p 안에서 겹친다.
- **P3:** 합친 kernel에서 warp 4는 Orin에서 warp 2보다 10% 이상 느리고, 3090에서는 ±3% 안이다.
- **P4:** 실제 모델에서 3090의 dense 손해는 대부분 `MAXNW=4`로 회복된다. `MINBLK` 값(1~16)은 두 GPU 모두에서 디코드를 0.5% 넘게 바꾸지 않는다.
- **P5:** Orin에서 `MAXNW=4`를 켜면 합친 kernel이 느려져서, MoE 모델 디코드가 규칙 그대로일 때보다 낮아진다.

시험 측정에서 이미 본 것: Orin master(4×4)도 행 256(64 block)에서 52%, 행 4096(1024 block)에서 85%였다. 작은 행렬에서는 Orin에서도 무언가가 작용한다. 그것이 block 수 때문인지, kernel 하나마다 드는 고정 비용 때문인지는 위 비교로 가린다.

### 7.4 분리 실험: 결과와 판정 (2026-11-23)

두 GPU 모두 dense 측정 2,142개(nsys kernel 시간), 모델 측정 108개(설정 9개 × 모델 4개 × 3바퀴)를 얻었다. 3바퀴 사이 흔들림은 0.0~0.3%였다. 분석은 `scripts/ablation_analysis.py <폴더> <SM 수>`로 하고, 결과는 `profiling/ablation_orin/`과 `profiling/rtx3090/ablation/`에 있다. 손잡이를 켠 상태의 테스트 5,411개도 Orin에서 통과했다.

**H1 (나눠 처리): 기각.** 큰 행렬(8192행 이상)에서 행/block을 같게 두고 warp 2와 4를 비교하면, 시간 비(warp 2 / warp 4)가 Orin 0.98~1.01, 3090 1.00~1.02다. 여러 번 나눠 처리하는 것 자체는 거의 비용이 없다. 예측 P1은 Orin에서 맞고 3090에서 틀렸다(5~15%로 예측).

**H2 (GPU를 못 채움): 채택.** 모양마다 가장 큰 행렬의 효율을 100으로 두고 wave 수별로 보면 이렇다.

| wave 수 | 0.5 미만 | 0.5~1 | 1~2 | 2~4 | 4~8 | 8 이상 |
|---|---:|---:|---:|---:|---:|---:|
| Orin, 행렬 하나 | 52% | 70% | 80% | 88% | 94% | 99% |
| 3090, 행렬 하나 | 38% | 78% | 88% | 94% | 98% | 100% |
| Orin, 합친 kernel | 30% | 57% | 83% | 90% | 95% | 100% |
| 3090, 합친 kernel | 32% | 80% | 86% | 91% | 97% | 100% |

- 두 GPU 모두 약 2~4 wave 아래에서 효율이 떨어진다.
- 예측 P2(두 곡선이 10%p 안에서 겹친다)는 12칸 중 10칸에서 맞았다. 어긋난 곳은 가장 작은 구간(행렬 하나 0.5 미만 14%p, 합친 kernel 0.5~1 23%p)이다.
- block 수만 4배 줄이면(행/block 1 → 4, 나눠 처리 횟수는 같음) 256행 행렬에서 느려진다. Orin은 행렬 하나 1.02~1.19배, 합친 kernel 1.18~1.78배다. 3090은 행렬 하나 1.08~1.22배, 합친 kernel 1.62~2.02배다.
- SM이 많은 3090에서는 같은 행렬도 wave 수가 Orin의 약 1/5이다. 그래서 같은 모양이 3090에서 "못 채움" 구간에 더 자주 들어간다.

**H3 (합친 kernel의 warp 4): GPU에 따라 다르다.** 큰 행렬에서 시간 비(warp 2 / warp 4)는 Orin 0.88~0.97(warp 2가 3~12% 빠름), 3090 0.99~1.01(차이 없음)이다. 예측 P3은 방향이 맞았지만, Orin의 크기는 예측(10% 이상)보다 작은 경우가 있다.

**실제 모델 (master 대비, 3바퀴 중앙값, 전용 MoE kernel은 끔)**

| 설정 | Orin Qwen3-8B | Orin Qwen3-30B | Orin Qwen3.5 | Orin OLMoE | 3090 Qwen3-8B | 3090 Qwen3-30B | 3090 Qwen3.5 | 3090 OLMoE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 규칙 그대로 (r8) | +10.0% | +21.2% | +15.4% | +20.7% | −1.5% | +2.2% | +2.2% | +2.0% |
| + warp 4 | +6.0% | +13.7% | +11.5% | +12.8% | −0.4% | +2.8% | +3.1% | +2.0% |
| + SM당 block 4 | +10.0% | +21.2% | +15.7% | +20.2% | −1.2% | +1.9% | +3.0% | +1.9% |
| + 둘 다 | +5.8% | +13.6% | +11.8% | +12.9% | −0.4% | +2.7% | +3.3% | +2.1% |
| + warp 4 + block 16 | +5.9% | +13.2% | +11.8% | +12.7% | −0.2% | +2.2% | +2.7% | +0.9% |

- **warp 4는 Orin에서 4~8%p 손해, 3090에서 0.6~1.1%p 이득이다(예측 P5 맞음).** Orin에서 느려진 kernel은 합친 템플릿(Qwen3-30B gate+up 107.6 → 127.8 µs, Qwen3-8B gate+up 381.6 → 417.2 µs)과 Q6_K 출력층(1,543 → 1,630 µs)이다.
- **SM당 block 조건은 Orin에서 효과가 없다(값 1~16 모두 같음).** 3090에서는 1~8 사이 차이가 0.5%p 이하이고, 16이면 OLMoE가 1.5%p 떨어진다. 예측 P4는 Orin에서 맞고, 3090에서는 16을 빼면 맞다.

**결론**
1. 3090의 dense 손해의 원인은 **GPU를 채우지 못한 것(H2)**이다. "나눠 처리(H1)"가 아니다. warp 4가 3090에서 도운 것도, block이 적을 때 block당 warp를 늘려 동시에 떠 있는 일을 늘렸기 때문으로 해석된다.
2. **고정된 warp 상한 하나로 두 GPU를 다 맞출 수는 없다.** Orin은 합친 kernel의 레지스터 때문에 warp 2가 필요하다. 3090은 block이 적은 행렬에서 warp 4가 필요하다.
3. 그러니 규칙은 "행렬이 GPU를 몇 번에 걸쳐 채우는가(grid 크기 ÷ SM 수)"를 보고 모양을 골라야 한다. 이 값은 SM 수에 따라 달라진다. 다음 후보 규칙(v4)은 이 기준을 넣어, 이미 측정한 두 GPU의 sweep 데이터로 먼저 채점할 수 있다.

### 7.5 규칙 v4: 정의와 채점 기준 (2026-11-23, 채점 전에 기록)

**원리:** 7.4절에서 손해의 원인은 GPU를 못 채우는 것(H2)이었다. 반대로 GPU를 여러 번 채울 만큼 큰 행렬에서는 r8(Orin에서 고른 규칙)이 맞았다. 그러니 r8에서 출발하고, 행렬이 GPU를 충분히 채우지 못할 때만 모양을 바꾼다. 바꾸는 방향은 7.4절의 두 손잡이 그대로다. 먼저 block을 늘리고(행/block을 줄임), 그래도 모자라면 block당 warp를 늘린다.

**정의 (토큰 1개 GEMV, r8과 같은 13개 형식):**
1. r8로 warp 수와 행/block을 정한다(warp 1 또는 2, 8 KB 목표, thread당 누적값 4 이하).
2. wave 수를 계산한다.
   - 공식: wave = block 수 ÷ (SM 수 × 그 모양의 SM당 동시 block 수)
   - block 수: 행 방향 block 수 × (expert 수 또는 채널 수)
   - SM당 동시 block 수: kernel의 레지스터 수와 SM 한도(block 16, warp 48, 레지스터 65,536)로 계산한다. 구현할 때는 CUDA의 occupancy 함수로 물어본다.
3. wave가 **2 미만**인 동안 다음을 반복한다.
   - 행/block이 1보다 크면 절반으로 줄인다.
   - 행/block이 이미 1이고, 행 하나에 lane이 현재 warp 수 × 32보다 많이 필요하면 warp를 두 배로 늘린다(최대 4).
   - 둘 다 불가능하면 멈춘다.
4. 기준값 2는 7.4절 H2 표에서 정했다. 두 GPU 모두 2 wave 이상에서 행렬 하나 88~94%, 합친 kernel 86~91%였다. 민감도 확인용으로 1과 4도 함께 채점하지만, 판정은 2로 한다.

**채점 데이터 (새 측정 없음):**
- dense: 두 GPU의 강제 모양 12가지 sweep (`dense.csv`, 3개 형식 × 21개 크기 × 행렬 하나·합친 것)
- MoE: 두 GPU의 강제 모양 12가지 sweep (13개 형식 × 5개 행 길이 × 행렬 하나·합친 것, 출력 2048행 × expert 8개)

**성공 기준 (모두 만족해야 성공):**
- **S1:** 두 GPU, 두 데이터 모두에서 v4가 고른 모양이 최고 모양 대비 중앙값 97% 이상이다.
- **S2:** Orin에서 v4가 r8보다 3% 넘게 느린 조합이 없다. Orin에서 이미 검증된 r8을 망치지 않아야 한다.
- **S3:** 3090 dense에서 행 1024 이하 행렬의 중앙값 효율이 master 이상이다. 3090 손해가 난 바로 그 구간이다.

**실패하면:** 거기서 멈추고, 7.4절의 결론으로 보고서를 마무리한다.

**채점 결과 (`scripts/rule_v4_score.py`):** 각 칸은 최고 모양 대비 중앙값이다.

| | Orin dense 하나/합친 | Orin MoE 하나/합친 | 3090 dense 하나/합친 | 3090 MoE 하나/합친 |
|---|---|---|---|---|
| master | 98.8 / 82.0% | 79.1 / 64.1% (bb_v1) | 100.0 / 100.0% | 96.6 / 75.9% |
| r8 | 97.7 / 99.5% | 99.8 / 99.2% | 96.0 / 96.7% | 98.9 / 100.0% |
| **v4 (2 wave)** | **98.1 / 99.4%** | **99.8 / 99.2%** | **98.8 / 99.0%** | **98.9 / 100.0%** |

- **S1 통과:** 8칸 모두 97% 이상이다.
- **S3 통과:** 3090 dense의 1024행 이하에서 v4가 master와 같다(최고 대비 99.3%, 100%).
- **S2 실패:** Orin에서 v4가 r8보다 3% 넘게 느린 조합이 5개다. 모두 작은 합친 행렬(256~512행)이고 4~8% 느렸다. v4가 GPU를 채우려고 행/block을 2에서 1로 줄였거나(3개), warp를 2에서 4로 늘렸다(2개). Orin의 작은 합친 kernel에서는 wave 수보다 block당 바이트와 레지스터가 더 중요했다.
- **판정:** v4(2 wave)는 채택하지 않는다. 정해 둔 대로 7.4절의 결론으로 보고서를 마무리한다.
- **민감도 (판정에 쓰지 않음):** 기준 1 wave는 Orin에서 r8보다 느린 조합이 0개, 빠른 조합이 13개였다. 3090에서는 느린 7개, 빠른 46개였다. 기준 4 wave는 두 GPU 모두에서 더 나빴다. 1 wave는 결과를 본 뒤 고른 값이라, 채택하려면 이 sweep에 없는 데이터(실제 모델, 두 GPU)로 따로 검증해야 한다.

## 8. 배운 점과 한계

### 8.1 틀린 예측과 실제

| 예측 | 실제 | 배운 것 |
|---|---|---|
| 행이 가장 짧은 Qwen3.5가 가장 많이 빨라진다 | kernel은 그렇지만(36→83%), 전체는 +35%로 가장 작다 | 전체 개선 = kernel 개선 × 그 kernel의 시간 비중 |
| dense 모델은 +7% 이하 | +15% | 합친 gate+up kernel(67→82%)을 미니 벤치가 재지 않았다 |
| SM을 줄이면 포화 지점만 오른쪽으로 옮겨 간다 | 최고 속도도 함께 내려갔다 | SM 하나에도 13~18 GB/s 상한이 있다 |
| 계산이 무거운 형식은 계산량이 한계다 | IQ4_XS가 46%에서 85%가 됐다 | 대부분 block 모양 문제였다 |
| 첫 바이트 규칙이 실제 모델에서도 이긴다 | 4~12% 졌다 | 미니 벤치는 실제 시간을 쓰는 kernel 변형을 모두 포함해야 한다 |
| 둘째 규칙은 손해가 없다 | Qwen3 −2.2% | 레지스터는 데이터가 아니라 템플릿이 정한다 |

### 8.2 한계와 미해결 문제

- **미니 벤치와 실제 실행의 차이:** 같은 모양의 kernel이 실제 모델 안에서는 미니 벤치보다 약 13%(gate+up 약 8 µs) 느리다. 배치, expert id 중복, 측정 방법, 18 GB 범위·TLB는 원인이 아닌 것으로 확인했다. 레지스터는 2 µs만 설명한다.
- **아주 짧은 행의 저비트 형식:** 행 512에서 Q2_K, Q3_K, IQ4_XS는 여전히 38~49%다. 행 하나에 thread 16개면 충분한데, 규칙의 최소 단위가 warp 하나라서 절반이 논다. down 전용 kernel처럼 thread 16개가 행 하나를 맡는 방식이 후보다.
- **Q3_K, MXFP4:** Q3_K는 block 모양을 바꿔도 행 길이와 관계없이 72% 이하다. MXFP4는 행 1024 이하에서 56~72%다. 계산량 제약으로 보이지만 확인하지 않았다.
- **GPU는 Orin과 RTX 3090 두 개다(7.2절).** 4.2절의 다른 GPU 비교는 스펙 계산이다. 3090에서는 kernel 단위 현상이 재현됐지만, 전체 이득은 2~3%이고, 작은 dense 행렬을 위해 SM 수를 고려하는 조건이 필요했다.
- **남은 바닥 위 시간 6.4 ms:** 작은 연산 1.9 ms(29%), attention 1.0 ms, 호스트 0.9 ms가 남아 있다.
- **코드 상태:** 새 기능은 모두 환경 변수로 켜고 끄는 실험 코드이고 커밋하지 않았다. 최신 master용 patch는 `patches/bytes_rule_master_e300919.patch`이고, 작업 공간은 `llama.cpp-bb-master`, 빌드는 `build/master_bb`다.
- **upstream과 겹치는 부분:** master의 Orin 설정(7.1절)은 행렬 하나짜리 긴 행에서 우리와 같은 방향의 개선을 이미 담고 있다. 그러니 우리 기여를 "block당 바이트를 늘렸다"로만 설명하면 부족하다. **레지스터를 고려한 합친 kernel 모양과 짧은 행 처리, 형식 목록 없이 바이트로 정하는 규칙**이 차별점이다.

## 9. 남은 일과 일정

- [x] 11/22: 최종 규칙으로 logits를 다시 비교하고 perplexity를 쟀다(7절). GSM8K는 하지 않았다. 계산이 같은지는 logits와 perplexity가 더 직접적인 근거다.
- [x] 11/22: 최신 llama.cpp master(e300919)로 옮겨서 다시 쟀다(7.1절).
- [x] 11/22: 영어 보고서 초안 [REPORT.md](REPORT.md), 발표 자료 [SLIDES.md](SLIDES.md), 그림 `figures/`(`scripts/make_figures.py`)를 만들었다.
- [ ] 11/26~28 (선택): "thread 16개가 행 하나" 방식을 일반 규칙에 넣는다.
- [ ] 11/29~12/3: 보고서 초안(10pt, 최대 7쪽, 부록에 코드 위치와 실행 방법).
- [ ] 12/1~12/5: 발표 슬라이드와 영어 리허설(5분).
- 12/7 발표, 12/13 보고서 마감.

## 10. 부록: 코드 위치와 실행 방법

### 코드

| 위치 | 내용 |
|---|---|
| `llama.cpp-bb/ggml/src/ggml-cuda/mmvq.cu` | 모든 GEMV 변경. `llama.cpp`(13ec8f6)의 git worktree이고 커밋하지 않았다. `git -C llama.cpp-bb diff`로 볼 수 있다(+378줄) |
| ↳ 452, 586, 601행 | byte-balanced v1 (`calc_nwarps`, `mmvq_two_rows_ncols1`, `calc_rows_per_block`) |
| ↳ 1125~1170행, 1294행 | 바이트 기준 규칙 (`mmvq_bytes_target`, `mmvq_forced_shape`, `mmvq_bytes_shape_type`, 규칙 본체) |
| ↳ 1567행, 1688행 | gate+up 전용 kernel, down 전용 kernel |
| `llama.cpp-bb/tests/test-backend-ops.cpp` | 실제 크기 테스트 케이스 추가(+13줄) |
| `bench/gemv_shape.cu` | 행 길이 sweep(섹션 D)과 SM 수 제한(네 번째 인자, green context) |
| `bench/mmid_formats.cpp` | llama.cpp 라이브러리로 형식 13개 × 행 길이 5개를 재는 프로그램(`plain` / `fused`) |
| `scripts/` | 측정과 분석 스크립트. 사용법은 [PROFILING.md](PROFILING.md) |
| `scripts/validate_master.sh` | 최신 master 대비 테스트, 형식 측정, 모델 4개 비교(tmux에서 약 40분) |
| `scripts/accuracy.sh` | logits 비교와 `-ub 1` perplexity(tmux에서 약 40분) |
| `scripts/make_figures.py` | 측정 파일에서 `figures/`의 그림 4개를 다시 그린다 |
| `scripts/report_pdf.py` | `REPORT.md`를 A4·10pt PDF로 만든다(venv의 python으로 실행, LibreOffice 필요) |
| `llama.cpp-master`, `llama.cpp-bb-master` | 최신 master(e300919) 작업 공간. 수정본 patch는 `patches/bytes_rule_master_e300919.patch` |

### 환경 변수 (수정본에서만 의미 있음)

| 변수 | 의미 |
|---|---|
| `GGML_CUDA_MMVQ_BYTES=8192` | 바이트 기준 규칙을 켜고 block당 목표 바이트를 정한다 |
| `GGML_CUDA_MOE_GLU=0` | gate+up 전용 kernel(Q4_K)을 끈다. 변수가 없으면 켜져 있다 |
| `GGML_CUDA_MOE_PREFETCH=64` | gate+up kernel의 prefetch 거리(block 수). 기본값이 64다 |
| `GGML_CUDA_MOE_DOWN=0` | down 전용 kernel(Q4_K·Q6_K, 행 512~1024)을 끈다. 변수가 없으면 켜져 있다 |
| `GGML_CUDA_MMVQ_SHAPE=2,4` | 실험용. block 모양(warp 수, 행 수)을 강제로 지정한다 |

### 빌드와 실행

```
# 빌드 (둘 다 Release, CUDA, sm_87; 수정본은 테스트 포함)
cmake -S llama.cpp    -B /media/vlsi/NVMESSD/fused-moe/build/base -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release
cmake -S llama.cpp-bb -B /media/vlsi/NVMESSD/fused-moe/build/bb   -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON
cmake --build /media/vlsi/NVMESSD/fused-moe/build/bb --target ggml-cuda llama-bench test-backend-ops -j 8

# 최종 설정으로 디코드 속도
GGML_CUDA_MMVQ_BYTES=8192 GGML_CUDA_MOE_GLU=1 GGML_CUDA_MOE_DOWN=1 \
  /media/vlsi/NVMESSD/fused-moe/build/bb/bin/llama-bench -m /media/vlsi/NVMESSD/gguf/Qwen3-30B-A3B-Q4_K_M.gguf -ngl 99 -p 0 -n 64 -d 0,1024,4096 -r 5

# 정확성, 형식 측정, 모델 4개 비교를 한 번에 (tmux 안에서, 약 30분)
scripts/validate_rule.sh 8192
```

### 결과 파일 (`/media/vlsi/NVMESSD/fused-moe/profiling/`)

| 폴더 | 내용 |
|---|---|
| `rowlen/` | 행 길이 sweep, SM 수 실험 |
| `models/` | 모델 4개 원본 대 bb의 A/B와 nsys, `kernel_efficiency.txt` |
| `formats/` | 형식 13개 측정(원본, 첫 규칙) |
| `shapes/` | 모양 12가지 sweep, 레지스터 표 `regs.txt` |
| `rule_8192/` | 최종 규칙 검증(테스트, 형식, 모델 4개) |

## 11. Talk outline (5 minutes, English)

1. **Problem (30 s).** Batch-1 MoE decode on Jetson AGX Orin. Qwen3-30B-A3B reads 1.92 GB per token, so 180 GB/s caps it at 94 tok/s. llama.cpp reaches 43.4, which is 46% of that.
2. **Where the time goes (45 s).** Subtract the transfer floor. 10.3 of the 14.2 ms above it are GEMV kernel bodies. Launches are already fused into CUDA graphs. ncu shows every GEMV reads exactly its weight bytes, yet runs at about 50% of peak.
3. **Key observation (60 s).** Fine-grained experts have tiny rows (432 to 1,152 B). One row per thread block is too few bytes per block. Show the row-length sweep: speed follows bytes per block, not row length. Show the SM experiment with green contexts: small blocks scale with SM count, so the SMs are the limit, not DRAM.
4. **Method (60 s).** A block-shape rule from bytes and registers: warps for one row, rows until 8 KB per block, at most 4 partial sums per thread. Three iterations. The first lost on real models because the benchmark missed the fused gate+up kernel. Then registers explained it.
5. **Results (60 s).** All 65 format and row combinations got faster: median x1.28 for plain GEMV and x1.64 for fused, up to x3.97. End to end: Qwen3 +42%, Qwen3.5 +35%, OLMoE +40%, dense Qwen3-8B +15%. Tests and logits match.
6. **Takeaways (45 s).** The bottleneck depends on few SMs for the bandwidth, not on being an edge device. Benchmarks must include the kernel variants that dominate the real trace. Open items: 16-lane rows for 2 to 3 bit formats, the remaining small ops, and other GPUs.

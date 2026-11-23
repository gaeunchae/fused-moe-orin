# Orin 디코드 병목 프로파일링 절차

Qwen3-30B-A3B Q4_K_M, 배치 1 디코드에서 토큰당 시간을 "데이터 전송에 꼭 필요한 시간(floor)"과 "그 밖의 시간"으로 나눈다. 교수님 피드백(전송 비용을 제외한 병목 분석으로 방향을 정할 것)에 대응하는 절차다.

모든 명령은 `/home/vlsi/fused-moe`에서 실행한다. 결과는 `/media/vlsi/NVMESSD/fused-moe/profiling/`에 쌓인다. 결과와 해석은 [RESULTS.md](RESULTS.md)에 정리했다.

## 측정 전 확인

- [ ] NVMe가 마운트돼 있다. fstab에 없어서 재부팅 후에는 `sudo mount /dev/nvme0n1p1 /media/vlsi/NVMESSD`가 필요하다.
- [ ] 클럭이 고정돼 있다. 재부팅 후에는 `sudo jetson_clocks`를 다시 실행하고, `sudo jetson_clocks --show`에서 EMC 3199 MHz와 GPU 1300.5 MHz를 확인한다.
- [ ] 빌드나 다른 GPU 작업이 돌고 있지 않다. Jetson은 CPU와 GPU가 DRAM을 같이 쓰므로, 컴파일만 돌아도 메모리를 많이 읽는 kernel이 느려진다.

## 1단계: 토큰 시간 분해 (`nsys`, root 불필요)

```
scripts/nsys_depth_sweep.sh <llama-bench 경로> <라벨> 0 1024 4096 16384
```

- 출력: 깊이별 `d<깊이>.txt`(표)와 `.csv`.
- 읽는 법: `floor`는 읽어야 하는 바이트를 180 GB/s로 나눈 시간이고, `above`가 전송 외 비용이다. `% peak`가 낮을수록 개선 여지가 크다.
- KV cache는 컨텍스트 토큰 하나당 98,304 B이고, attention의 floor에 포함된다.

## 2단계: 호스트 시간 분해 (`nsys` CPU 샘플링, root 불필요)

```
nsys profile --trace=cuda,osrt --cuda-graph-trace=node --sample=process-tree \
  --sampling-period=137600 --cpuctxsw=process-tree -o host_d0 <llama-bench> -m <모델> -ngl 99 -p 0 -n 64 -d 0 -r 1
nsys export --type=sqlite -o host_d0.sqlite host_d0.nsys-rep
python3 -I scripts/host_gap.py host_d0.sqlite
```

그다음 그래프 단위 추적(`--cuda-graph-trace=graph`)과 추적 없는 `llama-bench`를 같은 조건으로 돌려, 프로파일러가 늘린 시간을 보정한다.

## 3단계: kernel 내부 원인 (`ncu`, root 필요)

```
sudo scripts/ncu_token.sh <llama-bench 경로> <라벨>
```

- 기본값은 안정된 디코드 토큰의 첫 2개 층(kernel 46개, 약 6분)이다. 앞의 3,330개는 모델 준비와 warmup이며, CUDA Graph를 끈 상태에서 `nsys`로 센 값이다.
- 다른 구간은 환경 변수로 고른다. 예: 토큰 전체는 `sudo SKIP=3330 COUNT=1110 scripts/ncu_token.sh ...`(약 2시간), 마지막 층과 출력층은 `SKIP=4411 COUNT=29`.
- 결과는 `profiling/ncu/<라벨>_token.ncu-rep`이고, 스크립트가 소유자를 일반 사용자로 되돌린다.

## 4단계: kernel별 시간 요약 (`nsys`, 모델 무관)

`profile_breakdown.py`는 Qwen3-30B-A3B의 텐서 크기가 코드에 들어 있다. 다른 모델은 이 스크립트로 kernel 이름과 grid별 실행 횟수, 평균 시간, 비중을 본다.

```
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none -o t <llama-bench> -m <모델> -ngl 99 -p 0 -n 32 -d 0 -r 1
nsys export --type=sqlite -o t.sqlite t.nsys-rep
python3 -I scripts/kernel_summary.py t.sqlite --top 15 --match mul_mat_vec
```

이름의 템플릿 인자로 형식(`t12` = Q4_K, `t14` = Q6_K), 합친 kernel 여부(세 번째 인자), block 모양(마지막 두 인자 = warp 수, 행 수)을 알 수 있다.

## 5단계: 미니 벤치 (행 길이, SM 수)

```
nvcc -O3 -std=c++17 -arch=sm_87 -o bench/gemv_shape bench/gemv_shape.cu -lcuda
bench/gemv_shape <모델.gguf> data/tensor_index.tsv D        # 행 길이 512~4096, block 모양 75가지
bench/gemv_shape <모델.gguf> data/tensor_index.tsv D 8      # 같은 측정을 SM 8개로 (green context, 최소 4)
```

실제 Q4_K 가중치 7.08 MB를 행 길이만 바꿔 자른다. 출력의 `SMs in use`가 실제로 쓰인 SM 수다.

## 6단계: llama.cpp kernel로 형식과 block 모양 측정

`bench/mmid_formats.cpp`는 llama.cpp 라이브러리를 그대로 불러 MUL_MAT_ID를 잰다. 원본과 수정본에 각각 링크한다.

```
g++ -O2 -std=c++17 -Illama.cpp-bb/ggml/include bench/mmid_formats.cpp -L$B/bb/bin -lggml -lggml-base -lggml-cuda \
    -Wl,-rpath,$B/bb/bin -o bench/mmid_formats_bb          # B=/media/vlsi/NVMESSD/fused-moe/build, base도 같은 방식
bench/mmid_formats_bb [형식|all] [plain|fused] [열 수]
scripts/shape_sweep.sh                                     # 모양 12가지 × plain/fused × 65조합, tmux에서 약 14분
python3 -I scripts/shape_analysis.py                       # 조합별 최고 모양과 규칙 후보의 점수
```

- `plain`은 행렬 하나(down 모양), `fused`는 gate와 up을 합친 kernel(gate+up 모양)이다.
- 시간은 노드당 wall clock이고 입력 양자화(약 2 µs)가 들어 있다. kernel만의 시간은 이 프로그램을 `nsys`로 감싸서 얻는다. 설정 하나당 MUL_MAT_ID launch가 288개(warmup 48개 포함)이므로 실행 순서로 나눈다.
- 레지스터 표 `profiling/shapes/regs.txt`는 `cuobjdump --dump-resource-usage`로 만들었다(18번 참고).

## 7단계: 모델별 A/B와 규칙 검증

```
MODEL=<모델.gguf> OUT=<폴더> CONFIGS="base bb_v1 bb_all bytes bytes_all" BYTES=8192 scripts/ab_bench.sh 0,1024,4096
scripts/validate_rule.sh 8192        # 테스트 3종 + 형식 측정 + 모델 4개, tmux에서 약 30분
```

## 놓치기 쉬운 점

1. **측정 중에는 빌드하지 않는다.** 호스트 시간이 부풀고, DRAM을 같이 쓰므로 GPU kernel도 느려질 수 있다.
2. **`nsys`에는 `--cuda-graph-trace=node`를 붙인다.** 없으면 CUDA Graph 하나가 덩어리 하나로만 보인다.
3. **짧은 값은 실행마다 흔들린다.** 토큰 사이 호스트 시간은 같은 조건에서도 0.7~1.0 ms로 달랐다. 결론 전에 반복한다.
4. **옵션은 버전마다 다르다.** 이 Orin의 `nsys` 2024.5는 샘플링 빈도 대신 `--sampling-period`(CPU 사이클)를 받는다. 막히면 `--help`부터 본다.
5. **CPU 샘플은 GPU가 쉬는 구간만 잘라 본다.** 전체로 집계하면 GPU를 기다리는 동기화 대기가 대부분을 차지한다.
6. **프로파일러도 시간을 쓴다.** kernel 단위 추적은 토큰당 약 0.8 ms를 더했고, 토큰 사이 CPU 샘플의 30%가 프로파일러 자신이었다. 호스트 수치는 추적 없는 실행과 비교해 보정한다.
7. **kernel 내부를 볼 때는 CUDA Graph를 꺼도 된다.** 바뀌는 건 launch 방식뿐이고 kernel은 같다(토큰당 1,110개로 동일).
8. **`ncu`는 `--clock-control none`으로 돌린다.** 기본값은 GPU를 base 클럭으로 고정해서, 실제 운영 클럭과 다른 결과가 나온다.
9. **`sudo`는 환경 변수를 지운다.** 그래서 스크립트 안에서 `env GGML_CUDA_DISABLE_GRAPHS=1`로 넘긴다. 스크립트에 넘길 값은 `sudo SKIP=... COUNT=... 스크립트` 형태로 쓴다.
10. **`ncu`는 kernel마다 비싸다.** kernel 하나를 18번 다시 실행하고, 매번 GPU 메모리를 백업·복원한다. 모델이 17 GB를 잡고 있어 kernel당 약 6.6초가 걸렸다. 같은 층이 반복되므로 대표 구간만 잡는다.

11. **`ncu` 지표는 질문부터 정하고 고른다.** 지표가 수백 개다. 이번에는 바이트(DRAM 바이트 ÷ 가중치 바이트), 기다림(stall 원인), 채움(occupancy와 그 상한)만 봤다.
12. **stall 이름의 뜻.** `long_scoreboard`는 메모리 load 대기, `lg_throttle`은 load 명령 과다, `barrier`는 `__syncthreads` 대기, `wait`과 `short_scoreboard`는 연산·shared memory 대기, `imc_miss`와 `no_instruction`은 명령어 캐시와 kernel 시작 비용이다.
13. **occupancy는 무엇이 상한을 정하는지 같이 본다.** 2 warp block은 SM당 block 16개 상한 때문에 warp가 67%까지만 찬다. Q6_K down은 레지스터 96개 때문에 42%에서 막힌다.
14. **시간은 `nsys`, 원인은 `ncu`에서 가져온다.** `ncu`는 매번 캐시를 비워서, 작은 kernel의 시간이 실제보다 몇 배 크게 나온다.
15. **다른 하드웨어용 최적화는 우리 모양에서 실제로 실행되는지부터 본다.** llama.cpp의 GB10용 prefetch는 행 안의 다음 iteration만 당기는데, 2048 폭 행은 iteration이 2번뿐이라 거의 동작하지 않았다.
16. **prefetch 거리는 sweep으로 정한다.** gate+up에서 64 block 앞은 +5~10%였지만, 384 block 이상은 −18~−32%였다. 미리 당긴 데이터가 L2에서 밀려나 두 번 읽히는 것으로 추정한다(미확인).
17. **얽힌 매개변수는 함께 맞춘다.** prefetch를 넣자 최적 block 모양이 2 warp에서 4 warp로 바뀌었다.
18. **레지스터는 `cuobjdump --dump-resource-usage`로 본다.** `grep -A1` 출력에는 `--` 구분 줄이 끼어서, 그대로 `paste - -`로 짝지으면 줄이 밀린다. 먼저 `grep -v '^--$'`로 지운다.
19. **새 kernel이 실제로 실행됐는지 `nsys`로 확인한다.** Qwen3.5에서는 gate+up 전용 kernel이 shared expert 때문에 한 번도 실행되지 않았다. 숫자만 보면 "효과 0"으로 오해한다. 새 kernel은 환경 변수로 켜고 끌 수 있게 만들어 A/B를 잰다.
20. **테스트를 통과해도 새 경로가 쓰였다는 뜻은 아니다.** 규칙이 고른 모양이 실제로 실행됐는지는 kernel 이름의 템플릿 인자와 grid로 확인한다.
21. **미니 벤치의 최적값은 실제 실행에서 다시 확인한다.** 같은 모양의 kernel도 실제 모델 안에서는 미니 벤치보다 약 13% 느렸다(원인 미확인).
22. **미니 벤치는 실제 추적에서 시간을 많이 쓰는 kernel 변형을 모두 포함해야 한다.** 행렬 하나짜리 kernel만 재서 만든 첫 규칙은 실제 모델에서 4~12% 졌다. 시간을 가장 많이 쓰는 gate+up은 행렬 두 개를 읽는 합친 kernel이었다.
23. **레지스터는 데이터가 아니라 템플릿이 정한다.** 합친 kernel 템플릿은 gate가 없어도 gate용 부분합 레지스터를 잡는다. 레지스터가 늘면 SM에 올라가는 block 수가 줄고(96개면 128-thread block 5개), 동시에 읽는 바이트가 준다.
24. **입력 분포와 메모리 배치를 실제와 맞춘다.** expert id는 top-k처럼 중복 없이 뽑는다. 노드마다 다른 가중치를 읽게 해서 L2에 다시 걸리지 않게 한다.
25. **예측을 측정 전에 적는다.** 틀린 예측(dense +7% 이하 → +15%, 첫 규칙의 손해)이 원인을 찾는 출발점이 됐다. 측정 뒤에 쓴 설명은 무엇이든 맞아 보인다.
26. **같은 조건으로 다시 재서 재현성을 본다.** 원본 Qwen3는 세 번 재서 43.36 / 43.39 / 43.43 tok/s였고, bb_v1은 이틀 간격으로 57.85와 58.05였다. 그래서 날짜가 다른 결과를 한 표에 넣을 수 있었다.
27. **SM 수를 바꾸는 실험은 green context로 한다.** sudo와 재부팅이 필요 없고 클럭과 DRAM은 그대로다. `nvpmodel`의 30W 모드도 SM을 8개로 줄이지만, GPU 클럭 상한과 CPU 설정까지 함께 바꾼다.
28. **오래 걸리는 측정은 tmux에서 돌린다.** 편집기의 백그라운드 작업은 편집기를 끄면 같이 끝난다. `tmux new-session -d -s <이름> "<명령> | tee <로그>"`로 실행하고 `tmux attach -t <이름>`으로 본다.
29. **`pkill -f`는 자기 셸까지 죽일 수 있다.** 명령줄 전체를 비교하기 때문이다. 프로세스 이름으로 찾는 `pkill -x`를 쓴다.
30. **규칙의 매개변수를 데이터에 맞추면 과적합될 수 있다.** 형식 측정으로 고른 매개변수는 그 데이터에 없는 실제 모델 4개로 따로 검증했다.

## 지금까지의 결과 (byte-balanced 패치본, 2026-11-18)

이 절과 다음 절은 2026-11-18 시점 기록이다. 최신 결과는 [RESULTS.md](RESULTS.md)에 있다.

컨텍스트 0 기준 토큰당 17.28 ms(추적 없음, 57.86 tok/s) 중 전송 floor는 10.67 ms다. 전송 외 시간은 컨텍스트 16K까지 7.4~7.7 ms로 거의 일정했다(kernel 단위 추적 기준).

| 전송 외 시간 | ms/토큰 | 비고 |
|---|---:|---|
| 양자화 GEMV | 3.9 | gate+up 1.41 ms가 가장 큼 |
| 작은 연산 | 1.9 | 약 675개 kernel, 평균 2.8 µs |
| attention 고정 비용 | 0.9 | 16K에서 attention은 peak의 90% |
| 호스트 | 0.3~0.6 | 보정 후 추정 |
| 호스트-GPU 복사 | 0.1 | 토큰 시간의 0.6% |

## kernel 내부 분석과 prefetch 실험 (2026-11-18)

`ncu`(kernel 336개, 약 14.6개 층): 모든 GEMV가 필요한 바이트만 정확히 읽는다(비율 1.00~1.02). expert kernel마다 막히는 이유가 다르다.

| kernel | 대역폭 | 막히는 이유 | prefetch 효과 |
|---|---:|---|---:|
| gate+up | 69% | 메모리 대기, occupancy 상한 67% | +5~10% |
| down Q6_K | 68% | 레지스터 때문에 occupancy 상한 42% | 미측정 |
| down Q4_K | 64% | 명령 발행 67% | −5% |

재현 kernel(`bench/gemv_prefetch.cu`)에서 gate+up 최적값은 4 warp, 2행, 64 block 앞 L2 prefetch로 86.6 µs(peak의 91%)다. 같은 조건 prefetch 없는 2 warp, 2행은 93.7 µs였다.

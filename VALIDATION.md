# 로컬 검증 기록

2026-09-14. 아래 결과는 구현 검증이며 음악 분리·복원 성능 결과가 아니다.

## 환경과 명령

- macOS local CPU, Python 3.9 계열, PyTorch 2.8.0.
- Core/smoke 실행은 CPU 2 threads. CUDA는 사용하지 않았다.
- `PYTHONPATH=src python3 -m unittest discover -s tests -v`
- `PYTHONPATH=src python3 -m msr.cli smoke`
- `PYTHONPATH=src python3 -m msr.cli smoke --config configs/moises6.json --samples 4096 --threads 2`
- `python3 -m compileall -q src tests`

## 검정 범위

28개 unit/integration 검정:

- reference analysis/synthesis: 길이 1, hop 직전/직후, FFT 직전/직후와 비정렬 tail
- LieRE orthogonality, angular sin/cos periodicity, finite nonzero parameter gradient
- Temporal의 frequency coordinate 부재, G/C의 공통 물리 주파수 좌표
- zero gate의 A0 exact output recovery
- frozen backbone/head state 불변 및 adapter/PE 학습 경로
- token 수준과 synthesis latency를 고려한 waveform future perturbation
- KV joint permutation invariance, coordinate-only permutation 민감도
- explicit wrong-track memory를 사용하는 end-to-end intervention
- all-masked attention finite zero output/gradient
- release clock 오류 거부, 4D SDPA 호출
- one-batch synthetic target에 대해 12-step MSE가 초기의 70% 미만으로 감소
- known-absence vs missing-label 구분 및 explicit degraded MSR input
- song split leakage, official evaluation→train 오용 거부
- parent gradient/selection lineage의 held-out leakage 거부
- WAV 내용 변경 fingerprint와 variant-independent sample order
- six→eight safe head mapping 및 새로운 head의 공통 초기값
- crop sufficient statistics와 전체 waveform metric 일치
- CPU continuous 2-step vs 1-step+resume 결과의 모든 state tensor bitwise 일치
- synthetic paired WAV 기반 pretrain→A1/A2 adapter→MSR fine-tune→test evaluation→8 WAV inference
- 실제 MoisesDB metadata 형태 fixture의 resampling/mono/other mapping/export
- checkpoint config hash 손상 거부
- paired result 비교에서 repeated view를 song group으로 묶는 bootstrap

Test fixture는 임시 디렉터리의 합성 audio이며 실제 데이터가 아니다. 끝나면 임시 파일은 test cleanup으로 제거한다. 실제 학습 checkpoint를 생성한 것으로 해석하지 않는다.

## 기본 config smoke

`d_model=128`, heads 4, layers 4, FFT 1024, bands 16, 48 kHz, 4096-sample stereo input에 대해 모든 arm의 output shape는 `[1,6,2,4096]`이며 finite forward/loss/backward를 통과했다.

| 항목 | 값 |
|---|---:|
| 공통 allocated parameter 수 | 3,767,451 |
| A0 활성 학습 parameter 수 | 3,568,536 |
| A1 전체 fine-tuning 때 학습 수 | 3,765,915 |
| A2 전체 fine-tuning 때 학습 수 | 3,767,451 |
| Frozen A1 adapter 학습 수 | 197,379 |
| Frozen A2 adapter 학습 수 | 198,915 |
| 세 arm gate-off baseline 최대 차이 | 0.0 |

A0도 checkpoint/초기값 비교의 명확성을 위해 비활성 injection module을 저장한다. Allocated 수를 실제 실행되는 model size와 혼동하지 않는다. 4×4 LieRE raw generator parameter는 1,536개이며 skew projection 때문에 모든 raw 성분이 독립 자유도인 것은 아니다.

최적화 후 한 번의 local forward+loss+backward smoke 관측은 A0 약 0.10초, A1 약 0.09초, A2 약 0.23초였다. 반복 평균·throughput benchmark가 아니며 warmup/order에 영향을 받는다. 4096 sample은 48 kHz에서 약 85 ms이므로 2초+pre-roll 학습의 메모리/시간으로 외삽하지 않는다.

## 미검증

- 실데이터 pretraining/fine-tuning, held-out 성능 개선, artifact 개선, source별 청취
- CUDA kernel selection, CUDA numerical parity, full crop GPU peak memory/속도
- 다른 hardware/version에서 bitwise resume, mixed precision, distributed training
- official MSR Multi-Mel-SNR/Zimtohrli/FAD-CLAP 및 MOS
- 강한 baseline 대비 우월성 또는 LieRE 고유 기여

검토 기록: `reviews/01_math_and_causality.md`, `02_experimental_controls.md`, `03_execution_and_handoff.md`.

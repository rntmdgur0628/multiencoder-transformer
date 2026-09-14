# 보고서 → 코드 대응 및 baseline audit

설계 근거: `/Users/kuseunghyeok/Documents/Guitarist Feature Research/outputs/msr_liere_minimal_design/REPORT.md`.

## 감사한 기존 경로

- `/Users/kuseunghyeok/Downloads/RESEARCH/multiencoder`: 과거 run/checkpoint/result archive. 재사용할 source package가 없었다.
- `/Users/kuseunghyeok/Downloads/RESEARCH/learnable-single-operator-sigma/src/multi_operator/q1q2/families.py`: Temporal은 학습 Conv channel이고 Gabor/Chirplet의 `StreamProjection`은 전체 spectrum을 channel vector로 압축한다.
- 같은 디렉터리 `model.py`: TCN 기반 family/expert mask head와 최종 mask mixture. 새 common Transformer baseline과 checkpoint contract가 다르다.

따라서 기존 파일을 수정하거나 모델 checkpoint를 재해석하지 않았다. 원시 complex 분석의 개념만 참고하여 별도 self-contained package를 구현했다. 과거 operator bank의 multirate 성능이나 waveform winner 비율을 현재 코드의 성능 근거로 사용하지 않는다.

## 수식·구현 대응

| 보고서 항목 | 구현 | 제한 |
|---|---|---|
| 세 native feature | `encoders.py` | 첫 baseline은 single common FFT grid; historical multirate 재현 아님 |
| frequency token 보존 | G/C의 band별 coefficient projection | band 내부 bin 순서는 input channel에 보존; **band 간** attention PE |
| Temporal time-only | `[B,T,1,D]`, angular component zero | temporal channel을 frequency로 해석하지 않음 |
| Q/K/V content projections | `ResidualInjection.qkv` family별 projection | source waveform/mask는 memory에 없음 |
| 동시 branch update | 원 Z에서 Q/K/V 전부 계산 후 update | 순차 update와 별도 pair token 없음 |
| 작은 scalar gate | learnable unconstrained branch scalar 0.001 | simplex/확률 아님; 음수가 될 수 있음 |
| 공통 Hz angular chart | `position.angular_coordinates` | 선형 Hz→반원은 hypothesis |
| LieRE | 4×4 skew generator blocks + matrix exponential | 시간 원점 이동 불변성/physical SO(2) symmetry 미주장 |
| 공동 Transformer | joint family/band + causal temporal attention | 모든 arm이 이미 branch interaction 수행 |
| 같은 output head | shared band complex-ratio-mask heads | full missing coefficient 복원 불가; 후속 head 비교는 별도 |
| gate-off baseline | 동일 module state + 추가 block 우회/zero gate | 새로운 A0 pretrain checkpoint 기준 |
| Moises→MSR | `checkpoint.transfer_to_msr` | canonical head mapping 네 개만 자동 이전 |
| 작고 통제된 실험 | pretrain A0 → frozen A1/A2 screen | A0는 optimizer 없는 anchor |
| source별 평가 | track sufficient statistics + paired song bootstrap | 공식 MSR 지표는 별도 |

## Clock과 geometry

Frame k의 release sample은 `k*hop + hop - 1`이다. Center는 각 analysis support의 중심으로 계산하지만, memory availability는 center가 아닌 release로 결정한다. 현재 구현은 공통 release grid에 한정하며 배열의 증가/hop을 검증한다. 다른 native hop의 asynchronous memory를 자동으로 처리하는 API로 설명하지 않는다.

Reference analysis는 `left=n_fft-hop`으로 시작하고 완전한 overlap tail까지 zero flush한다. Hamming analysis/synthesis WOLA로 임의 길이를 복원한다. Waveform output은 시간 순서상 즉시가 아니라 최대 `n_fft-1` 지연을 갖는다. 지연 이전 구간의 future perturbation 불변성을 검정한다.

`F={temporal,gabor,chirplet}`는 label 집합이다. Family embedding은 ID lookup일 뿐 family 덧셈/회전/순환을 정의하지 않는다. Angular coordinates는 bin energy 값이나 complex phase가 아니다. G/C는 complex coefficients의 real/imag를 모두 content로 유지한다. Chirplet의 positive-frequency 부분만 사용하는 것은 feature 선택이며 chirplet 자체의 PR 주장을 하지 않는다.

## 학습·데이터 contract

Loss는 모든 arm에 같은 waveform L1 + multi-resolution magnitude L1 + log1p-magnitude L1이다. Spectral loss는 target crop 안에서만 계산하며 short crop은 zero-pad한다. GAN, teacher oracle, degraded-mixture consistency, representation-alignment 보조 loss는 없다. Validation checkpoint 선택은 track-macro reconstruction loss다.

Preroll과 synthesis 뒤 문맥은 실제 같은 곡에서 읽고 track 바깥만 zero-pad한다. Sampler는 `(seed, step)`의 local RNG로 crop을 결정하므로 variant의 module 생성이 data order를 바꾸지 않는다. `A1/A2`는 부모 model state 전체를 동일하게 복사하고 해당 injection parameter만 학습한다. Head parameter가 frozen이어도 head 입력은 바뀌므로 output mask 값은 변한다.

Manifest/audio content 및 parent lineage를 저장한다. Group alias가 잘못 기입된 서로 다른 recording을 acoustic fingerprint로 찾아내는 시스템은 아니다. Exporter의 zero-padding은 alignment correction이 아니며 원본 시간 정렬은 별도로 확인해야 한다.

## 아직 하지 않은 일

- MoisesDB 실데이터 pretraining, MSR pairs fine-tuning 및 held-out quality 평가
- GPU full-length preflight/성능·메모리 측정, AMP 또는 DDP
- Official Multi-Mel-SNR/Zimtohrli/FAD-CLAP integration 및 청취 평가
- time-only LieRE, 같은 좌표의 RoPE, linear-frequency control, branch-local capacity control
- native multirate encoder/tap transfer와 streaming cache

이 항목들은 코드 실패를 숨긴 목록이 아니라 이번 최소 구현이 성공했다고 주장하지 않는 범위다.

## 외부 근거

- [LieRE 원 논문](https://arxiv.org/html/2406.10322v1): position→skew generators→matrix exponential로 Q/K 회전. 음악 복원 실증은 아님.
- [MoisesDB 공식 loader](https://github.com/moises-ai/moises-db/blob/main/moisesdb/track.py): native metadata layout 확인에 사용. Exporter의 tail policy는 이 프로젝트에서 명시적으로 선택했다.
- [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html): bool attention mask는 True가 허용 위치다. 실행 경로에서 dropout은 0.

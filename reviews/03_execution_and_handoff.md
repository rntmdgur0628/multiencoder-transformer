# 3차 비평·수정 — 실행 비용·재현·전달

## 비평

1. Local SDPA의 입력이 `[B,T,H,Q,D]` 5차원이어서 CPU 검정만 통과하고 CUDA fused attention 경로를 놓칠 수 있었다.
2. 동일 native feature에 Q/K/V projection과 LieRE exponential을 두 번 계산하고 있었다. Q/K가 공유할 회전을 중복 계산할 이유가 없다.
3. 파라미터 수만 공개하면 A0에서 비활성화된 injection/PE까지 실제 사용량으로 오해할 수 있다. Tiny test만 통과하면 기본 width에서도 된다고 단정할 수 없다.
4. 평가 command만 만들고 준비 데이터 규약을 생략하면 사용자가 missing annotation을 zero stem으로 만들거나 validation을 train으로 오용할 수 있다.
5. Module 단위 shuffle test만으로는 intervention이 common backbone PE까지 바꾸는 실수를 잡지 못한다.

## 수정

- SDPA 호출 직전에 batch×time을 합쳐 4차원 Q/K/V로 전달. CPU 수치·인과성·gradient 재검정 및 입력 rank 회귀 test 추가. CUDA fused kernel 실제 선택/속도는 미측정이다.
- 동일 memory의 projection과 회전을 forward 안에서 재사용. Cross-track/shuffled memory의 별도 좌표는 유지한다. 영구 cache나 streaming state는 추가하지 않는다.
- Active/trainable parameter 수와 실제 baseline gate-off 차이를 출력한다. 기본 128차원·4-layer 모델에서도 forward/backward smoke를 실행한다.
- Official MoisesDB metadata layout을 읽는 별도 exporter 추가. 명시적 song split, stereo/rate 정규화, unknown category→other, absent target, tail-padding 및 원본 hash를 기록한다. MSR은 정렬된 raw/degraded paired manifest를 별도 요구한다.
- 전체 모델에서 gate-off, frequency-coordinate-only permutation, KV joint permutation, explicit wrong-track memory를 지원. Injection 후 원 좌표를 복원하여 common trunk PE를 고정한다.
- 평가 결과의 paired 비교는 underlying song group으로 bootstrap하고, repeated view는 group 안에서 먼저 평균한다. Checkpoint는 best score까지 한 번의 atomic save에 기록한다.

## 검증·남은 범위

- Core, data, transfer, resume, CLI round-trip, exporter, paired-statistics 테스트를 재실행한다. 최종 집계는 `VALIDATION.md`.
- 공식 데이터 다운로드, 장시간 pretrain/fine-tune, CUDA 메모리/속도 측정, official MSR metrics는 실행하지 않았다.
- Full 2초 + pre-roll batch의 GPU profile 전에는 현재 default가 목표 GPU에 적합하다고 주장하지 않는다. 기본값은 engineering starting point이지 튜닝 결과가 아니다.
- 검토는 동일 에이전트가 서로 다른 초점으로 수행한 실제 순차 3회 비평·수정이다. 외부/독립 심사를 수행한 것으로 표현하지 않는다.

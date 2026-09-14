# 1차 비평·수정 — 수학·신호·인과성

검토 대상: 최초 core 구현. 같은 에이전트의 순차 자체 검토이며 독립 심사라고 주장하지 않는다.

## 비평

1. 기존 Gabor/Chirplet StreamProjection은 주파수 전체를 채널에 압축한다. 이를 그대로 가져오면 bin/band PE 실험이 아니다.
2. Local attention은 과거 frame index를 읽는다. release 배열이 뒤집혀 있어도 동일 배열끼리만 비교하면 통과하여 인과성 contract를 위반할 수 있다.
3. 단순 left-pad STFT는 마지막 sample의 overlap을 충분히 합성하지 못할 수 있다. 인과성은 token 시점과 지연된 waveform 시점을 따로 검정해야 한다.
4. gate 값만 작다고 PE gradient 또는 baseline 복구가 보장되지는 않는다.

## 수정

- 원래 all-frequency latent/checkpoint를 사용하지 않는 새 band-preserving baseline을 구현. Temporal은 time-only, G/C는 complex coefficient를 band 안에서만 projection한다.
- family label/order, shape/validity, 공통 release clock과 정확한 hop 증가를 검증하도록 수정. 임의 native asynchronous grid는 지원한다고 주장하지 않는다.
- full-tail right flush와 WOLA normalization을 구현. Hamming reference window와 보수적 `n_fft-1` sample 지연을 명세.
- residual 경로 바깥에는 normalization을 삽입하지 않음. 공유 LieRE skew generators, FP32 exponential, A0/A1/A2 공통 초기화 및 frozen-backbone gradient 경로를 검정.

## 검증

최초 10개 unit test 통과 후 release-clock 회귀 검정, coordinate-only shuffle 민감도, branch-level future perturbation을 추가했다. 이후 테스트 결과는 `VALIDATION.md`에 최종 집계한다.
이 검증은 구현 정확성에 한정되며 음악 분리·복원 성능 결과가 아니다.

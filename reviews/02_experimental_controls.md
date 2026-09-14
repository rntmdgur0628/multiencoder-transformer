# 2차 비평·수정 — 실험 비교·데이터·checkpoint

## 비평

1. Manifest 파일 hash만 같아도 WAV 내용은 바뀔 수 있다. 같은 실험의 재개라고 보기 어렵다.
2. 현재 fine-tuning manifest만 분리해도 parent pretraining의 학습 곡이 새 평가 세트로 들어갈 수 있다. Parent validation 곡을 새로운 untouched test로 부르는 것도 잘못이다.
3. Run 폴더에 relative-path manifest를 그대로 복사하면 다른 기준 디렉터리에서 잘못 해석된다.
4. Gate diagnostic이 detach된 view이면 optimizer.step 뒤 값이 바뀌어 같은 forward의 residual RMS와 일치하지 않는다.
5. Crop 평균 SI-SDR를 곡 전체 SI-SDR로 보고하면 평가 단위가 달라진다. 6→8 head를 shape만 맞춰 복사하면 taxonomy가 섞인다.

## 수정

- 시작 시 audio header와 실제 file SHA-256을 검사하고 aggregated audio fingerprint를 run/resume contract에 추가했다. 대용량 데이터에서는 이 읽기 비용이 발생한다.
- Parent의 gradient-training group과 checkpoint-selection group을 lineage로 상속하고 held-out 역할 충돌을 차단했다. 같은 곡의 dataset 간 alias는 사용자가 canonical group에 표시해야 한다.
- Run manifest snapshot에는 해석 가능한 absolute audio path를 저장한다.
- Gate 값은 forward 시점 clone으로 기록한다.
- Full-track sufficient statistics로 joint-stereo SI-SDR/SNR을 계산하고 곡별 결과를 보존한다. 이것을 official MSR metric으로 표시하지 않는다.
- Safe source mapping은 vocals/bass/drums/guitar→guitars뿐이다. Piano→keyboards는 자동 이전하지 않으며 새 head 초기값은 모든 arm에서 같다.
- 동일 model/module 초기화, variant-independent sampler, frozen adapter 학습, checkpoint contract exact-match 및 CPU bitwise resume 검정을 추가했다.

## 범위

실제 MoisesDB/MSR 음원은 이 작업에서 학습하지 않았다. Small synthetic paired WAV로 pretrain→A1/A2 adapter→MSR transfer→evaluation→WAV inference를 검정한다. Sample order와 loss/initialization을 통제하지만 A1−A0에는 추가 adaptation capacity, A2−A1에는 time+frequency PE 효과가 함께 포함된다.

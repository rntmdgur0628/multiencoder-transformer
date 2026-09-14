# Multiencoder Transformer — MSR 최소 실험

상태: **실행 가능한 연구 코드와 로컬 검증 완료. 실제 MoisesDB/MSR 학습·성능 검증 전.**

세 encoder가 분리를 끝내고 결과를 섞지 않는다. Temporal/Gabor/Chirplet feature를 공동 Transformer가 읽고, 공통 complex-mask head로 source를 출력한다. Encoder 직후 한 곳에 `Z̃_i = Z_i + g_i CA(Z_i, Z_-i)`를 넣는다. Q3 pair token, family group 연산, Markov/router, GAN은 없다.

| 조건 | 추가 injection | 추가 block의 PE | Moises adapter 단계 학습 범위 |
|---|---|---|---|
| A0 | 꺼짐 | 없음 | 업데이트 없는 anchor |
| A1 | 작은 scalar gate | 없음 | attention + gate |
| A2 | A1과 동일 | 시간 + 주파수 각도 LieRE | attention + gate + LieRE |

공통 Transformer의 PE·tokenization·head는 세 조건에서 같다. A0도 공동 Transformer 안에서 branch 관계를 학습한다. 따라서 A1−A0는 추가 injection의 효과이며, A2−A1은 time+frequency PE 묶음의 효과다. LieRE 고유 우월성은 별도 matched-PE control이 필요하다.

## 설치와 로컬 검증

```bash
cd /Users/kuseunghyeok/Downloads/RESEARCH/multiencoder-transformer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[prepare]'
python -m unittest discover -s tests -v
python scripts/validate.py
msr smoke
msr smoke --config configs/moises6.json --samples 4096
```

설치 없이 기존 PyTorch 환경에서는 `PYTHONPATH=src python3 -m msr.cli ...`도 가능하다. Tests는 unittest를 사용하며 pytest가 필요 없다. Exporter 검정에는 optional `scipy`가 필요하다. 실제 검증 환경·결과는 [VALIDATION.md](VALIDATION.md).

`scripts/validate.py`는 재실행 가능한 테스트 로그와 machine-readable smoke 결과를 `validation_artifacts/`에 저장한다.

## 보고서에서 구현으로 고정한 선택

- **새 baseline lineage**다. 기존 `multiencoder` 폴더는 결과 archive였고, historical 구현의 G/C latent는 주파수 전체를 projection했다. 기존 checkpoint를 새 모델인 것처럼 불러오지 않는다.
- 현재 기본값은 48 kHz stereo, FFT 1024/hop 256, 16개 equal-bin band, width 128, 4 heads, 4 blocks. 실험 최적값이 아닌 명시적인 시작값이다.
- Temporal: learned waveform convolution → 시간당 한 token. 가짜 주파수 좌표 없음.
- Gabor: fixed Gaussian complex coefficients → band별 learned projection. Chirplet: 같은 grid의 nonzero rates `(-4,+4)` coefficients를 band 내부에 함께 보존. 두 branch 모두 energy-only가 아니다.
- 이는 **단일 공통 FFT grid의 첫 baseline**이며 과거 4-scale native-multirate bank를 그대로 재현한 것이 아니다. Scale/rate 의미를 새로운 좌표축으로 추가하지 않는다.
- Common Transformer: frame 안에서 family/band joint self-attention → slot별 causal temporal attention. 고정 local window로 full time×frequency dense attention의 크기를 제한한다.
- Frequency band 중심의 물리 Hz를 반원에 배치하고 `(time/τ, v cosθ, v sinθ)`를 추가 block의 LieRE에 입력한다. `v=0`인 Temporal의 angular 성분은 0이다. 반원은 구현 제안이지 실제 주파수 원형 대칭 주장이 아니다.
- LieRE는 head마다 공유하는 **4×4 skew-generator rotation blocks**를 사용한다. SO(2) 입력 각도와 SO(4) 회전 block은 다르며 exact relative displacement/equivariance를 보장하지 않는다.
- Common head는 source별 **unbounded complex ratio mask**다. Mixture consistency, source softmax, expert-output sum은 없다. 완전히 사라진 mixture coefficient는 이 head만으로 생성할 수 없다는 표현 제약이 남는다. 첫 attention 비교에서는 동일 head를 유지한다.
- `gate=0`에서 동일 A0 출력으로 정확히 복구된다. 이것은 과거 TCN checkpoint와의 동등성이 아니라 **새로 pretrain한 공통 A0**와의 동등성이다.

더 자세한 대응 및 미구현 범위: [DESIGN.md](DESIGN.md).

## MoisesDB 준비

데이터를 자동 다운로드하지 않는다. 정식으로 확보한 원본과 명시적인 track split 파일을 준비한다.

```json
{
  "actual-track-uuid-a": {"split": "train", "group": "canonical-song-a"},
  "actual-track-uuid-b": {"split": "validation", "group": "canonical-song-b"},
  "actual-track-uuid-c": {"split": "test", "group": "canonical-song-c"}
}
```

`group`은 다른 mix/버전/데이터셋에 등장해도 같은 원곡이면 같은 값이어야 한다. 위 ID는 예시이며 실제 metadata UUID로 바꾼다. Exporter는 split에 명시한 곡만 준비한다.

```bash
msr prepare-moises --root /absolute/path/to/moisesdb \
  --splits /absolute/path/to/song_splits.json \
  --output-dir /absolute/path/to/prepared_moises6 --sample-rate 48000
```

`vocals/bass/drums/other/piano/guitar` taxonomy를 사용한다. 그 밖의 metadata stem category는 `other`에 합친다. Mono는 stereo 복제, sample rate는 polyphase resampling, 짧은 녹음의 tail은 zero-pad로 처리하고 `preparation.json`에 기록한다. FLOAT WAV로 저장해 clipping/independent normalization을 하지 않는다. Audio가 정렬돼 있다는 데이터의 전제는 자동 교정하지 않는다. Download/license 및 metadata completeness는 원본 데이터 책임 범위다.

## 학습 순서

아래 update 수는 명령 사용 예시다. 실제 비교에서는 seed·batch·data order·update budget을 공통으로 고정한다. 명령을 제공했을 뿐 이 작업에서 실행한 실데이터 학습은 없다.

```bash
# 1. 공동 six-stem A0 pretrain
msr train --stage pretrain --variant A0 --config configs/moises6.json \
  --manifest /absolute/path/to/prepared_moises6/manifest.json \
  --run-dir runs/moises_a0 --steps 1000 --validation-every 100 --device cuda

# 2. 같은 parent에서 A1/A2 adapter 학습; backbone/head는 frozen
msr train --stage adapt --variant A1 \
  --parent runs/moises_a0/checkpoint_best.pt \
  --manifest /absolute/path/to/prepared_moises6/manifest.json \
  --run-dir runs/moises_a1 --steps 1000 --device cuda

msr train --stage adapt --variant A2 \
  --parent runs/moises_a0/checkpoint_best.pt \
  --manifest /absolute/path/to/prepared_moises6/manifest.json \
  --run-dir runs/moises_a2 --steps 1000 --device cuda

# 3. 후속 MSR 전이: 공통 six-stem A0 parent에서 출발
msr train --stage finetune --variant A2 \
  --parent runs/moises_a0/checkpoint_best.pt \
  --manifest /absolute/path/to/msr_training_pairs.json \
  --run-dir runs/msr_a2 --steps 1000 --device cuda
```

MSR에서도 A0/A1/A2를 비교하려면 마지막 명령의 variant와 run directory만 바꾸고 조건을 맞춘다. 현 CLI는 MSR 단계에서 **all-parameter fine-tuning**을 하며, 이미 학습한 Moises adapter의 선택적 warm-start는 첫 protocol에 넣지 않았다. 네 대응 head(vocals/bass/drums/guitar→guitars)만 옮기고 나머지는 공통 새 초기값을 사용한다. Piano→keyboards의 부분 이전은 자동으로 하지 않는다.

Frozen A0에는 가짜 optimizer update를 만들지 않고 부모 checkpoint 자체를 평가한다.

## MSR paired manifest

준비된 정렬 stereo 파일을 사용한다. `sources` 순서는 다음과 같아야 한다.

```json
{
  "schema_version": 1,
  "task": "msr8",
  "sample_rate": 48000,
  "sources": ["vocals", "guitars", "keyboards", "bass", "synthesizers", "drums", "percussion", "orchestral"],
  "tracks": [{
    "id": "degraded-view-a",
    "group": "canonical-song-a",
    "split": "train",
    "provenance": "training_raw_degraded_pair",
    "length": 480000,
    "mixture": "audio/degraded.wav",
    "targets": {
      "vocals": ["audio/raw_vocals.wav"],
      "guitars": [], "keyboards": [], "bass": [],
      "synthesizers": [], "drums": [], "percussion": [], "orchestral": []
    }
  }]
}
```

실제 manifest에는 별도 validation/test 곡도 필요하다. `[]`는 **명확히 부재하는 source**이며 annotation 누락을 뜻하지 않는다. 모든 파일은 선언한 sample rate·stereo·length와 일치해야 한다. 상대경로는 manifest의 디렉터리를 기준으로 해석한다. `mixture`는 MSR에서 필수다. Raw target 합으로 degraded input을 대체하지 않는다. `provenance=official_validation|official_test`를 train에 넣으면 거부한다. Dataset 간 동일 곡의 자동 acoustic matching은 구현하지 않았다.

## 평가·진단·복원

```bash
msr evaluate --checkpoint runs/moises_a0/checkpoint_best.pt \
  --manifest /absolute/path/to/prepared_moises6/manifest.json --split test --output runs/a0_test.json
msr evaluate --checkpoint runs/moises_a2/checkpoint_best.pt \
  --manifest /absolute/path/to/prepared_moises6/manifest.json --split test --output runs/a2_test.json
msr compare --baseline runs/a0_test.json --candidate runs/a2_test.json --output runs/paired_a2_a0.json

msr infer --checkpoint runs/msr_a2/checkpoint_best.pt \
  --input /absolute/path/to/degraded_stereo_48k.wav --output-dir runs/restored_example
```

`evaluate`의 `--gate-off`, `--intervention frequency_shuffle`, `--intervention kv_joint_permutation`, `--intervention wrong_track`으로 추가 block만 조작할 수 있다. Coordinate shuffle은 동일 frame 안의 frequency-coordinate cyclic permutation이며 content/time/mask와 common trunk PE를 바꾸지 않는다. Wrong-track은 별도 held-out song group이 있어야 한다. Joint KV permutation 결과는 수치 오차 범위에서 같아야 한다. 나머지 조작으로 성능이 내려가도 인간 청각 기전의 증거라고 해석하지 않는다.

현재 평가는 full-track sufficient statistics의 joint-stereo waveform SNR/SI-SDR, MAE, output RMS와 곡 단위 reconstruction loss다. **Official Multi-Mel-SNR, Zimtohrli, FAD-CLAP, MOS는 미구현**이다. 별도 공식 evaluator에 출력 WAV를 제공해야 한다. Silent target의 dB 지표는 null로 두고 output RMS/MAE를 보존한다. `compare`는 같은 곡을 짝지으며 반복 view를 song group 안에서 먼저 평균한다.

## 인과성·재개·비용

- Analysis token은 frame 끝 sample까지 읽는다. Synthesis의 보수적 지연 상한은 `n_fft-1` sample. 기본값에서 약 21.31 ms이며 전체 연산이 실시간이라는 주장은 아니다.
- 모든 arm에서 최대 injection 문맥을 포함한 동일 pre-roll을 사용한다. 기본값은 pre-roll 35,584 samples, target 96,000 samples, 지연용 실제 뒤 문맥 1,024 samples다. 미래 sample을 즉시 사용할 수 있다는 의미가 아니라 output latency를 명시한 것이다.
- 위치 원점은 **pre-roll을 포함한 고정 crop/window 시작**이다. 서로 다른 chunk 길이/원점에서 동일한 출력이라는 보장은 없다. Streaming KV cache와 full-pass/chunk-pass 동등성은 구현하지 않았다.
- 실제 GPU에서는 먼저 기본 config, 목표 길이+pre-roll의 `smoke --device cuda`로 메모리/시간을 확인한다. CPU short-clip 통과를 2초 GPU profile 통과로 해석하지 않는다. FP32만 정식 실행 경로이며 AMP/DDP는 현재 범위 밖이다.
- Checkpoint, run contract, source hash, manifest/audio-content hash, data crop 순서와 validation trajectory를 저장한다. 큰 데이터셋은 startup content hashing 시간이 든다.
- 새 run은 빈 디렉터리를 요구한다. 같은 run 재개는 아래처럼 실제 completed step보다 큰 `--steps`로 실행한다. 기존 batch/lr/seed/crop/loss/device/코드/데이터 조건을 유지해야 한다.

```bash
msr train --stage adapt --variant A2 \
  --manifest /absolute/path/to/prepared_moises6/manifest.json \
  --run-dir runs/moises_a2 --resume runs/moises_a2/checkpoint_last.pt \
  --steps 2000 --device cuda
```

코드가 바뀐 checkpoint를 같은 실험의 exact resume으로 통과시키지 않는다. Architecture 의미를 바꾸는 개발에서는 checkpoint schema도 갱신해야 한다.

## 비평·수정 기록

1. [수학·신호·인과성](reviews/01_math_and_causality.md)
2. [실험 비교·데이터·checkpoint](reviews/02_experimental_controls.md)
3. [실행 비용·재현·전달](reviews/03_execution_and_handoff.md)

동일 에이전트의 순차 자체 검토다. 실제 데이터 성능이나 외부 독립 심사를 수행한 것처럼 표현하지 않는다.

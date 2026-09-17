# SW T/C vs STFT pilot — 실행 및 검증 기록

## 이번 코드가 하는 일

하나의 명령으로 아래를 **순차 실행**한다. B0는 재학습/재평가하지 않고 이미
완료된 `bsr_b0_full_20260917_v1`을 사용한다.

1. 같은 SW-Fixed에서 T/C adapter만 200 update 학습 → validation28곡 전체 평가.
2. 원본 SW-Fixed를 새로 로딩하여 STFT adapter만 같은200 update 학습 → 같은28곡 평가.
3. `TC−B0`, `STFT−B0`, **`TC−STFT`**의 곡별·stem별 차이를 저장한다.

Zero-gate/gradient/동결 확인은 별도의 연구 실험이 아니라 시작부/학습 중 자동 검사다.
첫 step의 memory gradient=0은 gate가 정확히0인 초기화에서 정상이다. 3 step까지
memory에 gradient가 도달하지 않으면 실패로 중단한다.

## 고정한 비교 조건

| 항목 | 설정 |
|---|---|
| Parent | 기존 SW-Fixed, 모든 parameter frozen, dropout 포함 eval mode |
| 주입 위치 | band_split 직후, 원본 time/frequency Transformer 앞 |
| Query / Key·Value | parent band token / 해당 frame의 추가 feature token |
| Attention | width64, heads4, frame마다 cross-band attention, dropout0 |
| 추가 parameter | **TC644,609 = STFT644,609**, 모든 trainable tensor 초기값 동일 |
| Gate | scalar, 초기0, AdamW로 학습 |
| Train 데이터 | 기존194곡, 매 cycle의 song shuffle 후 각곡 random crop |
| 구간 순서 | seed20260917, 두 arm이 하나의 `schedule.json`을 공유 |
| Crop / batch | **588800 sample≈13.35초 / batch1**, parent 기본 chunk와 동일 |
| 예산 | arm당200 update, 첫194회에194곡 모두 방문; 약44.50분 입력/arm |
| Optimizer | AdamW, lr1e−4, weight_decay0, grad clip1.0 |
| Loss | 원본 SW 방식 waveform L1 + complex multi-STFT L1의 합, 6stem 모두 |
| Precision | CUDA fp16 forward, float32 loss, GradScaler |
| 평가 | 고정200번째 checkpoint, 28곡 전체, B0와 같은 sample rate/chunk/overlap/metrics |
| 제외 | 새 PE/LieRE, source head 변경, output fusion, 증강, validation-best 선택 |

TC memory는 short waveform projection 한 token과 Gaussian chirp(−4,+4)의
band별 token이다. STFT memory는 short512 Hann STFT 한 token과 **1024/2048
Hann 창을 같은2048 FFT grid에 배치한 두 spectrum**의 band별 token이다.
기존 tiny prototype의 long-STFT 단순 복제 control은 사용하지 않는다.

Short real STFT의 DC/Nyquist imaginary0 두 좌표는 제외하여 N개의 실수로
표현한다. Interior real/imag는√2를 곱한 real-unitary packing, short Hann 창은
RMS-normalized다. Long bank 창은 각각 L2-normalized다. 이로써 short projection의
parameter 수도 waveform과 같다. Parameter/token 수가 같다는 것이 effective rank,
입력 통계, FLOPs, 표현 정보가 모두 같다는 뜻은 아니며 wall time/VRAM을 별도 기록한다.
Fixed bank 정의의 차이는 실험 조건이며 T/C의 유리함을 전제하지 않는다.

### 메모리·데이터 처리

원본 Transformer와 mask head의 backward 계산은 유지한다. Downstream block만
activation checkpointing으로 재계산하며, **band_split/injection hook은 checkpoint하지
않는다**. 일시 hook을 제거한 뒤 backward에서 adapter가 빠지는 오류를 피하기 위해서다.
원본 YAML의 `use_torch_checkpoint`는 false 그대로 유지한다.

48k prepared cache를 다시 만들지 않는다. 학습에서는 필요한 crop 주변만 읽고,
polyphase 시작점을 맞춘 충분한 halo와 함께44.1k resample한다. Full-file resample
후 crop한 결과와 일치하는지 테스트했다. 평가에서는 기존 B0 함수를 그대로 사용한다.
Train의 파일 hash는 arm별 첫 사용 때 기록하며 두 arm 간 동일성을 확인한다.
Validation hash는 이전 B0와 동일해야 한다. 환경 version이나 B0 evaluator hash가
바뀌면 비교를 중단한다. 기존 `src/msr`, `baseline.py`, `sw_backend.py`는 수정하지 않았다.

## 실행 — 기존 Git/기존 컨테이너만 사용

새 패키지 설치, 새 venv, tar 전송, Docker 재생성은 필요 없다.
현재 서버의 B0 실행 환경·mount·파일을 그대로 사용한다.

### 1. 로컬 Mac: 코드 반영

Assistant는 commit/push를 실행하지 않았다. 아래는 관련 새 파일만 포함한다.
예상1분 이내(인증/네트워크 제외).

```bash
cd /Users/kuseunghyeok/Downloads/RESEARCH/multiencoder-transformer
git add src/bsr_probe/sw_adapter.py src/bsr_probe/pilot.py configs/bsr_sw_pilot.json tests/test_sw_pilot.py reviews/SW_TC_STFT_PILOT_20260917.md
git commit -m "Add matched SW TC versus STFT adapter pilot"
git push origin main
```

### 2. 서버 HOST: 기존 저장소 폴더에서

```bash
git pull --ff-only origin main
```

예상1분 이내. Host project 절대경로는 새로 추정하지 않는다. 기존 컨테이너의
`/workspace/project`로 mount한 **그 저장소**에서 실행한다. 충돌/인증 오류면 멈춘다.

### 3. 기존 CONTAINER: 확인 후 한 번 실행

먼저 기존 파일/GPU 확인, 예상1분 이내:

```bash
cd /workspace/project
ls -lh bsroformer/bsroformer/BS-Rofo-SW-Fixed.ckpt bsroformer/bsroformer/BS-Rofo-SW-Fixed.yaml
ls -lh /workspace/cache/moises6_48k_seed20260915_v3/manifest.json /workspace/runs/bsr_b0_full_20260917_v1/summary.json
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

정상이면:

```bash
PYTHONPATH=/workspace/project/src python -u -m bsr_probe.pilot --config /workspace/project/configs/bsr_sw_pilot.json --output /workspace/runs/sw_tc_stft_pilot_200_20260917_v1 --device cuda:0
```

이 명령 안에 두 arm의 학습·평가·비교가 모두 포함된다. **TC만 실행되는 것으로
착각해 별도 STFT 명령을 동시에 띄우지 않는다.** 순서는 TC학습→TC평가→STFT학습→STFT평가다.

예상 총30–90분: B0의 전체 평가10.2분/회는 실측이나 adapter backward 및 입력
파일 최초 hashing은 아직 서버에서 미측정이다. 두 번 평가만 최소 약20분 수준이고
training/IO가 추가된다. 첫3 step에서 `startup_training_profile`에 arm 학습 시간
추정과 peak allocation을 기록하므로 그 실측으로 갱신한다. CPU 검증의 속도를 GPU
학습 시간으로 전용하지 않는다.

첫 step에는 zero-gate 두 번의 추가 forward와 음원 hashing이 있어 다음 step보다
느릴 수 있다. CUDA OOM/NaN/무결성 실패면 중단하며, crop/precision을 자동으로
바꾸지 않는다. **Inference peak1.50GiB가 training도1.50GiB라는 뜻은 아니다.**

다른 컨테이너 terminal에서 진행 확인:

```bash
tail -f /workspace/runs/sw_tc_stft_pilot_200_20260917_v1/tc/progress.jsonl
```

TC 평가 중이면:

```bash
tail -f /workspace/runs/sw_tc_stft_pilot_200_20260917_v1/tc/evaluation/progress.jsonl
```

STFT는 같은 위치의 `tc`를 `stft`로 바꾼다. 최종 root `summary.json`에
`complete: true`가 있어야 두 arm 모두 완료한 것이다.

## 출력과 중단 처리

| 파일 | 의미 |
|---|---|
| `contract.json`, `settings.json` | parent/B0/version/code/학습 조건 |
| `schedule.json` | 두 arm에 공통인200개 song/start/crop 목록 |
| `tc/startup_checks.json`, `stft/startup_checks.json` | 실제 첫 crop의 zero-gate 검사 |
| `<arm>/progress.jsonl`, `train_summary.json` | loss/gate/gradient/시간/동결/VRAM |
| `<arm>/training_audio_hashes.json` | train 음원 provenance |
| `<arm>/last.pt` | adapter+optimizer+scaler, parent weight 중복 저장 없음 |
| `<arm>/evaluation/` | 28곡 점수·첫2곡10초 prediction WAV |
| `<arm>/vs_b0.json` | arm−B0의 paired 차이 |
| `comparison.json` | TC−B0, STFT−B0, **TC−STFT**, 부재 stem RMS |
| `summary.json` | 전체 완료 여부 |
| `failure.json` | 실패/중단 시 traceback |

Checkpoint는50 update마다 및200번째에 저장하며 평가 전에 보존한다. 이번 CLI는
자동 resume를 구현하지 않았다. 실패 결과를 지우거나 같은 output에 재실행하지 말고
`failure.json`·logs를 가져온다. 다른 조건으로 재시도할 때는 새 run 이름을 사용한다.
단순히 학습이 끝났다고 더 나은 representation이라고 판정하지 않는다. 비교의 primary
질문은 **TC−STFT**이며 stem별 median/mean/악화곡·부재 stem 누출도 함께 본다.

## 로컬로 가져오기

최근 B0에서 확인한 기존 tunnel/mount가 유지된 경우 아래를 **로컬 Mac**에서 실행한다.
Host run root가 바뀌었다면 실제 mount를 확인하며 `/workspace/runs`를 rsync source로
사용하지 않는다. 전송은 예상약100MB, 네트워크에 따라1–5분. Tunnel terminal 유지.

```bash
mkdir -p /Users/kuseunghyeok/Downloads/RESEARCH/multiencoder-transformer/msr-a0-pilot/runs/sw_tc_stft_pilot_200_20260917_v1
rsync -avP -e "ssh -p 22022 -o HostKeyAlias=research-inner-via-gateway" intern_2026_summer@127.0.0.1:/private/intern_2026_summer_private_dataset/ku/msr/runs/sw_tc_stft_pilot_200_20260917_v1/ /Users/kuseunghyeok/Downloads/RESEARCH/multiencoder-transformer/msr-a0-pilot/runs/sw_tc_stft_pilot_200_20260917_v1/
```

## 로컬에서 실제 확인한 범위

- 전체53 tests 통과 (7개 새 pilot test 포함). No network/weights tiny fixtures로
  parameter/init 일치, checkpointed/uncheckpointed gradient 일치, parent 동결,
  zero-gate, resampling phase/boundaries, schedule,3-update artifacts, 평가 순서 및
  input hash 불일치 거부, invalid SI-SDR 표기를 검사했다.
- 별도로 **실제 SW-Fixed 1,939 weight 항목을 strict load**하고, CPU fp32에서
  각 arm8192-sample synthetic 입력으로3 update backward를 수행했다.
- 두 arm644,609 parameters. Initial zero-gate 최대오차0,3 update 후 gate-off
  복구오차0, parent persistent-state hash 불변, parent gradient 없음.
- TC memory gradient norm: 0 → 4.662e−5 → 9.338e−5.
  STFT: 0 → 4.671e−5 → 9.355e−5. 첫0은 의도한 초기화다.
- 이는 실가중치 연결 검사이지 실제 음악 성능 결과가 아니다. **Full13.35초 CUDA
  backward와200-update 음악 학습은 아직 미실행**이며 서버 첫 step에서 검증된다.
- 이전 B0 결과를 보존했고 기존 학습 code나 가중치는 수정하지 않았다.

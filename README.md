# NASA battery SOH forecasting with Chronos

02는 Chronos T5 zero-shot, 03은 배터리 단위 leave-one-battery-out(LOBO) LoRA 실험입니다.
데이터 처리·추론은 `src/soh_experiment.py`를 공유하고, 현재 03의 학습·선택은 `src/soh_checkpoint.py`를 사용합니다.

## 현재 LoRA: EOS loss 제외 + best checkpoint

03은 출력 EOS loss를 제외하고 validation을 학습에 합치지 않습니다. 학습률 `1e-5`, `3e-5`, `1e-4`를
각각 최대 100 optimizer steps까지 학습하고, 10 steps마다 세 validation seed의 평균 CRPS를 계산합니다.
네 fold에서 학습률과 best checkpoint를 모두 선택한 뒤 그 가중치를 재로딩해 평가합니다. Full-data refit은 하지 않습니다.

설정은 `configs/lora_checkpoint.json`, 세부 기준은 [현재 프로토콜](docs/lora_checkpoint_protocol.md),
결과는 [학습률 비교 보고서](results/checkpoint_lora/report.md)에 있습니다. Step 0도 후보이고, 학습 seed는 42 하나입니다.
세 고정 후보를 모두 실행하므로 Optuna는 사용하지 않습니다.

이번 실행에서는 네 fold 모두 `1e-4`가 선택됐고, B0005/B0006/B0007/B0018의 선택 step은 각각 100/90/90/80입니다.
Zero-shot 대비 중앙값 RMSE는 `0.016307 → 0.014355`(11.97% 감소), CRPS는 `0.008769 → 0.007461`(14.92% 감소)였습니다.
네 배터리 모두 중앙값 RMSE가 개선됐지만, 80% 예측구간 포함률은 `80.32% → 76.53%`로 낮아졌습니다.
재로딩 중 발생한 sample 한 개의 불일치와 평가 재현성 검사 보완은 결과 보고서에 기록했습니다. 최종 실행에서는 validation 배열 36개가 모두 일치했습니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py zeroshot
.\.venv\Scripts\python.exe scripts\run_checkpoint_lora.py
```

## 이전 기록: 2026-09-29 EOS 분리 진단

02의 zero-shot 기준선과 03의 기존 LoRA·EOS 분리 진단을 보관합니다.
노트북 마지막 절에서 당시 저장된 결과를 읽을 수 있습니다. [당시 정리 문서와 04 비교 조건](docs/02_03_closeout.md),
[상세 진단 보고서](results/diagnostics/20260929/diagnosis.md)에 결론과 한계를 모았습니다.

| 모델 | 중앙값 RMSE | 평균 예측 RMSE | CRPS | 중앙값 R² ↑ | 평균 예측 R² ↑ |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 0.016307 | 0.018540 | 0.008769 | 0.676844 | 0.616669 |
| 기존 LoRA | 0.016176 | 0.294040 | 0.017684 | 0.668538 | -65.167624 |
| EOS loss 제외 LoRA | 0.014894 | 0.015011 | 0.007552 | 0.731631 | 0.732641 |

고정 XPU 실행에서 네 배터리·세 추론 seed를 동일 가중치로 평균했습니다. 학습 seed는 42 하나입니다.
EOS 제외는 사후 원인 진단이며, 모든 배터리·지표가 개선된 것은 아닙니다. 80% 구간 포함률은 약 74.46%로 낮아졌습니다.
**04 Full fine-tuning은 아직 실행하지 않았습니다.** 위 EOS 제외 수치는 과거 full-data 학습 결과이며 현재 best-checkpoint 점수와 구분합니다.

`configs/experiment.json`의 학습률·epoch·patience 기본값은 **기존 EOS 포함 + 전체 데이터 refit 실험**의 기록입니다.
현재 학습에서는 `configs/lora_checkpoint.json`의 학습률 후보와 step 예산을 적용합니다.
EOS 제외 진단은 `scripts/diagnose_training.py`로 별도 실행했으며, 명령과 산출물은 상세 진단 보고서에 있습니다.

## R² 성능평가

현재 best-checkpoint LoRA의 macro R²는 중앙값 예측 **0.733519**, 평균 예측 **0.705528**이다. Zero-shot은 각각 **0.676844**, **0.616669**다.
배터리·seed별 모든 window/horizon을 펼쳐 `R² = 1 - SSE/SST`를 계산한 뒤 배터리에 같은 가중치를 준다. 1은 완벽한 예측, 0은 평가 target 평균을 항상 예측한 수준이다. 음수도 가능하며 상수 target은 NaN이다. Horizon/window 평가는 각 범위의 평균을 사용한다.

[R² 포함 평가표](results/evaluation_r2/README.md)에 baseline, 중앙값·평균 예측, 배터리·horizon·window, 현재 validation 후보·이력을 저장했다. 기존 실행의 해시와 원시 예측은 보존한다. R²는 보고용이며 checkpoint 선택 기준은 CRPS다.
`python scripts/add_r2_metrics.py`로 재학습 없이 재생성한다. 과거 20-sample LoRA는 원시 예측이 없어 반올림된 RMSE로 계산한 근삿값임을 별도 CSV에 표시했다.

## 환경과 실행

기존 `.venv`는 그대로 사용할 수 있습니다. 현재 확인한 환경은 Python 3.12.10과 Intel Arc B390 XPU입니다.
새 환경에서는 장치에 맞는 PyTorch를 먼저 설치한 뒤 나머지 패키지를 설치하세요.
Intel GPU용 설치 인덱스는 [PyTorch 공식 안내](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/get_start_xpu.md)를 따릅니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/xpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

다른 GPU 또는 CPU 환경은 해당 장치용 PyTorch를 사용하세요. `device: auto`는 XPU, CUDA, CPU 순서로 선택합니다.
requirements는 이번에 확인한 직접 의존성 버전이며 완전한 transitive lockfile은 아닙니다.

`data/raw/`에 `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat`이 필요합니다.
폴더명·데이터 위치 변경 뒤 저장된 실험을 다시 사용할 때는 [로컬 경로 정리](docs/local_workspace.md)를 참고하세요.
VS Code에서 `.venv` 커널로 02와 03을 순서대로 전체 실행하거나, 프로젝트 루트에서 다음을 실행합니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py zeroshot
.\.venv\Scripts\python.exe scripts\run_checkpoint_lora.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

현재 03 / `run_checkpoint_lora.py`는 호환되는 zero-shot 실행이 없으면 먼저 생성합니다.
`run_experiment.py lora` 또는 `all`은 과거 epoch/refit 실험을 재현하는 명령입니다.
완료된 호환 실행이 있으면 검증 후 재사용합니다. 새 반복 실행은 CLI에 `--force`를 붙이거나 노트북에서 `FORCE_RERUN = True`로 설정하세요.
모델이 이미 캐시에 있고 네트워크 접속 없이 실행하려면 `$env:HF_HUB_OFFLINE='1'`을 설정할 수 있습니다.

## 기존 epoch/refit 실험의 설정과 평가 방법

다음은 `configs/experiment.json`으로 재현하는 기존 실험의 기록입니다. 현재 03의 선택 방식은 위 프로토콜 문서를 따릅니다.

- SOH는 첫 관측 용량 기준 `Qk / Q1`입니다. context 64개, horizon 16개, stride 1입니다.
- 각 예측 시점마다 **실제 관측 context**를 제공합니다. 초기 64개만 알고 전체 수명을 예측하는 실험은 아닙니다.
- 모델 revision을 고정하며 zero-shot과 LoRA 모두 100 samples 및 추론 seed 42/43/44를 사용합니다.
- 보간 중앙값과 sample mean을 같은 원시 samples에서 계산합니다. 주 비교는 median RMSE/MAE입니다.
- 학습 seed 기본값은 `[42]`입니다. 학습 초기화까지 반복하려면 `[42, 43, 44]` 등으로 변경하세요. 추론 seed 반복과 학습 seed 반복은 구분합니다.
- 평가 배터리는 학습과 epoch 선택에서 제외합니다. 나머지 3개 배터리 각각의 마지막 20%를 validation target으로 사용합니다.
- Train target의 끝이 경계 이전, validation target의 시작이 경계 이후인 window만 사용합니다. 경계에 걸친 target window는 제외합니다. Validation context에는 그 시점까지 실제 관측된 과거를 사용할 수 있습니다.
- Validation의 배터리별 RMSE를 동일 가중치로 평균해 epoch를 선택합니다. validation samples=32, 고정 seed=31415이며 epoch 0도 후보입니다.
- 최대 10 epoch, patience 3, min_delta 1e-5입니다. 선택 후 base model과 adapter를 새로 생성하여 학습 배터리 3개 전체로 선택된 epoch만큼 재학습합니다.
- 최종 비교는 같은 test target과 같은 추론 설정을 사용합니다. Test 성능으로 epoch나 estimator를 선택하지 않습니다.

이 validation은 학습 배터리 내 시간순 검증입니다. 배터리 간 nested CV는 아닙니다. 새로운 배터리로의 일반화는 outer LOBO로 평가합니다.
중첩 window와 fold별로 중복되는 학습 배터리 때문에 결과를 수백 개 독립 표본처럼 해석할 수 없습니다.
Seed 고정만으로 다른 장치·드라이버·라이브러리 버전 간 bitwise 재현성을 보장하지 않습니다.

## 기존 epoch/refit 실행의 결과 보관 및 재검산

실행마다 `results/runs/<timestamp>_<kind>_<id>/`에 새 디렉터리를 만듭니다.
현재 best-checkpoint 실행의 후보별 기록·선택 동결·분포 점수·재로딩 검사는 [현재 프로토콜](docs/lora_checkpoint_protocol.md)의 파일 목록과 audit 명령을 사용하세요. 아래 표와 명령은 기존 실행 형식에 해당합니다.

| 파일 | 내용 |
|---|---|
| `manifest.json` | 완료 상태, 설정, 모델 revision, 데이터·소스·산출물 해시, 환경 |
| `predictions/*.npz` | context, target, cycle 번호, 원시 samples, median/mean, q10/q90 |
| `metrics.csv` | 배터리 및 seed별 RMSE, MAE, 80% coverage/구간 폭 |
| `horizon_metrics.csv` | horizon별 오차 |
| `window_metrics.csv` | 예측 시작점별 오차 |
| `macro_by_run.csv`, `summary.csv` | 배터리를 동일 가중치로 평균한 점수와 실행 간 변동 |
| `selection/*.json` | LoRA 검증 분할, loss, validation RMSE, 선택 epoch |
| `folds/*.json` | 최종 학습 배터리, loss, adapter 위치 및 해시 |
| `paired_comparison.csv`, `paired_macro.csv`, `paired_windows.csv` | 동일 조건 zero-shot/LoRA 비교 |

LoRA adapter는 `models/<run_id>/<battery>_train<seed>/`에 저장합니다. gitignore로 모델 가중치는 Git에서 제외됩니다.
데이터 또는 설정이 다르거나 산출물이 변조된 baseline은 재사용하지 않습니다. 중단된 실행은 완료된 결과로 취급하지 않습니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py audit --run results\runs\실행디렉터리
```

이 명령은 데이터와 파일 해시를 확인하고 원시 예측 samples에서 지표를 다시 계산합니다.
`src.soh_experiment.reload_adapter(config, adapter_path)`로 저장 모델을 불러올 수 있습니다.

추가로 다음 명령은 같은 seed로 adapter 초기화가 일치하는지, 각 fold의 저장 adapter를 다시 불러왔을 때
첫 추론 batch의 samples가 원본과 정확히 일치하는지 확인합니다. 전체 데이터 재학습 없이 수행하는 현재 환경의 재현성 검사입니다.

```powershell
.\.venv\Scripts\python.exe scripts\check_reproducibility.py results\runs\LoRA실행디렉터리
.\.venv\Scripts\python.exe scripts\summarize_run.py results\runs\LoRA실행디렉터리
```

두 번째 명령은 `results/improvement_summary.md`와 PNG 비교 그래프를 만듭니다.
실험 파일에는 mean estimator도 남기지만 드문 극단 samples가 평균을 크게 움직일 수 있으므로 median과 구분해 해석합니다.

## 이전 실험

개선 전의 수정 중이던 노트북과 실행 출력은 `notebooks/archive/20260923_before_improvements/`에 그대로 복사해 두었습니다.
`results/metrics/`의 기존 결과도 보존합니다. 과거 실험은 20 samples 및 lower median을 사용했으므로 새 실험과 점수만 직접 비교하지 마세요.
검토 내용은 `results/notebook_review_2026-09-23.md`에 있습니다.

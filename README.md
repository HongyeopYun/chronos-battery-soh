# NASA battery SOH forecasting with Chronos

02는 Chronos T5 zero-shot, 03은 배터리 단위 leave-one-battery-out(LOBO) LoRA 실험입니다.
두 노트북은 `src/soh_experiment.py`의 같은 데이터 처리·추론·평가 함수를 사용합니다.

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
VS Code에서 `.venv` 커널로 02와 03을 순서대로 전체 실행하거나, 프로젝트 루트에서 다음을 실행합니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py all
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`zeroshot`, `lora` 모드로 개별 실행할 수 있습니다. 03 / `lora`는 호환되는 zero-shot 실행이 없으면 먼저 생성합니다.
완료된 호환 실행이 있으면 검증 후 재사용합니다. 새 반복 실행은 CLI에 `--force`를 붙이거나 노트북에서 `FORCE_RERUN = True`로 설정하세요.
모델이 이미 캐시에 있고 네트워크 접속 없이 실행하려면 `$env:HF_HUB_OFFLINE='1'`을 설정할 수 있습니다.

## 실험 설정과 평가 방법

설정 파일은 `configs/experiment.json`입니다.

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

## 결과 보관 및 재검산

실행마다 `results/runs/<timestamp>_<kind>_<id>/`에 새 디렉터리를 만듭니다.

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

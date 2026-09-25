# NASA battery SOH forecasting with Chronos

NASA 배터리 데이터에서 Chronos-T5의 zero-shot 예측과 LoRA 미세조정을 비교합니다.
배터리 하나를 평가용으로 제외하는 leave-one-battery-out(LOBO) 방식이며, 노트북과 CLI가 같은 학습·평가 코드를 사용합니다.

| 노트북 | 실행 내용 |
|---|---|
| [02_chronos_zeroshot.ipynb](notebooks/02_chronos_zeroshot.ipynb) | Zero-shot 및 LastValue·Linear 기준 모델 평가 |
| [03_chronos_lora_lobo.ipynb](notebooks/03_chronos_lora_lobo.ipynb) | 다중 검증 seed, 10/20 epoch, 배터리 균형 loss를 비교하는 LoRA 실험 |

## 설치

확인한 환경은 **Python 3.12.10, NVIDIA RTX 3080 12GB, PyTorch 2.11.0+cu128**입니다.
다음 명령은 Windows PowerShell 기준이며 프로젝트 루트에서 실행합니다. 기존 `.venv`가 있으면 생성 단계를 생략합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
```

CPU/XPU용 PyTorch가 이미 설치되어 있으면 CUDA wheel 설치 명령에 `--force-reinstall`을 추가합니다.
VS Code에서는 `.venv/Scripts/python.exe` 커널을 선택하고 패키지 변경 후 재시작합니다.
의존성은 확인한 직접 의존성 버전으로 고정되어 있으며, 전체 의존성을 잠그는 lockfile은 아닙니다.

`data/raw/`에 `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat`을 준비합니다.
원시 데이터와 모델 가중치는 Git에 포함하지 않습니다.

## 실행

노트북 02와 03을 순서대로 전체 실행하거나 다음 명령을 사용합니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py zeroshot
.\.venv\Scripts\python.exe scripts\run_robust_lora.py
```

03과 robust CLI는 호환되는 zero-shot 결과가 없으면 먼저 생성합니다.
완료된 결과는 설정·환경·산출물·adapter 해시를 검증한 뒤 재사용합니다.
새 실험에는 `--force`, 중단된 실험의 재개에는 `--resume`을 사용합니다.
노트북에서는 `FORCE_RERUN = True`로 새 실행을 시작할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_robust_lora.py --force
.\.venv\Scripts\python.exe scripts\run_robust_lora.py --resume results\runs\ablations\실행디렉터리
```

재개 시 완료된 학습 경로는 검증 후 재사용하고, 학습 도중 중단된 경로는 처음부터 다시 학습합니다.
코드나 설정이 달라진 미완료 실행은 재개하지 않습니다.
입력 검증 보완 전의 완료된 robust 실행은 명시적으로 확인한 이전 코드 해시에 한해 호환성을 유지합니다.

- 기본 장치는 `cuda`입니다. 사용할 수 없으면 오류로 원인을 알립니다. `auto`는 CUDA → XPU → CPU 순서로 선택합니다.
- CUDA의 네이티브 BF16 지원을 확인하고, 미지원 장치는 FP32로 실행합니다.
- 메모리가 부족하면 `batch_size`와 `inference_batch_size`를 낮춥니다. 추론 배치를 바꾸면 baseline과 LoRA를 같은 설정으로 다시 실행해야 합니다.
- 모델이 캐시에 있을 때 `$env:HF_HUB_OFFLINE='1'`로 오프라인 실행할 수 있습니다. CLI는 오프라인 모드를 강제하지 않습니다.

## 최신 실험 결과

[다중 검증 seed·학습 기간·배터리 균형 비교 보고서](results/lora_robust_report.md)의 최종 결과입니다.
학습 seed 42/43/44와 추론 seed 42/43/44를 사용했으며, 배터리와 각 seed에 동일한 가중치를 줍니다.

| 지표 | Zero-shot | 선택된 LoRA | Zero-shot 대비 |
|---|---:|---:|---|
| 중앙값 예측 RMSE | 0.016271 | 0.016152 | 0.730% 감소 |
| 중앙값 예측 MAE | 0.011556 | 0.011467 | 0.766% 감소 |
| 평균 예측 RMSE | 0.019495 | 0.019769 | **1.405% 증가** |
| CRPS | 0.008717 | 0.008665 | 0.596% 감소 |

최종 12개 모델 중 8개는 학습된 checkpoint, 4개는 epoch 0을 선택했습니다.
중앙값 예측과 CRPS의 개선은 작으며, 평균 예측의 오차는 오히려 증가했습니다.
이 결과만으로 LoRA가 전반적으로 더 안정적이라고 결론 내리지 않습니다.
네 배터리는 이전 실험에서도 확인한 데이터이므로 독립 benchmark나 통계적 유의성 검증으로 해석하지 않습니다.

## 실험 방법과 설정

공통 설정은 [configs/experiment.json](configs/experiment.json), 03의 추가 설정은 [configs/lora_robust.json](configs/lora_robust.json)입니다.

| 항목 | 기본 설정 |
|---|---|
| 모델 | `amazon/chronos-t5-base`, revision 고정 |
| SOH | 첫 관측 용량 기준 `Qk / Q1` |
| 예측 구간 | Context 64, horizon 16, stride 1 |
| 평가 샘플링 | 100 samples, 추론 seed 42/43/44 |
| 검증 샘플링 | Seed 31415–31419에서 각각 64 samples |
| 학습 | LoRA rank 8, 학습률 `3e-5`, FP32 CE + `0.5 × KL(base \|\| adapted)` |
| Checkpoint 선택 | 중앙값·평균 예측 RMSE guard를 통과한 후보 중 CRPS 개선, `min_delta=1e-5` |
| 최종 모델 | 검증에서 선택한 checkpoint 사용, `refit_full_data=false` |

평가 배터리는 학습과 checkpoint 선택에서 제외합니다. 나머지 세 배터리 각각의 마지막 20%를 validation target으로 사용하며,
train/validation 경계에 걸치는 target window는 제외합니다. 각 예측 시점에는 그때까지 실제 관측한 context를 제공합니다.
초기 64개 관측만으로 전체 수명을 예측하는 실험과는 구분해야 합니다.

03은 아래 정책을 비교합니다. 10/20 epoch 정책은 동일한 학습 경로를 공유하며 조기 종료를 적용하지 않습니다.
검증 지표는 seed별·배터리별로 계산한 뒤 동일 가중치로 평균합니다.

| 정책 | 검증 seed 수 | 학습 상한 | 배터리 균형 loss | 용도 |
|---|---:|---:|---|---|
| `single_seed10` | 1 | 10 epoch | 미사용 | 진단용 |
| `multi_seed10` | 5 | 10 epoch | 미사용 | 최종 후보 |
| `multi_seed20` | 5 | 20 epoch | 미사용 | 최종 후보 |
| `balanced20` | 5 | 20 epoch | 사용 | 최종 후보 |

균형 loss는 각 window의 CE·KL에 `전체 window 수 / (배터리 수 × 해당 배터리 window 수)`를 곱합니다.
학습 seed 42에서 validation만으로 fold별 정책을 고정한 뒤, 선택한 설정을 seed 43/44에서도 학습합니다.
모든 fold의 선택과 최종 checkpoint가 준비된 후 평가 배터리의 예측을 생성합니다. 평가 점수로 정책을 다시 선택하지 않습니다.

Robust 공통 설정은 `max_epochs=20`, `battery_balance=false`, 서로 다른 검증 seed 2개 이상,
`selection_metric=guarded_crps`, `refit_full_data=false`를 요구합니다. 비교 조건을 깨는 설정은 학습 전에 거부합니다.
Epoch 0도 후보이며, 겹치는 window와 반복 seed를 독립적인 배터리 표본으로 취급하지 않습니다.

## 결과 확인과 검증

새 일반 실행은 `results/runs/`, robust·ablation 실행은 `results/runs/ablations/`에 저장합니다.
Adapter는 `models/<run_id>/` 아래에 저장하며 정확한 경로와 fold별 설정은 manifest에 기록합니다.
Git에는 코드와 요약 보고서를 포함합니다. **새 실행의 원시 예측과 adapter는 로컬 파일이므로, 새 clone에서 검산하려면 직접 실행하거나 해당 파일을 별도로 가져와야 합니다.**
기존에 추적하던 과거 결과는 참고 자료로 남아 있습니다.

| 산출물 | 내용 |
|---|---|
| `manifest.json` | 설정, 환경, 데이터·코드·산출물 해시, 최종 fold 및 adapter 정보 |
| `predictions/*.npz` | Context, target, 원시 samples, median/mean, 예측 구간 |
| `metrics.csv`, `horizon_metrics.csv`, `window_metrics.csv` | 배터리·seed·예측 horizon·시작점별 지표 |
| `selection/*.json` | Checkpoint 선택 이력 |
| `training/*.json`, `search_decisions.json` | Robust 학습 경로와 평가 전 고정한 정책 선택 |
| `policy_evaluation/` | 학습 seed 42에서 각 정책을 평가한 결과 |
| `paired_*.csv` | 동일 조건의 zero-shot/LoRA 비교 |

다음 명령의 `실행디렉터리`를 실제 실행 경로로 바꿉니다.

```powershell
# 원시 예측 재검산 및 adapter 해시 검사
.\.venv\Scripts\python.exe scripts\run_experiment.py audit --run results\runs\ablations\실행디렉터리 --verify-adapters

# CUDA에서 adapter 재로딩 후 첫 추론 배치의 정확한 일치 확인
.\.venv\Scripts\python.exe scripts\check_reproducibility.py results\runs\ablations\실행디렉터리

# Robust 보고서(Markdown, CSV, JSON) 생성
.\.venv\Scripts\python.exe scripts\summarize_robust_lora.py results\runs\ablations\실행디렉터리

# 회귀 테스트
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`audit`는 저장된 설정으로 검산하므로 기존 GPU나 코드 버전이 없어도 실행할 수 있습니다.
재현성 검사는 일반 LoRA·ablation·robust 실행을 지원하며 저장된 실행·fold별 설정을 사용합니다.
선택 사항인 `--config`는 저장된 전체 설정과의 일치를 확인합니다. 재현성 검사와 캐시 재사용은 현재 평가 환경·코드의 일치도 요구합니다.
Seed 고정만으로 다른 장치·드라이버·라이브러리 버전 사이의 정확한 일치를 보장하지 않습니다.

확인한 검증 범위는 다음과 같습니다.

- 회귀 테스트 **38개 통과**: CUDA 선택, 데이터 분할, loss, checkpoint 보존, 설정 검사, 캐시 종류 및 호환성 검사.
- [노트북 실행 기록](results/lora_robust_notebook_checks.json): 02의 5개, 03의 8개 코드 셀 실행 완료.
- [CUDA 재현성 기록](results/lora_robust_reproducibility_checks.json): 최종 12개 fold의 첫 추론 배치와 zero-shot 첫 배치가 저장된 samples와 정확히 일치. 산출물 142개와 최종 adapter 해시 검사 통과.

`lora_robust_completion_checks.json`의 31개 테스트는 실험 완료 당시의 기록이며, 이후 설정·재현성 검사 수정으로 현재 38개가 되었습니다.

## 이전 실험과 상세 기록

기본 `run_experiment.py lora`와 `all`은 단일 검증 seed를 쓰는 일반 LoRA 경로입니다. 03의 robust 실험은 `run_robust_lora.py`로 실행합니다.
학습률·KL·조기 종료를 비교하는 이전 ablation은 `run_lora_ablation.py`로 실행하고 `summarize_lora_ablation.py 실행경로`로 요약합니다.
후보별 overrides는 학습 관련 필드만 허용하며, 평가·검증·선택 조건 변경은 학습 전에 거부합니다.

| 기록 | 내용 |
|---|---|
| [CUDA 전환 검토](docs/notebook_review.md) | 노트북 02·03의 환경 및 평가 경로 검토 |
| [학습·검증 개선 기록](docs/cuda_improvements.md) | KL, 분포 검증, 선택 checkpoint 보존 |
| [CUDA 전후 비교](results/cuda_improvement_report.md) | 이전 설정과 개선 설정의 비교 |
| [LoRA ablation 보고서](results/lora_ablation_report.md) | 학습률·KL·학습 기간 비교 |
| [후속 검토](docs/lora_followup_review.md) | 검증 seed 민감도와 robust 실험의 배경 |

`scripts/compare_runs.py`에 `--before 이전실행 --after 이후실행`을 전달하면 동일 평가 조건의 실행을 비교할 수 있습니다.
`notebooks/archive/20260923_before_improvements/`와 기존 `results/metrics/`는 과거 기록입니다.
과거 XPU 실험은 샘플 수와 중앙값 정의도 다르므로 현재 CUDA 결과와 점수만 직접 비교하지 않습니다.

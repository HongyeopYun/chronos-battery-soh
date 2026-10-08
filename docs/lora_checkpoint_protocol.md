# LoRA 학습률 3개 비교와 best checkpoint 선택

## 사전에 고정한 조건

기존 진단에서 EOS loss와 full-data refit의 영향을 확인한 뒤, 아래 프로토콜을 평가 결과를 보기 전에 고정했다.

| 항목 | 설정 |
|---|---|
| 모델·데이터·추론 | `configs/experiment.json`의 기존 설정 |
| 학습률 후보 | `1e-5`, `3e-5`, `1e-4` 순서로 모두 평가 |
| 학습 배터리 | 평가 배터리를 제외한 세 배터리 |
| 학습 구간 | 각 학습 배터리의 시간순 앞쪽 구간 |
| Validation | 각 학습 배터리의 뒤쪽 20% target; 경계에 걸친 target window 제외 |
| EOS | 입력 EOS 유지, 마지막 출력 EOS label의 loss만 제외 |
| Loss 가중치 | 예측-only CE에 `16/17`을 곱해 미래값 토큰당 기존 가중치 유지 |
| LoRA | q/v, rank 8, alpha 16, dropout 0.05 |
| 업데이트 | batch 16, accumulation 1, 최대 100 optimizer steps |
| 검증 시점 | 0·10·20·…·100 steps; 조기 종료/pruning 없음 |
| 검증 예측 | seed 31415·31416·31417, 각각 32 samples |
| 선택 | 배터리별·seed별 CRPS의 동일 가중 평균 최소화 |
| 동점 처리 | `min_delta=1e-5`를 넘는 개선만 반영; 먼저 평가한 step/학습률 유지 |
| 최종 모델 | 저장된 best adapter를 재로딩해 사용; refit 없음 |
| 평가 | 네 fold의 선택을 모두 고정한 후 test 배터리 예측 |
| 반복 | 학습 seed 42, 추론 seed 42·43·44 |

세 후보만 필요하므로 Optuna 의존성을 추가하지 않고 고정된 세 값을 순회한다. 총 12개 후보 학습이다. 학습률 이외의 후보 초기화와 minibatch 순서는 동일하며, validation 전후 Python/NumPy/Torch 및 GPU 난수 상태를 복구한다.

Step 0이 가장 좋으면 미학습 adapter를 선택할 수 있다. 이는 LoRA 학습의 개선이 아니다. 학습률과 step 선택은 test 배터리의 점수를 사용하지 않는다. 중앙값·평균 RMSE와 coverage는 함께 기록하지만 이번 선택 조건에 추가 guard로 사용하지 않는다.

## 실행과 확인

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\run_checkpoint_lora.py
.\.venv\Scripts\python.exe scripts\run_checkpoint_lora.py --audit results\runs\실행디렉터리
.\.venv\Scripts\python.exe scripts\summarize_checkpoint_lora.py results\runs\실행디렉터리
```

동일한 완료 실행은 해시와 원시 예측 지표를 검산한 후 재사용한다. 새 실행은 `--force`, 중단된 실행은 `--resume 실행디렉터리`를 사용한다. Resume 시 완료 후보는 검증 후 재사용하고, 중단된 후보는 초기화부터 다시 실행한다. 코드나 설정이 달라진 실행은 이어서 사용하지 않는다.

- `trials/*/history.json`: 검증 시점별 CRPS·RMSE·coverage와 실제 step/노출 예제 수.
- `trials/*/selected_validation/*.npz`: 각 후보 best checkpoint의 검증 원시 예측.
- `validation_candidates.csv`: 세 학습률의 best validation 결과.
- `selection_frozen.json`: test 예측 시작 전 확정한 네 fold의 선택.
- `checkpoint_replay.json`: 선택 adapter를 재로딩한 뒤 검증 sample 배열이 저장 시점과 완전히 같은지 확인.
- `predictions/*.npz`: 선택 모델의 held-out 원시 예측과 중앙값/평균/분위수.
- `distribution_metrics.csv`: 동일 예측 조건의 zero-shot·선택 LoRA 지표.
- `manifest.json`: 설정·모델 revision·환경·소스·산출물·adapter 해시.

이번 모듈은 `src/soh_checkpoint.py`, 실행 설정은 `configs/lora_checkpoint.json`이다. 기존 `src/soh_experiment.py`의 epoch/refit 프로토콜과 진단 파일은 변경하지 않는다. 현재 03 노트북은 새 프로토콜을 실행하고, 02는 기존 zero-shot 기준선을 사용한다.

## 해석 범위

### R² 추가 평가

저장된 예측으로 중앙값·sample mean 각각의 Mean window R²를 계산한다. 각 예측창의 실제 미래 평균으로 `1 - SSE/SST`를 구한 뒤 유효 창별 점수를 단순평균하고 seed·배터리를 동일 가중 평균한다. 음수는 유지하고 상수 창은 NaN으로 제외한다. Horizon 한 개는 창당 값이 한 개이므로 R²를 정의할 수 없다.
Validation은 각 후보의 선택된 checkpoint에 남은 `selected_validation/*.npz`에서 같은 방식으로 계산한다. 원시 예측이 없는 다른 step의 과거 이력은 NaN이다. 집계 RMSE로 창별 R² 평균을 복원하지 않는다. R²는 보고용이며 후보 선택은 기존 CRPS 기준이다. 원본 기록을 보존하고 `results/evaluation_r2/`에 파생 평가와 출처를 저장한다.

### 현재 실행의 재로딩 검사 보완

이번 실행에서는 저장 가중치가 정확히 로드되는데도 실행 경로에 따라 validation sample 한 개가 달라 비트 단위 검사가 실패했다. 별도 프로세스에서 두 번씩 완전 일치한 경우도 있어 원인은 아직 확정하지 않았다.
학습·후보 선택을 끝낸 실행에 한해 `python scripts/resume_checkpoint_verified.py results/runs/실행디렉터리`로 평가를 이어갈 수 있다.
이 경로는 모든 adapter tensor의 완전 일치를 요구하고 sample 차이와 재생성 배열을 저장한다. Macro validation CRPS 차이는 기존 선택 `min_delta=1e-5` 미만이어야 한다.
이는 **held-out 평가 전에 정한 재현성 검사 변경**이며, 학습 코드·선택 결과·모델 가중치를 변경하지 않는다. 별도 평가 코드 해시와 적용한 기준을 manifest에 기록한다.
실제 차이와 중단 이력은 [결과 보고서](../results/checkpoint_lora/report.md)에 남긴다. 기본 학습 실행의 엄격한 재생성 검사는 그대로 보존했다.

기존 EOS 제외 진단은 validation을 포함한 full-data 학습이었다. 이번에는 validation을 남기므로, 두 결과의 차이를 학습률이나 refit 한 가지의 효과로만 해석할 수 없다. 과거 진단과 새 프로토콜의 결과는 구분해 보고한다.

학습 seed 하나의 소규모 후보 비교이며, 세 추론 seed는 학습 반복이 아니다. 네 배터리와 중첩 window는 이미 관찰한 데이터다. 이번 결과를 독립적인 새 벤치마크나 통계적 유의성 검증으로 해석하지 않는다.

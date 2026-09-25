# 최신 main의 노트북 02·03 재검토

이 문서는 개선 전 검토입니다. 이후 반영한 코드와 전체 CUDA 비교 실험은 [개선 기록](cuda_improvements.md)을 참고하세요.

2026-09-24에 `git fetch origin`으로 확인한 `origin/main`의 `f5bf4cc`를
현재 `cuda-fullft` 브랜치에 fast-forward로 반영했습니다. 두 노트북과 실제 실행 코드인
`src/soh_experiment.py`, 설정, 저장된 예측을 함께 검토했습니다.

앞선 검토는 원격을 갱신하기 전의 `65debfc`를 대상으로 했습니다. **최신 main에는 3번 노트북이 있습니다.**
2번의 점수 하드코딩·중복 추론 문제도 최신 main에서 이미 해결됐습니다.
앞선 로컬 변경은 `codex CUDA edits before refreshing origin/main 2026-09-24`라는 Git stash에 보관했습니다.

## 최신 코드에서 확인한 정상 동작

- 02·03 모두 공통 데이터 처리·추론·평가 함수를 사용합니다. 입력은 과거 SOH 64개, 예측 horizon은 16개입니다.
- SOH를 첫 관측 용량으로 정규화하며 미래 값으로 context scale을 계산하지 않습니다.
- 테스트 배터리는 해당 LOBO fold의 학습과 epoch 선택에서 제외됩니다.
- 학습 배터리의 시간순 validation target은 train context/target과 겹치지 않습니다.
  Validation context에 이미 관측된 과거를 사용하는 것은 명시된 rolling-origin 평가와 일치합니다.
- 모델·adapter 생성 전에 seed를 고정하고, 선택된 epoch로 새로운 모델을 전체 학습 데이터에 재학습합니다.
- Tokenizer는 context의 scale을 label에 적용하고 mask를 `-100`으로 처리합니다.
- 중앙값·평균, 여러 추론 seed, horizon별 지표와 예측구간을 이미 평가합니다.
- 원시 samples와 지표가 저장되어 있습니다. 기존 zero-shot 및 LoRA의 예측 지표 총 48행을
  실제 데이터와 NPZ에서 재계산해 RMSE/MAE와 일치함을 확인했습니다.
  저장 산출물의 해시도 zero-shot 17개, LoRA 28개 모두 일치했습니다.

## 우선 개선할 사항

### 1. LoRA의 예측 분포 악화 — 실험상 우선순위 높음

위치: `src/soh_experiment.py`의 `_validation_score()`, `select_epochs()`와
`results/runs/20260923T063628Z_lora_2df6cb/`.

현재 epoch 선택 기준은 **중앙값의 validation RMSE** 하나입니다. 저장된 XPU 실험에서
중앙값은 약간 개선됐지만 평균 예측과 예측구간은 크게 악화됐습니다.

| 지표 | Zero-shot | LoRA |
| --- | ---: | ---: |
| Median macro RMSE | 0.016307 | 0.016176 |
| Median macro MAE | 0.011607 | 0.010950 |
| Sample mean macro RMSE | 0.018540 | 0.294040 |
| 평균 80% 예측구간 폭 | 0.042410 | 0.188000 |
| 평균 80% coverage | 80.32% | 81.20% |

LoRA raw sample 범위는 약 `-12.68 ~ 14.49`이고, SOH가 2를 넘는 sample이 약 1.74%입니다.
이는 `Qk/Q1`의 이 데이터 범위와 크게 벗어납니다. Coverage 증가만으로 분포가 개선됐다고 보기 어렵습니다.

권장: **학습 배터리의 validation에서** CRPS 또는 pinball loss, 예측구간 폭,
비현실적인 sample 비율을 함께 추적하세요. 더 작은 learning rate, 정규화 강도 등을 같은 validation에서 비교할 수 있습니다.
이 검토에서는 테스트 결과에 맞춰 epoch·모델 선택 규칙·clipping을 변경하지 않았습니다.

### 2. 학습 seed 반복과 배터리별 개선 확인 — 실험상 우선순위 중간

위치: `configs/experiment.json`의 `train_seeds: [42]`.

추론 seed 3회는 학습 초기화의 변동성을 측정하지 못합니다. 현재 중앙값 macro RMSE 개선은
약 **0.80%**이고, B0007만 개선됐으며 B0005·B0006·B0018의 RMSE는 악화됐습니다.

| 배터리 | Zero-shot RMSE | LoRA RMSE |
| --- | ---: | ---: |
| B0005 | 0.012601 | 0.013148 |
| B0006 | 0.018900 | 0.019232 |
| B0007 | 0.011423 | 0.009454 |
| B0018 | 0.022303 | 0.022868 |

권장: 학습 seed도 여러 개로 반복하고 학습/추론 변동을 구분해 보고하세요.
같은 네 배터리의 성능을 보며 hyperparameter를 계속 선택하면 독립 테스트로서의 의미가 약해집니다.
중첩 window 320개를 독립 배터리 320개로 취급하지 않아야 합니다.

### 3. 저장 결과의 이식성과 재검산 경로 — 코드 개선 우선순위 중간

위치: `run_lobo()`의 `zeroshot_run`·`adapter_path` 저장,
`verify_saved_run()` 및 03의 결과 불러오기 셀.

기존 manifest에는 이전 PC의 사용자 디렉터리를 가리키는 Windows 절대 경로가 있습니다.
현재 PC에는 그 경로와 당시 학습 adapter가 없습니다. 또한 `verify_saved_run()`은 현재 환경과 source hash가
다르면 저장 예측의 재검산 이전에 중단합니다. 새로운 CUDA 실험의 캐시로 XPU 결과를 거부하는 것은 적절하지만,
**과거 결과의 재검산**과 **동일 조건 결과의 캐시 재사용**은 요구 조건이 다릅니다.

권장: 프로젝트 또는 실행 디렉터리 기준 상대 경로를 저장하고,
원시 NPZ/CSV 검산과 adapter 재추론 검사를 분리하세요. 위 수치 검산은 원시 NPZ를 직접 확인한 결과이며,
기존 XPU adapter의 재추론을 현재 PC에서 확인했다는 의미는 아닙니다.

추가로 `find_lobo()`는 LoRA 실행 자체만 검증합니다. 연결된 zero-shot 디렉터리가 삭제되면 LoRA 캐시를 찾은 후
노트북의 baseline 검증에서 실패할 수 있으므로, 연결된 baseline까지 확인한 뒤 캐시를 반환하는 편이 좋습니다.

## 이번 CUDA 수정

- requirements의 XPU 실험용 `torch==2.14.0`을 현재 검증한 `torch==2.11.0`으로 변경하고 CUDA 12.8 wheel 설치 안내 추가.
- 실행 설정을 `device: cuda`로 지정. CUDA 미사용 환경은 첫 설정 셀에서 커널·설치 안내와 함께 중단.
- `auto` 사용 시 CUDA를 우선 선택하며 CUDA 장치 인덱스를 검증.
- 네이티브 BF16 지원을 확인하고 미지원 CUDA GPU는 FP32 사용.
- 장치·정밀도·GPU 모델·compute capability·CUDA runtime을 manifest와 캐시 호환성 정보에 기록.
  기존 코드는 장치 종류만 비교하므로 다른 CUDA GPU/정밀도를 같은 조건으로 볼 여지가 있었습니다.
- 두 노트북과 노트북 생성 스크립트에 동일한 runtime 진단 반영.
- 과거 XPU 셀 출력은 현재 CUDA 코드의 결과로 혼동되지 않도록 제거. 원본 출력은 Git main 이력에 보존.

## 이번 실행 검증의 범위

환경: RTX 3080 12GB, CUDA 12.8, PyTorch 2.11.0+cu128, BF16.

- 단위 테스트 **11개 통과**, `pip check` 통과.
- 02 기본 추론 batch **16 windows × 100 samples × 16 horizon**의 shape·유한값 확인.
  이 batch에서 PyTorch peak allocated memory 약 **5.30 GiB**.
- 03과 같은 공통 함수로 실제 LoRA **1 optimizer step** 실행. 유한한 loss `2.140625` 확인.
- 학습 가능한 파라미터 **884,736개**, LoRA tensor 72개 갱신 확인.
  Frozen 파라미터의 gradient가 없고 표본으로 비교한 frozen tensor가 불변임을 확인.
- 학습 adapter를 저장·재로딩한 후 **2 windows × 4 samples × 16 horizon** 예측이 정확히 일치.
- 최신 두 노트북 JSON 및 모든 코드 셀 구문 검사 통과.

상세 smoke 결과는 로컬 `results/cuda_validation/smoke.json`에 저장했습니다.
**최신 설정의 전체 4-fold × 최대 10-epoch 실험은 다시 실행하지 않았습니다.**
위 성능 비교표는 Git에 저장된 기존 XPU 실험을 검산한 값이고, CUDA에서 새로 얻은 최종 성능이 아닙니다.

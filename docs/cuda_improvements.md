# CUDA 실행 및 LoRA 개선

이번 변경은 `f5bf4cc`의 CUDA 이식 후 코드를 비교 기준으로 합니다. 실험을 시작하기 전에 다음 설정을 고정했습니다.

## 변경 내용

1. **분포를 포함한 checkpoint 선택:** validation의 empirical CRPS가 감소하고 중앙값/평균 예측 RMSE가 epoch 0보다 악화되지 않는 후보만 채택합니다(허용오차 `min_delta=1e-5`). 판정은 학습 배터리의 시간순 validation에서만 수행합니다. 개선 후보가 없으면 epoch 0을 사용합니다.
2. **검증한 모델 유지:** 선택된 adapter를 즉시 저장하고 그 checkpoint를 그대로 최종 평가에 사용합니다. 선택 후 전체 데이터 refit으로 모델이 바뀌지 않습니다. `refit_full_data=true`는 선택 옵션으로 남깁니다.
3. **학습 안정화:** learning rate를 `1e-4`에서 `3e-5`로 낮추고, FP32 CE에 `0.5 × KL(base || adapted)`를 추가했습니다. 기준 모델은 같은 context와 학습 label에서만 분포를 계산합니다. Padding은 KL에서 제외합니다. 기준 모델이 완전한 정답이라는 가정은 하지 않으며 학습으로 인한 급격한 분포 변화를 억제하는 정규화입니다.
4. **학습 반복:** 개선 모델은 seed 42/43/44로 반복합니다. 추론 seed 42/43/44와 구분해 기록합니다.
5. **실험 이식성과 검산:** 프로젝트 기준 상대 경로를 저장하고 기존 Windows 경로도 이전할 수 있게 했습니다. 다른 GPU에서의 원시 예측 검산을 캐시 재사용 검사와 분리했고, LoRA 캐시는 연결된 baseline까지 확인합니다.

## 계산과 검증

CRPS는 `E|X-y| - E|X-X'|/2`인 empirical distribution의 점수입니다. 샘플을 정렬하여 두 번째 항을 계산해 S×S 행렬을 만들지 않습니다.
이 구현은 [scoringrules의 CRPS 설명](https://scoringrules.readthedocs.io/en/latest/generated/scoringrules.crps_ensemble.html)에 제시된 empirical distribution 정의에 해당하며, 작은 입력에서 모든 sample 쌍을 직접 계산한 값과 비교하는 테스트가 있습니다.

KL은 [PyTorch KL 연산](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.kl_div.html)으로 vocab 축을 합산한 뒤 유효한 token만 평균합니다.
Padding 위치에 gradient가 전달되지 않는지, 동일한 분포의 KL이 0인지 검사합니다.
Checkpoint 테스트는 마지막 epoch가 더 나빠져도 검증에서 채택한 이전 checkpoint가 유지되는지 검사합니다.

Test 배터리의 결과로 clipping·epoch·가중치를 조정하지 않습니다. Raw samples는 원형으로 보존합니다.
Validation guardrail은 검증 데이터에 대한 조건이며, 새로운 테스트 배터리의 성능을 보장하는 조건은 아닙니다.

## 비교 실험

- 수정 전 구현과 설정을 `results/cuda_validation/baseline_implementation.py`, `baseline_config.json`에 보관하고 같은 RTX 3080에서 전체 4-fold를 실행했습니다.
- 개선 모델은 4-fold × 학습 seed 3개를 실행합니다. 두 방법 모두 inference seed 3개, 100 samples입니다.
- 직접 비교는 공통 학습 seed 42만 사용합니다. 개선 모델의 추가 seed 43·44는 별도의 안정성 표로 제시합니다.
- 설정 변경 전후는 learning rate, 정규화, checkpoint 선택·refit 정책이 함께 다르므로 특정 요소 하나의 인과 효과로 주장하지 않습니다.
- 원시 예측을 재검산한 결과와 그래프는 [CUDA 실험 보고서](../results/cuda_improvement_report.md)에 기록합니다.

실행별 raw NPZ와 adapter는 로컬 `results/runs/`, `models/`에 남으며 Git 추적에서는 제외합니다. 보고서와 요약 CSV/JSON/PNG는 추적 가능한 별도 파일입니다.

## 완료한 검증과 결과

수정 전 4-fold와 개선 후 4-fold × 학습 seed 3개를 모두 실행했습니다.
공통 학습 seed 42에서 평균 예측 RMSE는 `0.263034 → 0.019495`(92.59% 감소),
CRPS는 `0.017466 → 0.008717`(50.09% 감소)했습니다.
중앙값 RMSE는 1.48% 감소했지만 MAE는 1.82% 증가했습니다. 배터리별로도 개선과 악화가 함께 있습니다.

**12개 중 11개가 epoch 0을 선택했습니다. 공통 seed 42에서는 4개 모두 사전학습 모델을 유지하여
zero-shot과 같은 예측을 냈습니다. 확인된 효과는 나빠진 미세조정을 걸러내는 것이며,
새 LoRA 학습의 zero-shot 대비 우위는 확보하지 못했습니다.** B0006의 학습 seed 43만 epoch 5를 채택했습니다.

- 단위 테스트 18개, `pip check`, `git diff --check` 통과.
- 최신 02·03의 코드 셀 총 12개를 `notebooks/` 실행 위치에서 모두 실행. 완료된 CUDA 결과의 캐시 재사용과 그래프 생성 확인.
- 개선 LoRA의 배터리별 지표 72행·산출물 68개, zero-shot 지표 32행·산출물 17개 검증.
- 12개 adapter를 각각 재로딩하여 첫 추론 batch의 raw samples가 저장본과 정확히 일치함을 확인.
- 같은 seed의 adapter 초기화 및 zero-shot 첫 batch 반복 일치 확인.
- 수정 전후 zero-shot의 예측 파일 12개에서 모든 samples가 정확히 일치함을 확인.

검증 기록: [JSON](../results/cuda_improvement_validation.json). 이 결과를 보고 설정을 추가 조정하지 않았습니다.
비교 기준 설정과 재구성 패치: [실험 설정](cuda_experiment_protocol.json), [main 대비 control 패치](cuda_control_from_main.patch).

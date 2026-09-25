# LoRA 추가 개선 검토 — 2026-09-24

후속 진행(2026-09-25): 아래의 다중 seed 검증, 10/20 epoch 비교, 배터리 균형 loss를 모두 구현·실행했고,
캐시 종류 검사도 수정했습니다. [완료된 비교 결과](../results/lora_robust_report.md)와
[노트북 실행 확인](../results/lora_robust_notebook_checks.json)을 확인할 수 있습니다.
아래 내용은 수정 전에 수행한 진단 기록입니다.

현재 결과에는 추가 개선 여지가 있다. 다음 우선순위는 단일 검증 seed에 의존하는 선택을 안정화하는 것이다.
학습 기간 확대와 학습 배터리별 가중치 조정은 그다음에 분리해 확인할 후보다.
이 검토에서는 재학습이나 기존 checkpoint 변경 없이, 저장 결과 분석과 추가 validation 추론을 수행했다.

## 1. 확인된 문제: 검증 seed에 따른 분포 안정성 변화

기존 학습 seed 42에서 선택된 비영(非零) checkpoint 두 개를 대상으로,
학습 배터리의 validation만 사용해 추론 seed 31415–31419를 각각 64 samples로 평가했다.
평가 배터리의 target은 이 검사에 사용하지 않았다. 원래 seed 31415의 지표는 기존 기록과 정확히 재현됐다.
후보를 다시 고르거나 기존 실험의 평가 점수를 변경하지 않았다.

| 모델의 평가 제외 배터리 | 선택 checkpoint | 기존 기준 통과 seed | 5개 seed 평균으로 기준 통과 | 평균 예측 RMSE: base → LoRA |
|---|---|---:|---|---:|
| B0006 | KL 0, epoch 7 | 4/5 | 통과 | 0.013869 → 0.013758 |
| B0018 | KL 0.5, epoch 9 | 4/5 | **실패** | 0.010732 → 0.011914 |

B0018용 모델은 seed 31417에서 평균 예측 RMSE가 0.010226 → 0.016642로 **62.75% 증가**했다.
이 seed에서는 SOH>2 sample 비율도 base 0에서 LoRA 약 0.01028%로 증가했다.
5개 seed 평균에서도 평균 예측 RMSE는 **11.01% 증가**해 기존의 mean-RMSE guard를 통과하지 못한다.
같은 검사에서 중앙값 RMSE와 CRPS는 개선됐으므로, 중앙값 개선과 드문 극단 예측의 문제를 구분해야 한다.
4/5라는 수는 이 다섯 seed에서의 관찰 빈도이며 모집단 통과 확률 추정으로 해석하지 않는다.

원래 평가에서 중앙값 RMSE가 0.688% 개선된 결과 자체는 바뀌지 않는다.
다만 그 개선량의 **71.66%가 B0018의 오차 감소에서 나왔으며**, 새 validation 검사에서 이 fold의 분포 안정성 한계가 드러났다.
이번 수치는 다른 데이터 구간과 추론 seed를 사용하는 validation 진단이므로, 원래 test 점수에 섞어 평균내지 않는다.

**다음 실험:** 미리 정한 여러 validation seed의 배터리별 지표를 평균해 checkpoint와 후보를 선택한다.
첫 비교는 학습 설정을 고정하고 선택 방식만 바꾼다. 기존 checkpoint만으로 과거 모든 epoch를 재선택할 수는 없으므로,
새 학습에서는 epoch별 후보를 보존하거나 학습 중 여러 seed로 검증해야 한다.
극단 예측 진단과 평균 RMSE guard를 유지한다. 추가 sample 수나 seed 수를 늘렸다고 성능 개선이 보장되지는 않는다.

근거: [추가 진단 JSON](../results/lora_validation_sensitivity.json),
[seed별 CSV](../results/lora_validation_sensitivity.csv),
[`_validation_score`](../src/soh_experiment.py), [`validation_candidate_improves`](../src/soh_experiment.py).

## 2. 확인된 학습 상한: 10 epoch에서 끝나지 않은 후보

B0018용 최종 모델의 선택 epoch는 학습 seed 42/43/44에서 **9/10/10**이었다.
43/44는 마지막 10 epoch에서도 validation CRPS가 개선됐고, training loss도 직전 epoch보다 낮았다.
현재 기록에는 10 epoch 이후가 없으므로 더 학습했을 때의 성능은 알 수 없다.

**다음 실험:** 검증 방식을 먼저 고정한 뒤 max_epochs만 10→20으로 늘려 비교한다.
learning rate, KL, rank, patience를 동시에 바꾸지 않는다. 높은 학습률 후보는 이전 validation에서 분포가 흔들렸으므로,
우선 기존 낮은 학습률의 연장을 확인한다. B0018의 test 결과를 보고 그 fold만 특별 취급하지 않고 모든 outer fold에 같은 후보를 적용한다.

근거: [선택 epoch와 마지막 학습 기록](../results/lora_training_limits.csv).

## 3. 확인된 목적 가중치 차이: 학습은 window 수, 평가는 배터리 수

`_training_loader`는 모든 배터리의 window를 합쳐 균일하게 섞는다.
현재 시간 분할에서 B0005/B0006/B0007의 학습 window는 각각 55개, B0018은 26개다.
B0018이 학습에 포함되는 fold에서 해당 배터리의 데이터 비중은 **26/136 = 19.12%**다.
반면 validation과 최종 macro 지표에서는 각 배터리가 동일한 비중을 갖는다.

이 차이는 코드 오류라고 단정할 수 없지만, 모델을 고르는 목적과 학습 loss의 데이터 가중치가 다르다는 뜻이다.
**다음 실험:** loss를 배터리별로 평균한 뒤 동일 가중치로 합치거나, 배터리 균형을 맞춘 sampling을 별도 후보로 비교한다.
짧은 배터리를 반복 추출해도 독립적인 새 데이터가 늘어나는 것은 아니다.

근거: `src/soh_experiment.py:488`의 결합 및 DataLoader, `:550`의 배터리별 validation 평균,
각 selection JSON의 `splits`.

## 4. 확인된 구현 문제: 실험 종류를 확인하지 않는 일반 LoRA 캐시

`find_lobo`의 `*_lora_*` 패턴은 `*_lora_ablation_*` 디렉터리도 찾는다.
`verify_saved_run`은 kind가 `lora`인 경우에만 학습 설정을 비교하므로,
완료된 ablation을 일반 LoRA 결과로 잘못 받아들일 수 있다.
CLI는 실행이 끝난 뒤 ablation을 별도 폴더로 옮기지만, 직접 API 호출이나 이동 직전 중단은 이 보호를 우회한다.

격리된 임시 디렉터리에서 실제 `find_lobo`/`verify_saved_run`을 호출해 잘못된 종류의 결과가 수락되는 것을 재현했다.
이 재현에서는 평가 signature와 산출물 검산만 mock으로 대체해 종류 검사의 누락을 분리했다.
현재 실제 캐시는 정상 `lora` 실행 `20260924T004908Z_lora_9e14e5`를 선택했고,
완료된 ablation은 `results/runs/ablations/`에 있어 **현재 보고된 점수에는 영향이 없었다**.

**수정 지점:** `find_lobo`에서 `manifest.kind == 'lora'`를 명시적으로 요구하고,
ablation의 저장 위치를 생성 시점부터 분리한다. 이번 검토에서는 학습 핵심 코드와 기존 캐시 signature를 변경하지 않았다.

근거: `src/soh_experiment.py:769–788`, `src/soh_ablation.py:105`,
`scripts/run_lora_ablation.py:21–31`, [격리 재현 및 현재 캐시 확인](../results/lora_cache_review.json).

## 5. 후순위 후보: LoRA 적용 범위와 loss

현재 LoRA는 rank 8, `q`와 `v`에만 적용한다.
적용 범위를 넓히는 비교는 가능하지만 trainable parameter 수와 과적합 위험도 함께 바뀐다.
`q/v`와 더 넓은 적용 범위를 비교할 때 rank와 learning rate까지 동시에 바꾸지 않는 편이 원인을 구분하기 쉽다.
PEFT의 지원 범위는 [공식 LoRA 문서](https://huggingface.co/docs/peft/en/package_reference/lora)를 확인했다.
이 저장소에서 넓은 적용 범위가 더 좋다는 증거는 아직 없다.

현재 CE는 수치 값을 token으로 양자화하는 Chronos의 원래 학습 방식이다.
따라서 회귀 loss가 아니라는 사실만으로 버그로 볼 수 없다.
수치적으로 먼 bin의 확률을 억제하는 보조 loss는 연구 후보지만,
EOS/padding과 실제 SOH scale을 구분해야 하며 현재 실험으로 개선 효과가 검증되지는 않았다.
원래 학습 방식의 근거는 [Chronos 논문](https://arxiv.org/abs/2403.07815)이다.

## 재현

```powershell
.\.venv\Scripts\python.exe scripts\review_lora_validation.py results\runs\ablations\20260924T082717Z_lora_ablation_958398
```

이 명령은 기존 모델과 adapter를 로컬 캐시에서 읽어 validation 추론만 다시 수행한다.
추가 진단은 기본 설정이나 기존 `choices.json`, 예측 파일, 학습 가중치를 수정하지 않는다.
3번 노트북의 일반 실행은 기존 보수적 설정을 사용하고, ablation 결과는 별도 셀에서 확인하는 현재 구성을 유지했다.

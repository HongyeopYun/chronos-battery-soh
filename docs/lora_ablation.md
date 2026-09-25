# LoRA 후속 실험의 설계

앞선 CUDA 개선은 중앙값·평균 RMSE가 나빠지지 않으면서 CRPS가 개선된 checkpoint만 선택했다.
12개 모델 중 11개가 epoch 0을 선택했으므로, 이는 나쁜 미세조정을 거르는 효과이지
LoRA 학습이 zero-shot을 이긴 증거는 아니었다.

이번 실험은 그 원인을 구분한다. 결과는 [실행 보고서](../results/lora_ablation_report.md)에 기록한다.

## 완료된 실행의 결과

CUDA에서 후보 12개와 추가 seed 8개, 총 20개 학습 경로를 각각 10 epoch까지 실행했다.
최종 12개 모델을 3개 추론 seed로 평가한 결과, zero-shot 대비 macro median RMSE가
0.016271에서 0.016159로 0.688% 감소했다. Median MAE는 0.672%, CRPS는 0.543% 감소했다.
학습 seed 42/43/44 모두 macro median RMSE가 zero-shot보다 낮았다.

이전의 보수적 실험은 12개 중 11개가 epoch 0이었지만, 이번에는 9개가 학습된 checkpoint를 선택했다.
B0018의 RMSE 감소율이 1.428%로 가장 컸고, B0007은 사실상 변화가 없었다(-0.001%).
따라서 조사한 범위에서 작은 학습 이득은 확인했지만, 큰 폭의 개선이나 일반적인 우수성을 입증한 것은 아니다.

같은 학습 경로를 짧게 끊으면 놓치는 개선이 있었으므로 조기 종료가 일부 원인이었다.
반면 높은 학습률은 여러 fold에서 중앙값 오차를 줄이면서 평균 예측 오차를 크게 늘렸다.
KL 제거는 B0006 fold의 validation에서만 최종 선택됐다. KL을 일괄 제거할 근거는 없다.
기존 기본 설정은 보존하고, fold별 검증으로 고른 추가 실험과 결과를 별도로 사용할 수 있도록 했다.

## 변경한 요인

| 비교 | 기준 | 변경 |
|---|---|---|
| 조기 종료 | LR 3e-5, KL 0.5, patience 3 | patience 10 |
| 학습률 | 위의 긴 학습 설정 | LR만 1e-4 |
| KL 정규화 | 위의 긴 학습 설정 | KL 가중치만 0 |

최대 학습 기간은 모두 10 epoch다. 첫 비교의 짧은 학습 결과는 긴 학습 경로를
patience 3으로 다시 판단한다. 매 epoch 학습 RNG를 따로 고정하므로 검증 횟수나
후속 학습 여부는 이미 진행한 epoch의 loss·예측에 영향을 주지 않는다.

샘플링 설정, LoRA rank/alpha/dropout, 데이터 분할, FP32 loss 누적,
checkpoint 선택 기준과 full-data refit 비활성화는 고정한다.
학습률과 KL을 동시에 바꾸는 전체 grid나 두 요인의 상호작용까지 검사한 것은 아니다.

## 선택과 평가의 분리

1. 학습 seed 42로 각 outer fold의 세 후보를 학습한다.
2. 각 후보의 학습과 validation에는 평가 배터리의 series를 전달하지 않는다.
3. 각 후보 안에서 epoch 0 대비 중앙값·평균 RMSE guard를 통과하며 CRPS가 개선된 checkpoint를 저장한다.
4. 후보 간에도 validation CRPS로 선택한다. 개선이 `min_delta` 이하이면 사전에 적은 후보 순서를 유지한다.
5. 네 fold의 후보 결정을 `choices.json`에 저장하고 해시를 고정한다.
6. 선택된 설정을 학습 seed 43/44로 반복한다. 이때도 checkpoint는 해당 학습 배터리의 validation에서 선택한다.
7. 모든 학습을 끝낸 뒤에만 평가 배터리의 예측을 생성한다. 추론 seed는 42/43/44, 각 100 samples다.

후보 탐색 seed는 하나다. 추가 학습 seed는 선택된 설정의 안정성 확인이며 후보 탐색 전체의 반복은 아니다.
각 fold는 다른 validation 배터리 조합을 사용하므로 선택한 설정이 달라도 된다.
Epoch 0을 선택한 경우를 학습된 LoRA의 개선으로 세지 않는다.

## 남기는 진단

- 모든 epoch의 validation 중앙값 RMSE, 평균 RMSE, CRPS, coverage와 구간 폭.
- 같은 학습 경로에서 patience 3과 10의 선택 차이.
- 중앙값 RMSE만 최소화했을 때의 epoch와 평균 예측 오차. 이 값은 진단 전용이다.
- 평가 배터리별 성능, 학습 seed별 macro 점수, epoch 0 선택 수.
- 각 후보와 추가 seed의 설정·선택 파일·adapter 해시.
- 최종 12개 adapter의 재로딩 후 첫 추론 batch가 원본 samples와 일치하는지 검사한 결과.

`scripts/summarize_lora_ablation.py`는 후보 선택을 validation 기록에서 다시 계산하고,
평가 배터리가 validation에 포함되지 않았는지 검사한다. 평가 지표도 저장 NPZ에서 재계산한다.
완료된 실험은 일반 LoRA 캐시와 혼동하지 않도록 `results/runs/ablations/` 아래 보관한다.
대용량 원시 예측과 adapter는 기존 정책대로 로컬에 보관하고, 간단한 보고서는 Git에 포함한다.

## 해석의 한계

이 배터리들은 이전 실험에서도 관찰했다. 이번의 fold별 선택은 평가 점수를 보지 않지만,
전체 연구 과정에서 전혀 보지 않은 독립 test set으로 주장할 수는 없다.
겹치는 예측 window와 반복 추론 seed도 독립적인 배터리 표본이 아니다.
작은 점수 차이만으로 일반적인 LoRA 우수성이나 통계적 유의성을 주장하지 않는다.

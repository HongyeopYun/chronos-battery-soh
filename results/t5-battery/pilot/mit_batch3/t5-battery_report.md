# t5-battery: mit_batch3 zero-shot

모델 `amazon/chronos-t5-base`, 관측 64 → 예측 16, stride 1.

각 배터리 측정 기록의 관측 cycle 범위를 3등분한다. 예측 시작 cycle로 초·중·후반을 배정하고 해당 window의 전체 horizon을 평가한다. 경계를 넘는 horizon도 시작 구간에 포함한다. 실제 전체 수명 3등분을 뜻하지 않는다.

Mean window R²: 한 예측창의 미래 target 평균을 기준으로 R² = 1 − SSE/SST를 계산한 뒤 각 배터리·구간 또는 전체에서 유효한 창별 R²를 단순평균한다. 이후 배터리 점수를 동일 가중 평균한다. 음수는 유지하고 상수 target 창은 NaN으로 제외한다. r2_n_windows는 유효 창 수, r2_undefined_windows는 제외 창 수다. 전체 점수는 전체 창의 평균이며 초·중·후반 평균을 단순평균하지 않는다. 실제값이 거의 일정하면 R²가 크게 음수여도 계산 오류가 아니므로 RMSE·곡선을 함께 본다.

| 모델 | 구간 | RMSE ↓ | R² ↑ | WQL ↓ | PICP@80 | 80% 구간 폭 |
|---|---|---:|---:|---:|---:|---:|
| LastValue | all | 0.002843 | -3.269120 | — | — | — |
| Linear | all | 0.001192 | -0.414277 | — | — | — |
| t5-battery | all | 0.001755 | -47.154062 | 0.001343 | 34.5% | 0.003552 |
| LastValue | early | 0.000407 | -3.183928 | — | — | — |
| Linear | early | 0.000224 | -1.825341 | — | — | — |
| t5-battery | early | 0.001574 | -133.143609 | 0.001491 | 24.2% | 0.002646 |
| LastValue | middle | 0.000627 | -3.336667 | — | — | — |
| Linear | middle | 0.000241 | 0.105543 | — | — | — |
| t5-battery | middle | 0.000890 | -17.981378 | 0.000892 | 28.2% | 0.004252 |
| LastValue | late | 0.004783 | -3.271332 | — | — | — |
| Linear | late | 0.002003 | 0.248176 | — | — | — |
| t5-battery | late | 0.002445 | -4.147876 | 0.001707 | 49.9% | 0.003593 |

WQL은 q0.1~q0.9의 9개 twice-pinball loss를 평균하고 구간 target 절댓값 합으로 정규화한다. PICP@80은 q0.1~q0.9 포함률이며 명목값은 80%다. RMSE와 구간 폭은 SOH 비율 단위다.

최근 실제 관측으로 매번 예측하는 rolling zero-shot이다. 학습·clipping·평활화 없이 같은 입력과 window를 사용한다. C2는 독립 단변량 입력과 cross_learning=False를 사용한다. T5는 100 samples, seed 42; C2는 직접 분위수 예측으로 sampling 설정을 사용하지 않는다.

대표 곡선은 각 구간 중앙에 가장 가까운 유효 예측창이다. 그래프의 R²는 표시된 window 한 개의 점수이며 구간 전체 점수와 다르다. 겹치는 window는 독립 표본이 아니다. 일부 배터리만 평가한 pilot이며 전체 데이터셋 성능으로 해석하지 않는다.

[설정](t5-battery_run.json) · [배터리별 점수](t5-battery_battery_metrics.csv)

![대표 SOH 곡선](t5-battery_forecast_windows_01.png)

# R²를 포함한 성능평가

`python scripts/add_r2_metrics.py`로 재생성한다. 재학습은 하지 않는다.

Mean window R2: compute standard 1-SSE/SST separately for each forecast window, centering on that window's actual future target mean; arithmetic mean of valid window scores within each battery/seed/phase, then equal seed and battery weights. Negative values are retained; constant windows or horizons shorter than 2 return NaN. No pooled/global or within-variance-weighted R2. R2 is a reporting metric only; checkpoint selection remains unchanged.

원본 `results/runs`의 예측·학습 기록·선택·서명된 CSV는 보존하고, 여기에는 R²를 추가한 평가표를 저장한다. RMSE/MAE도 원시 예측에서 검산한다. 02·03 노트북과 현재 보고서는 이 평가표를 사용한다. 원시 validation 예측이 없는 과거 이력의 R²는 NaN이다. aggregate RMSE로 창별 R² 평균을 복원하지 않는다. 이미 평균한 RMSE에서 macro R²를 역산하지 않는다.

창별 R²는 1이면 완전 예측, 0이면 해당 창의 실제 미래 평균을 항상 예측한 기준과 같다. 이 평균은 평가 구간의 사후 평균이므로 Last-value baseline과 다르다. 음수도 유효하며 자르지 않는다. 정답이 상수이거나 값이 2개 미만이면 분모가 정의되지 않아 NaN으로 표시한다. 짧고 변화가 작은 window의 R²는 매우 낮아질 수 있다. R² 차이는 퍼센트 개선율로 계산하지 않는다.

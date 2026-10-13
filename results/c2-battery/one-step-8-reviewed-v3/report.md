# C2 현재 SOH 예측: 64→1 · 32→1 · 16→1 · 8→1

과거 SOH 8개 기록 위치로 현재 SOH 한 점을 예측한다. 실제 과거 SOH로 입력을 갱신하는 rolling one-step 평가다.

- 기존 reviewed_native_v3 정제·NaN 마스크, 배터리 분할, few-shot 학습/검증 정답과 97,868개 테스트 시점을 유지한다.
- 비교 조건: LastValue, C2 zero-shot, C2 head-only. Linear 제외.
- Head-only: backbone 고정, output_patch_embedding만 학습. Fold당 학습 8개·검증 4개 정답, LR 1e-6/1e-5, 각각 100 steps, seed 42. 검증으로 head 선택 후 테스트한다.
- R²: 같은 배터리 기록의 여러 테스트 시점 한 점 예측을 모아서 계산한 뒤 기록별 동일 가중 평균. 음수 유지, 정의 불가능한 값의 수는 CSV에 표시한다.
- RMSE/MAE는 SOH 비율 단위, 기록별 동일 가중 평균이다. LastValue MSE 감소율도 각 기록의 비율을 먼저 구한 뒤 평균한다.
- 16→1과 8→1은 동일한 데스크톱 CUDA·GPU·패키지 환경에서 계산했다. 기존 64/32 결과는 XPU다. 하드웨어·패키지 차이에 의한 수치 및 head 선택 차이는 입력 길이 효과와 완전히 분리되지 않는다.
- CALCE의 파일 사이 미확인 물리 cycle, MIT 계측 불일치, 결측 입력 관련 기존 제한을 유지한다.

## 8→1 결과

| 데이터셋 | 조건 | Test R² | RMSE | MAE | LastValue 대비 MSE 감소율 |
|---|---|---:|---:|---:|---:|
| nasa_aging | LastValue | 0.941505 | 0.008843 | 0.005188 | +0.00% |
| nasa_aging | zero-shot | 0.934735 | 0.009176 | 0.005151 | -6.88% |
| nasa_aging | head-only | 0.935755 | 0.009131 | 0.005264 | -5.77% |
| calce_cs2_1c | LastValue | 0.999131 | 0.005686 | 0.003443 | +0.00% |
| calce_cs2_1c | zero-shot | 0.999121 | 0.005722 | 0.003453 | -1.11% |
| calce_cs2_1c | head-only | 0.998915 | 0.006354 | 0.004116 | -24.74% |
| mit_batch1 | LastValue | 0.996393 | 0.000615 | 0.000382 | +0.00% |
| mit_batch1 | zero-shot | 0.996288 | 0.000545 | 0.000313 | +24.51% |
| mit_batch1 | head-only | 0.996312 | 0.000531 | 0.000303 | +28.72% |
| mit_batch2 | LastValue | 0.999555 | 0.000956 | 0.000640 | +0.00% |
| mit_batch2 | zero-shot | 0.999682 | 0.000706 | 0.000441 | +46.74% |
| mit_batch2 | head-only | 0.999705 | 0.000634 | 0.000381 | +56.65% |
| mit_batch3 | LastValue | 0.999910 | 0.000368 | 0.000224 | +0.00% |
| mit_batch3 | zero-shot | 0.999964 | 0.000229 | 0.000132 | +58.71% |
| mit_batch3 | head-only | 0.999974 | 0.000182 | 0.000090 | +70.74% |
| xjtu_2c | LastValue | 0.999582 | 0.001049 | 0.000745 | +0.00% |
| xjtu_2c | zero-shot | 0.999643 | 0.000965 | 0.000636 | +15.95% |
| xjtu_2c | head-only | 0.999651 | 0.000953 | 0.000623 | +18.03% |
| xjtu_3c | LastValue | 0.998497 | 0.001587 | 0.001254 | +0.00% |
| xjtu_3c | zero-shot | 0.999665 | 0.000793 | 0.000632 | +71.67% |
| xjtu_3c | head-only | 0.999649 | 0.000806 | 0.000656 | +71.36% |

## 입력 길이별 head-only 비교

| 데이터셋 | 64 R² | 32 R² | 16 R² | 8 R² | 64 LV 감소율 | 32 LV 감소율 | 16 LV 감소율 | 8 LV 감소율 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nasa_aging | 0.939368 | 0.934822 | 0.931121 | 0.935755 | -2.89% | -11.67% | -9.35% | -5.77% |
| calce_cs2_1c | 0.999092 | 0.998883 | 0.998556 | 0.998915 | -3.97% | -25.22% | -61.59% | -24.74% |
| mit_batch1 | 0.997124 | 0.996937 | 0.996678 | 0.996312 | +46.85% | +35.66% | +8.39% | +28.72% |
| mit_batch2 | 0.999763 | 0.999813 | 0.999752 | 0.999705 | +49.70% | +72.51% | +60.68% | +56.65% |
| mit_batch3 | 0.999968 | 0.999980 | 0.999974 | 0.999974 | +67.28% | +77.57% | +70.13% | +70.74% |
| xjtu_2c | 0.999554 | 0.999543 | 0.999571 | 0.999651 | -0.02% | -4.15% | -2.48% | +18.03% |
| xjtu_3c | 0.999745 | 0.999749 | 0.999606 | 0.999649 | +75.33% | +76.67% | +67.50% | +71.36% |

입력 길이 간 차이는 동일한 LastValue 기준 MSE 감소율의 차이(percentage points)로 표시한다.

![입력 길이 비교](comparison.png)

분위수 역전은 정렬·보정하지 않는다. 점 예측 지표와 별개로 summary.csv의 quantile_crossing_fraction 및 invalid_interval_fraction을 확인한다.

[전체 지표](context_comparison_scores.csv) · [LastValue 비교](lastvalue_comparison.csv) · [길이별 차이](context8_lastvalue_change.csv) · [검증](verification.json)

정제 기준을 통과한 SOH만 표시한다. 제외된 점은 점·선 모두 숨기고 원래 cycle 위치에 빈 구간을 남긴다.

![nasa_aging](nasa_aging/representative.png)

![calce_cs2_1c](calce_cs2_1c/representative.png)

![mit_batch1](mit_batch1/representative.png)

![mit_batch2](mit_batch2/representative.png)

![mit_batch3](mit_batch3/representative.png)

![xjtu_2c](xjtu_2c/representative.png)

![xjtu_3c](xjtu_3c/representative.png)

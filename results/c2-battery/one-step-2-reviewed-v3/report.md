# C2 현재 SOH 예측: 64→1 · 32→1 · 16→1 · 8→1 · 4→1 · 2→1

과거 SOH 2개 기록 위치로 현재 SOH 한 점을 예측한다. 실제 과거 SOH로 입력을 갱신하는 rolling one-step 평가다.

- 기존 reviewed_native_v3 정제·NaN 마스크, 배터리 분할, few-shot 학습/검증 정답과 97,868개 중 입력 두 위치가 모두 결측인 CALCE 3점을 제외한 97,865개 테스트 시점을 사용한다. 학습/검증 정답은 모두 동일하며, 길이별 비교도 이 공통 시점에서 다시 계산한다. 제외 목록은 excluded_targets.csv에 저장한다.
- 비교 조건: LastValue, C2 zero-shot, C2 head-only. Linear 제외.
- Head-only: backbone 고정, output_patch_embedding만 학습. Fold당 학습 8개·검증 4개 정답, LR 1e-6/1e-5, 각각 100 steps, seed 42. 검증으로 head 선택 후 테스트한다.
- R²: 같은 배터리 기록의 여러 테스트 시점 한 점 예측을 모아서 계산한 뒤 기록별 동일 가중 평균. 음수 유지, 정의 불가능한 값의 수는 CSV에 표시한다.
- RMSE/MAE는 SOH 비율 단위, 기록별 동일 가중 평균이다. LastValue MSE 감소율도 각 기록의 비율을 먼저 구한 뒤 평균한다.
- 16→1·8→1·4→1·2→1은 동일한 데스크톱 CUDA·GPU·패키지 환경에서 계산했다. 기존 64/32 결과는 XPU다. 하드웨어·패키지 차이에 의한 수치 및 head 선택 차이는 입력 길이 효과와 완전히 분리되지 않는다.
- CALCE의 파일 사이 미확인 물리 cycle, MIT 계측 불일치, 결측 입력 관련 기존 제한을 유지한다.

## 2→1 결과

| 데이터셋 | 조건 | Test R² | RMSE | MAE | LastValue 대비 MSE 감소율 |
|---|---|---:|---:|---:|---:|
| nasa_aging | LastValue | 0.941505 | 0.008843 | 0.005188 | +0.00% |
| nasa_aging | zero-shot | 0.945260 | 0.008379 | 0.004552 | +10.96% |
| nasa_aging | head-only | 0.943203 | 0.008443 | 0.004779 | +9.90% |
| calce_cs2_1c | LastValue | 0.999129 | 0.005686 | 0.003441 | +0.00% |
| calce_cs2_1c | zero-shot | 0.999144 | 0.005634 | 0.003434 | +1.75% |
| calce_cs2_1c | head-only | 0.999090 | 0.005826 | 0.003607 | -4.77% |
| mit_batch1 | LastValue | 0.996393 | 0.000615 | 0.000382 | +0.00% |
| mit_batch1 | zero-shot | 0.995902 | 0.000586 | 0.000357 | +14.05% |
| mit_batch1 | head-only | 0.994323 | 0.000591 | 0.000331 | +16.76% |
| mit_batch2 | LastValue | 0.999555 | 0.000956 | 0.000640 | +0.00% |
| mit_batch2 | zero-shot | 0.999569 | 0.000862 | 0.000571 | +21.54% |
| mit_batch2 | head-only | 0.999290 | 0.000831 | 0.000484 | +26.13% |
| mit_batch3 | LastValue | 0.999910 | 0.000368 | 0.000224 | +0.00% |
| mit_batch3 | zero-shot | 0.999932 | 0.000319 | 0.000191 | +24.28% |
| mit_batch3 | head-only | 0.999953 | 0.000260 | 0.000146 | +46.76% |
| xjtu_2c | LastValue | 0.999582 | 0.001049 | 0.000745 | +0.00% |
| xjtu_2c | zero-shot | 0.999659 | 0.000948 | 0.000663 | +18.14% |
| xjtu_2c | head-only | 0.999684 | 0.000905 | 0.000569 | +26.26% |
| xjtu_3c | LastValue | 0.998497 | 0.001587 | 0.001254 | +0.00% |
| xjtu_3c | zero-shot | 0.998993 | 0.001307 | 0.001040 | +31.62% |
| xjtu_3c | head-only | 0.999458 | 0.000974 | 0.000773 | +60.91% |

## 입력 길이별 head-only 비교

| 데이터셋 | 64 R² | 32 R² | 16 R² | 8 R² | 4 R² | 2 R² | 64 LV 감소율 | 32 LV 감소율 | 16 LV 감소율 | 8 LV 감소율 | 4 LV 감소율 | 2 LV 감소율 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nasa_aging | 0.939368 | 0.934822 | 0.931121 | 0.935755 | 0.930770 | 0.943203 | -2.89% | -11.67% | -9.35% | -5.77% | -7.16% | +9.90% |
| calce_cs2_1c | 0.999089 | 0.998879 | 0.998550 | 0.998913 | 0.998965 | 0.999090 | -4.06% | -25.26% | -61.70% | -24.79% | -19.59% | -4.77% |
| mit_batch1 | 0.997124 | 0.996937 | 0.996678 | 0.996312 | 0.995639 | 0.994323 | +46.85% | +35.66% | +8.39% | +28.72% | +28.66% | +16.76% |
| mit_batch2 | 0.999763 | 0.999813 | 0.999752 | 0.999705 | 0.999628 | 0.999290 | +49.70% | +72.51% | +60.68% | +56.65% | +54.00% | +26.13% |
| mit_batch3 | 0.999968 | 0.999980 | 0.999974 | 0.999974 | 0.999972 | 0.999953 | +67.28% | +77.57% | +70.13% | +70.74% | +67.49% | +46.76% |
| xjtu_2c | 0.999554 | 0.999543 | 0.999571 | 0.999651 | 0.999678 | 0.999684 | -0.02% | -4.15% | -2.48% | +18.03% | +26.52% | +26.26% |
| xjtu_3c | 0.999745 | 0.999749 | 0.999606 | 0.999649 | 0.999704 | 0.999458 | +75.33% | +76.67% | +67.50% | +71.36% | +75.40% | +60.91% |

입력 길이 간 차이는 동일한 LastValue 기준 MSE 감소율의 차이(percentage points)로 표시한다.

![입력 길이 비교](comparison.png)

분위수 역전은 정렬·보정하지 않는다. 점 예측 지표와 별개로 summary.csv의 quantile_crossing_fraction 및 invalid_interval_fraction을 확인한다.

[전체 지표](context_comparison_scores.csv) · [LastValue 비교](lastvalue_comparison.csv) · [길이별 차이](context2_lastvalue_change.csv) · [검증](verification.json)

정제 기준을 통과한 SOH만 표시한다. 제외된 점은 점·선 모두 숨기고 원래 cycle 위치에 빈 구간을 남긴다.

![nasa_aging](nasa_aging/representative.png)

![calce_cs2_1c](calce_cs2_1c/representative.png)

![mit_batch1](mit_batch1/representative.png)

![mit_batch2](mit_batch2/representative.png)

![mit_batch3](mit_batch3/representative.png)

![xjtu_2c](xjtu_2c/representative.png)

![xjtu_3c](xjtu_3c/representative.png)

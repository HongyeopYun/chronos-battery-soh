# 공통 데이터로 zero-shot 실행

`src.battery_zeroshot.battery_zeroshot`는 공통 SOH 데이터 묶음을 받아 Chronos 추론·기준 모델 비교·성능평가·대표 곡선을 저장한다. XJTU·NASA·MIT·CALCE는 같은 함수에 데이터 경로만 바꿔 전달한다. 기존 NASA 전용 02·03 및 저장된 실험은 유지한다.

```python
from pathlib import Path
from src.battery_zeroshot import battery_zeroshot

run = battery_zeroshot(Path("data/processed/clean/xjtu_2c"))
```

MIT는 경로를 `data/processed/clean/mit_batch1` 등으로 바꾼다. [공통 SOH 포맷](common_soh_format.md)의 `soh.csv`, `batteries.csv`, `schema.json`, `manifest.json`이 필요하다. 개별 CSV만 직접 전달하지 않는다. 로더가 완료 상태·파일 해시·기본키·Q/Q_ref 일관성을 먼저 검사한다.

## 설정과 실행

기본 설정은 `configs/battery_zeroshot.json`이다. NASA와 같은 모델 revision, context 64, horizon 16, 100 samples, seed 42/43/44, batch size 16을 사용한다. 학습·validation·LoRA 옵션은 없다. 데이터에 저장된 정규화 기준을 유지하고 clipping·보간·평활화를 하지 않는다.

```powershell
# 전체 eligible windows: XJTU 2C 2,579개, 3C 2,542개
.\.venv\Scripts\python.exe scripts\run_battery_zeroshot.py data\processed\clean\xjtu_2c
.\.venv\Scripts\python.exe scripts\run_battery_zeroshot.py data\processed\clean\xjtu_3c

# 모델을 불러오기 전에 window 수와 제외 기록만 확인
.\.venv\Scripts\python.exe scripts\run_battery_zeroshot.py data\processed\clean\mit_batch1 --prepare-only

# 배터리별 초반·중반·후반 window 3개로 실행만 검증
.\.venv\Scripts\python.exe scripts\run_battery_zeroshot.py data\processed\clean\xjtu_2c --max-windows 3

# 배터리 하나만 선택
.\.venv\Scripts\python.exe scripts\run_battery_zeroshot.py data\processed\clean\xjtu_2c --battery-ids batch1_2C_battery-1
```

[노트북](../notebooks/battery_zeroshot.ipynb)에서는 `DATA_BUNDLE`을 바꾸면 된다. `SMOKE_CHECK=True`는 window 3개만 사용하고, `False`는 기본 설정의 모든 window를 사용한다. 생성 원본은 `scripts/build_notebooks.py`에 있다.

설정을 Python에서 바꾸려면 `dataclasses.replace`를 사용한다.

```python
from dataclasses import replace
from src.battery_zeroshot import load_zeroshot_config

config = replace(load_zeroshot_config(), max_windows_per_battery=3)
run = battery_zeroshot(Path("data/processed/clean/xjtu_3c"), config)
```

`max_windows_per_battery`는 유효 window의 인덱스를 균등 간격으로 선택하며 오차나 target 값으로 고르지 않는다. 실행 범위는 `smoke`로 표시한다. 제한 없이 stride 1이면 `full`, stride를 늘리면 `strided`다. 일부 window 결과를 전체 평가 성능으로 해석하지 않는다.

CALCE는 파일 segment당 최대 50 cycle이라 기본 64→16 설정에서 실행하지 못한다. `--prepare-only`로 이를 확인할 수 있다. 짧은 context/horizon을 쓰려면 설정을 명시적으로 바꾸고 다른 길이의 실험으로 구분한다. 파일을 임의로 이어 붙이지 않는다.

## 평가와 산출물

실행별 결과는 `results/battery_zeroshot/<run_id>/`에 저장한다. 데이터 경로·모델·장치·설정·코드·패키지 버전과 입력 묶음 해시가 같은 완료 실행은 산출물 검사와 지표 재계산 후 재사용한다. `--force`는 새 실행을 만든다. 실패·중단 실행은 재사용하지 않는다.

| 파일 | 내용 |
|---|---|
| `manifest.json` | 입력 데이터 경로·해시, 모델 revision, 설정, 환경, full/smoke 범위, 결과 해시 |
| `inference_manifest.json` | 지표를 갱신한 실행에서만 생성하는 원본 추론 manifest; 당시 코드·환경·예측 해시 보존 |
| `coverage.csv` | 전체 관측 수, 유효·평가 window 수, window가 없어 제외한 기록 |
| `predictions/*.npz` | 실제 context/target, 원본 cycle 번호, 원시 samples, median/mean, q10/q90 |
| `metrics.csv` | 배터리·seed·모델·집계별 RMSE·R²·WQL·PICP@80 및 보조 지표 |
| `battery_summary.csv`, `summary.csv` | seed 평균과 배터리 기록에 같은 가중치를 준 macro 점수 |
| `macro_by_seed.csv` | 추론 seed별 macro 점수 |
| `horizon_metrics.csv`, `window_metrics.csv` | horizon·예측 시작 cycle별 RMSE·R²·WQL·PICP@80 및 구간 폭·MAE |
| `paired_comparison.csv` | 같은 평가 구간에서 Chronos와 LastValue/Linear 비교 |
| `report.md`, `forecast_windows_*.png` | 요약 및 배터리별 초반·중반·후반 예측 곡선 |

원시 예측 NPZ는 로컬에 보관하고 Git에서는 제외한다. 보고서·지표·곡선·해시는 커밋한다. 다른 PC에서 예측 파일이 없으면 재검산·완료 실행 재사용은 불가능하며 원본 데이터를 준비해 다시 실행해야 한다.

주 결과표는 **RMSE·R²·WQL·PICP@80** 네 지표를 사용한다. RMSE·R²의 주 점예측은 Chronos samples의 중앙값이다. 80% 구간 폭(`width80`)은 보조로 표시하고 sample mean 점예측·MAE·CRPS·비정상 sample 비율은 CSV에 유지한다.

LastValue는 마지막 실제 관측을 유지하고 Linear는 context의 최소제곱 직선을 연장한다. 두 기준 모델은 Chronos와 동일한 window와 target을 사용한다. 기준 점예측에는 확률 구간을 만들지 않으므로 WQL·PICP@80·구간 폭은 NaN으로 저장하고 결과표에는 `—`로 표시한다.

WQL은 samples에서 선형 보간으로 구한 **0.1~0.9의 9개 분위수**를 동일 가중치로 사용한다. 분위수별 오차 `e = y - forecast_q`에 대해 `WQL = 2 × mean_q(sum(max(q × e, (q - 1) × e))) / sum(abs(y))`다. 각 배터리·seed의 모든 window/horizon target을 합쳐 정규화한 뒤 seed와 배터리 기록을 평균한다. 개별 window WQL의 단순 평균과는 다르다. Horizon/window 표는 해당 범위 안에서 같은 식을 계산한다. 분모가 0이면 NaN이며 낮을수록 좋다. [AWS의 분위수 손실 정의](https://docs.aws.amazon.com/forecast/latest/dg/metrics.html)를 참고하며, 이 구현의 평균 분위수와 분모 0 처리 규칙은 위와 같이 고정한다.

PICP@80은 `mean(q10 <= y <= q90)`이다. 경계값을 포함하며 CSV의 `picp80`은 0~1 비율, 보고서는 백분율이다. **명목값 80%에 가까운지** 확인하고 구간 폭을 함께 해석한다. 단순히 높을수록 좋은 지표는 아니다. 기존 NASA 실험의 `coverage80`과 같은 포함률 정의이며 새 공통 표에서는 `picp80`으로 이름을 통일한다. 구간 폭은 `mean(q90 - q10)`으로 SOH 비율 단위다.

R²는 배터리·seed별 전체 window/horizon을 펼쳐 `1 - SSE/SST`로 계산한다. 상수 target 또는 관측 수 2개 미만이면 NaN, 음수도 유효하다. Horizon/window 표는 각 평가 범위의 target 평균을 사용한다. Macro는 seed 점수를 평균하고 배터리 **기록**마다 같은 가중치를 준다. MIT의 연속 측정 기록을 독립 물리 배터리로 세거나 여러 batch의 macro 점수를 무조건 평균하지 않는다.

매 예측 시점에 최근 실제 관측을 새로 제공하는 rolling 평가다. 첫 64개만 알고 수명 전체를 재귀 예측하는 방식이 아니다. 초기 시험·용량 이상치·부분 방전·수집 문제는 예측 context와 target에서 제외하고, 누락 cycle이나 segment 경계를 건너 window를 만들지 않는다. 겹치는 window와 여러 추론 seed를 독립 데이터 표본으로 해석하지 않는다.

```powershell
# 모델 추론 없이 저장 예측·지표·입력·해시 재검산
.\.venv\Scripts\python.exe scripts\run_battery_zeroshot.py --audit results\battery_zeroshot\실행폴더

# 지표 구현을 바꾼 뒤 기존 samples로 결과표만 갱신
.\.venv\Scripts\python.exe scripts\run_battery_zeroshot.py --refresh-metrics results\battery_zeroshot\실행폴더
```

`--refresh-metrics`는 원본 입력·설정·추론 코드·저장 예측을 먼저 검사한다. 기존 manifest를 `inference_manifest.json`으로 보존하고, 원시 NPZ와 곡선 PNG를 바꾸지 않고 지표·보고서만 갱신한다. 현재 manifest에는 원본 추론 기록의 해시와 새 평가 규칙을 기록한다. 모델 추론은 다시 실행하지 않으며 로컬 원시 예측 파일이 필요하다.

## 2026-10-06 실행 검증

XJTU 23개 배터리에서 대표 window 세 개씩, 총 69개 window를 100 samples·seed 42/43/44로 실제 추론했다. 저장 samples에서 네 주 지표를 재계산하고 파일 해시·target·cycle 번호를 확인했다. 지표 갱신 시 원시 예측과 곡선은 유지했다. 이 기록은 함수의 실행 검증이며 전체 성능 평가가 아니다.

| 그룹 | 배터리 | 검증 window | 전체 eligible windows | 결과 |
|---|---:|---:|---:|---|
| XJTU 2C | 8 | 24 | 2,579 | [보고서·곡선](../results/battery_zeroshot/20261006T030616Z_batteryzeroshot_bf1d52/report.md) |
| XJTU 3C | 15 | 45 | 2,542 | [보고서·곡선](../results/battery_zeroshot/20261006T030722Z_batteryzeroshot_fecc95/report.md) |

NASA·MIT·CALCE도 같은 준비 함수로 읽었다. NASA는 기존 window 정의와 같고 MIT Batch 3의 수집 문제 배터리 6개는 기본 규칙대로 제외한다. CALCE 4개는 기본 길이에서 window가 없음을 표시한다. 이 단계에서 MIT·CALCE의 실제 모델 추론이나 새로운 전체 NASA 평가는 실행하지 않았다.

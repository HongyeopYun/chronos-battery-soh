# 공통 데이터로 zero-shot 실행

이 문서는 전체 T5 평가와 최소 결과 형식에 관한 것이다. 모든 평가 R²를 Mean window R²로 통일했다. 모델별 설정·노트북·결과 이름은 `t5-battery`, `c2-battery`다. 초·중·후반 비교는 [t5-c2-battery](t5-c2-battery.md)에 설명한다.

공통 SOH 묶음의 경로만 바꿔 Chronos zero-shot을 평가한다. 기본 결과는 점수표와 예측 그래프만 남기는 최소 패키지다.

```python
from pathlib import Path
from src.battery_zeroshot import battery_zeroshot

run = battery_zeroshot(Path("data/processed/clean/xjtu_2c"))
```

입력은 [공통 SOH 포맷](common_soh_format.md)의 `soh.csv`, `batteries.csv`, `schema.json`, `manifest.json`이다. 데이터와 cycle 번호를 그대로 사용하며 학습·clipping·보간·평활화를 하지 않는다. 초기 시험·용량 이상치·수집 문제와 누락 cycle은 계속 제외한다. CALCE는 `calce_cross_file_windows=true`일 때 같은 배터리의 시간순 파일 경계를 연결한다. 원본 묶음의 segment는 수정하지 않는다.

## 실행

`configs/t5-battery.json`에서 모델 revision, context 64, horizon 16, 100 samples, seed 42, stride 1을 고정한다. 추론 batch size는 32다. 기존 3-seed 검증과 이번 1-seed 평가를 구분한다.

```powershell
# 모든 준비된 데이터셋; CALCE도 64→16, stride 1 (새 출력 경로 사용)
.\.venv\Scripts\python.exe scripts\run_battery_zeroshot.py --all --output-root results/t5-battery/new64-16

# 묶음 하나
.\.venv\Scripts\python.exe scripts\run_battery_zeroshot.py data\processed\clean\xjtu_2c

# 모델 로드 전 window 수 확인
.\.venv\Scripts\python.exe scripts\run_battery_zeroshot.py data\processed\clean\mit_batch1 --prepare-only

# 각 배터리 대표 window 세 개로 smoke check
.\.venv\Scripts\python.exe scripts\run_battery_zeroshot.py data\processed\clean\xjtu_2c --max-windows 3
```

기존 CALCE 실행은 파일 경계를 끊어 **32→8**로 평가했다. 새 설정은 파일 경계 자체를 단절로 보지 않고, 같은 배터리의 시간순 관측으로 **64→16**을 평가한다. 누락·불완전 관측과 protocol 변경은 여전히 창을 끊는다. 달력 시간의 휴지는 별도 모델 입력하지 않으며 cycle은 기존 관측 기록 순번이다. 이전 결과는 보존하고 [CALCE 재실험](t5-c2-battery.md#calce-6416-재실험)으로 분리한다.

`--stride 16`은 모든 배터리를 평가하되 예측 시작점 간격을 늘린다. `--all --stride 16`에서도 CALCE는 stride 1이다. `--max-windows 3`은 유효 window 위치로 초·중·후반을 선택하며 target이나 오차로 고르지 않는다. Smoke 결과는 별도 `<group>_smoke` 폴더에 저장한다.

## 최소 결과

`results/t5-battery/<group>/`에 다음만 남긴다.

| 파일 | 내용 |
|---|---|
| `summary.csv` | Chronos 중앙값·LastValue·Linear의 macro RMSE, Mean window R², WQL, PICP@80, 구간 폭 |
| `battery_metrics.csv` | 배터리 기록별 모델 점수, 평가 window 수, 제외 상태 |
| `run.json` | 입력 묶음, 실행 조건, full/strided/smoke 범위, 완료 상태 |
| `report.md` | 조건·점수·그래프를 묶은 보고서 |
| `forecast_windows_*.png` | 배터리별 첫·중간·마지막 window의 context·실제 SOH·중앙값·80% 구간 |

전체 실행의 `summary.csv`와 `README.md`는 `results/t5-battery/` 바로 아래에 만든다. 그룹별 점수를 하나의 전체 macro로 합치지 않는다. MIT의 일부 연속 측정 기록은 동일 물리 배터리와 연관되므로 기록 수를 독립 물리 배터리 수로 해석하지 않는다.

원시 samples, seed/window/horizon별 중복 CSV, mean 점예측, CRPS, 코드·환경 해시는 기본 결과에 남기지 않는다. 실행 중에는 완료한 배터리 점수와 대표 forecast만 임시 저장해 재개할 수 있게 하고 완료 시 임시 forecast 파일을 삭제한다. 입력 묶음과 설정이 같은 완료 패키지는 보관된 표·그림의 일관성을 확인한 뒤 재사용한다. `--force`는 같은 그룹 결과를 새로 계산한다.

`--audit <그룹폴더>`는 최소 패키지의 표·그림 일관성을 확인한다. Mean window R²는 집계 RMSE만으로 복원할 수 없다. 구 정의의 결과를 갱신하려면 `python scripts/unify_window_r2.py`로 저장된 입력·모델·seed·batch 설정을 사용해 다시 추론한다. 완료한 패키지는 정의를 확인하고 재사용한다. 원시 예측이 필요한 실험에는 `--detailed` 또는 `minimal=False`를 사용할 수 있으며 상세 형식도 같은 Mean window R²를 사용한다.

완료된 7개 그룹과 전체 요약표를 함께 갱신하려면 `python scripts/run_battery_zeroshot.py --refresh-metrics results/t5-battery`을 사용한다. 이 명령은 모델 추론을 실행하지 않는다.

## 점수 해석

- **RMSE:** samples 중앙값의 오차. SOH 비율 단위이며 0.005는 약 0.5%p다.
- **WQL:** 선형 분위수 0.1~0.9의 9개 twice-pinball loss를 동일 가중치로 평균하고 target 절댓값 합으로 정규화한다. 각 배터리·seed의 모든 window/horizon을 합쳐 계산한 뒤 seed·배터리 기록을 평균한다. 분모 0은 NaN이다. 낮을수록 좋다.
- **PICP@80:** `mean(q10 <= y <= q90)`. CSV는 0~1, 보고서는 백분율이다. 명목값 80%에 얼마나 가까운지 구간 폭과 함께 본다.
- **80% 구간 폭:** `mean(q90 - q10)`. SOH 비율 단위다.
- **Mean window R² (`r2`):** 한 예측창의 실제 미래 평균으로 `1 − SSE_w/SST_w`를 계산하고 유효 창별 점수를 단순평균한다. Seed·배터리 기록에는 동일 가중치를 준다. 전체·초·중·후반 모두 같은 정의다. 그래프에는 해당 한 창의 R²를 표시한다. 음수는 유지하며 상수 target 창은 NaN으로 제외한다. 작은 분모에 epsilon을 더하지 않는다. 합산 SSE/SST나 내부 변동량 가중 R²는 사용하지 않는다.

LastValue는 마지막 실제 관측을 유지하고 Linear는 context의 최소제곱 직선을 연장한다. 같은 target/window를 사용하며 확률 구간이 없어 WQL·PICP·구간 폭은 비운다. 매 예측 시점에 최신 실제 context를 제공하는 rolling 평가이며, 수명 전체를 재귀 예측하는 실험과 구분한다. 겹치는 window는 독립 표본이 아니다.

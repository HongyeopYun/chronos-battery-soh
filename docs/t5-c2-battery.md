# T5 / Chronos-2 battery zero-shot 비교

모델별 이름을 `t5-battery`, `c2-battery`로 구분한다. 두 모델 모두 최근 실제 SOH 64개로 미래 16개를 예측하며 stride 1, 가중치 학습 없이 실행한다. Chronos-2는 각 예측창을 독립 단변량 작업으로 전달하고 `cross_learning=False`를 사용한다. 서로 다른 시점의 창끼리 관측값을 공유하지 않는다.

## 첫 비교 범위

- MIT batch3: `b3c0`, `b3c1`, `b3c3`, `b3c4` (3,915 windows).
- XJTU 3C: `batch2_3C_battery-1`, `batch2_3C_battery-2` (431 windows).

각 모델이 동일한 6개 배터리, 4,346개 유효 예측창을 평가한다. 기존 MIT 첫 그림의 배터리와 XJTU 3C 앞 두 기록을 사용한 진단용 pilot이다. 전체 데이터셋 성능이나 통계적 유의성을 주장하지 않는다.

## 실행 및 확인

```powershell
# 추론 없이 배터리·window 수 확인
.\.venv\Scripts\python.exe scripts\run_battery_comparison.py --prepare-only

# C2를 먼저 실행하고 메모리를 해제한 뒤 T5 실행
.\.venv\Scripts\python.exe scripts\run_battery_comparison.py --model both

# C2를 준비된 전체 7개 그룹에 실행; 새 설정은 CALCE도 64→16
.\.venv\Scripts\python.exe scripts\run_battery_comparison.py --model c2 --all --output-root results/c2-battery/new64-16
```

`configs/t5-battery.json`, `configs/c2-battery.json`이 모델별 설정이다. T5는 기존 100 samples·seed 42를 사용한다. C2는 float32로 분위수를 직접 예측하므로 샘플 수·temperature·top-k·top-p를 사용하지 않는다. 공개 C2 모델은 `models/huggingface`에 캐시되며 Git에서 제외한다. 두 설정 모두 모델 revision을 고정한다. C2 모델 파일을 먼저 다운로드해야 한다.

```python
from huggingface_hub import snapshot_download
snapshot_download("amazon/chronos-2",
    revision="29ec3766d36d6f73f0696f85560a422f50e8498c",
    cache_dir="models/huggingface", allow_patterns=["config.json", "model.safetensors"])
```

이 명령은 전경 실행이다. 백그라운드 실행을 시작할 때는 사용할 컴퓨터·GPU·범위·예상 시간을 먼저 알린다. 현재 파일을 확인하는 [t5-battery 노트북](../notebooks/t5-battery.ipynb), [c2-battery 노트북](../notebooks/c2-battery.ipynb)은 저장된 점수와 그림만 읽으며 추론을 자동 실행하지 않는다.

Pilot 백그라운드 작업의 로그는 아래처럼 확인한다. `Ctrl+C`는 로그 보기를 중단하며 추론은 계속 진행된다. `COMPARISON_COMPLETE`가 나오면 두 모델의 비교 작업이 끝난 것이다.

```powershell
Get-Content results\t5-c2-battery-comparison\.runtime\t5-c2-battery.log -Tail 15 -Wait
```

## 결과 이름과 지표

새 비교 결과는 `results/<t5-battery 또는 c2-battery>/pilot/<group>/`에 저장한다. 파일에도 모델 이름을 붙인다: `<model>_summary.csv`, `<model>_battery_metrics.csv`, `<model>_run.json`, `<model>_report.md`, `<model>_forecast_windows_*.png`.

기존 C2 전체 결과는 `results/c2-battery/full/<group>/`에 저장했다(CALCE 32→8). 평가·제외 기록과 초·중·후반 window 수를 `c2-battery_coverage.csv`로 남긴다. 전체 색인은 `results/c2-battery/full/c2-battery_summary.csv`와 `c2-battery_report.md`다. 당시 165개 평가 가능 기록, 100,324개 window이며 기존 수집 문제로 제외한 MIT batch3 기록 6개는 평가하지 않았다. 일부 짧은 기록은 초반 1/3에서 충분한 context를 확보하지 못하므로 초반 점수 없이 표시한다. 해당 구간을 다른 시점의 예측으로 대체하지 않는다. 새 입력 설정으로 기존 결과를 덮어쓰지 않는다.

전체 실행의 진행 상황은 아래 로그로 확인한다. `ALL_C2_COMPLETE`가 나오면 끝난 것이다.

```powershell
Get-Content results\c2-battery\full\.runtime\c2-battery_all.log -Tail 15 -Wait
```

`results/t5-c2-battery-comparison/t5-c2-battery_summary.csv`는 완료된 그룹별 점수를 모은다. 같은 그룹의 T5/C2가 모두 끝나면 입력·window 조건과 baseline 점수 일치를 검사한다. 그룹 사이 점수는 합치지 않는다. 기존 전체 T5 결과도 아래 Mean window R²로 갱신하며 입력과 원래 그래프는 보존한다. 재계산은 `python scripts/unify_window_r2.py`, 진행 상태는 `results/window-r2/status.json`이다.

각 측정 기록의 첫~마지막 관측 cycle 범위를 3등분한다. 예측 시작 cycle에 따라 `early`, `middle`, `late`로 배정하고 그 window의 전체 16개 target을 평가한다. 경계를 넘는 horizon은 시작 구간에 포함한다. 이는 실제 배터리 전체 수명 단계가 아니라 해당 기록의 관측 기간 구분이다. `all`은 배터리 전체 평가 구간의 참고 점수다.

- RMSE: q0.5 점예측의 오차. SOH 비율 단위다.
- Mean window R²: 각 예측창의 미래 실제 평균으로 `1 − SSE_w/SST_w`를 계산한다. 각 배터리·구간 또는 전체의 유효 창별 R²를 단순평균한 뒤 배터리 점수를 동일 가중 평균한다. 전체는 전체 창의 평균이며 초·중·후반 평균의 단순평균이 아니다. 음수는 유지하고 상수 창은 NaN으로 제외한다. `r2_n_windows`·`r2_undefined_windows`는 유효·제외 창 수, `n_r2`는 유효 배터리 수다. 작은 분모에 epsilon을 더하지 않는다.
- WQL: q0.1~q0.9의 9개 twice-pinball loss를 평균하고 해당 배터리·구간 target 절댓값 합으로 정규화한다.
- PICP@80: q0.1~q0.9 사이에 실제값이 들어온 비율. 명목값 80%와 구간 폭을 함께 본다.

그래프는 각 구간 중앙에 가장 가까운 유효 예측창 3개를 보여준다. 표시된 window R²와 구간 전체 R²는 집계 범위가 다르다. 초기 target이 거의 일정하면 정석 R²도 크게 음수가 될 수 있으므로 RMSE·그래프를 함께 해석한다. 겹치는 window는 독립 표본이 아니다.

완료한 배터리의 점수와 대표 그림용 작은 임시 NPZ로 중단 후 재개할 수 있다. 완료 후 임시 NPZ는 삭제하고 표·조건·그래프만 보관한다. 기존 전체 T5 결과에는 원시 예측이 없어서 초·중·후반 점수를 복원할 수 없으므로, 같은 pilot 창에서 T5도 다시 추론한다.

Chronos-2의 수치 patch 입력과 직접 분위수 출력은 [공식 논문](https://arxiv.org/html/2510.15821v1), API는 [공식 저장소](https://github.com/amazon-science/chronos-forecasting)에서 확인할 수 있다. 개선되더라도 양자화만 원인이라고 단정할 수 없다. 구조와 사전학습 데이터도 함께 달라진다.

## CALCE 64→16 재실험

`calce_cross_file_windows=true`는 CALCE의 같은 배터리 기록을 시간순으로 연결한다. 파일 경계와 파일 내 `Cycle_Index`의 재시작 자체는 단절로 취급하지 않는다. 파일 내 source cycle 누락, 원래 관측 순번 누락, protocol 변경은 단절로 유지한다. 시각의 순서·겹침을 검사하고 불완전·품질 제외 관측이 포함된 창은 계속 제외한다. Capacity·SOH·기준 용량·원본 묶음은 변경하지 않는다.

CALCE의 가로축은 기존 추출본의 시간순 **관측 기록 순번**이다. 파일 사이 수시간~수일의 휴지는 그대로 timestamp에 남지만 모델 입력에 별도 time gap을 추가하지 않는다. 연결된 관측 순서가 연속된 실제 운전 시간이나 미기록 물리 cycle이 없음을 보장하지 않는다.

```powershell
# 추론 없이 4개 배터리의 64→16 창 확인
.\.venv\Scripts\python.exe scripts/run_calce_zeroshot.py --prepare-only

# T5 다음 C2를 같은 입력으로 평가
.\.venv\Scripts\python.exe scripts/run_calce_zeroshot.py --model both

# 현재 실험 로그
Get-Content results/t5-c2-battery-comparison/calce64-16/run.log -Tail 15 -Wait
```

4개 배터리의 유효 창은 CS2_35 429 / CS2_36 488 / CS2_37 504 / CS2_38 493, 총 **1,914개**다. 새 결과는 `results/<모델>/calce64-16/calce_cs2_1c/`, 비교표와 진행 상태는 `results/t5-c2-battery-comparison/calce64-16/`에 저장한다. 두 결과 확인 노트북의 기본 선택도 이 실험으로 바꿨다. 32→8 결과는 horizon과 평가 대상 창이 달라 새 점수와 직접적인 성능 개선율을 계산하지 않는다.

## R² 평가 범위

C2는 모든 배터리·모든 유효 예측창의 Mean window R²를 사용한다. T5 MIT batch1·3만 사용자 요청으로 배터리별 초·중·후반 대표 창 최대 3개를 선택했다. 각각 138개·120개 창이며 `r2_scope=representative_windows`는 전체 창 R²가 아니다. 두 그룹의 RMSE·WQL·PICP@80은 기존 전체 평가값을 유지한다. 나머지 T5 그룹은 전체 창 R²다.

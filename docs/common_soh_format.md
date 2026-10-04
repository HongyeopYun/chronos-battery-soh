# 데이터셋 Audit 및 공통 SOH 포맷 v1

데이터셋마다 측정 조건을 보존하면서 cycle별 용량과 SOH를 같은 구조로 저장한다. XJTU 2C·3C, NASA 4개, MIT 3개 batch, CALCE CS2 4개를 지원한다. 현재 NASA 02·03의 데이터·실험 코드와 저장된 결과는 유지한다. [전체 가공 현황](../results/data_audit/battery_datasets_overview.md)에서 그룹별 곡선과 필수 확인 결과를 볼 수 있다.

## 공통 데이터 묶음

| 파일 | 내용 |
|---|---|
| `soh.csv` | cycle별 관측 용량·SOH·시험 종류·품질 표시 |
| `batteries.csv` | 배터리 ID, batch, 정규화 기준, 시험 조건, 원본 이름·SHA-256 |
| `audit.csv` | 원본·추출 cycle 수, 비정상 관측, 시험 조건 일치, 예측 준비 상태 |
| `schema.json` | 포맷 버전·필드 정의·기본키 |
| `manifest.json` | 입력 파일·전처리 코드·출력 파일 해시, 적용한 규칙, 완료 상태 |

기본키는 `(dataset, battery_id, cycle)`이다. `dataset`은 소문자 식별자, `battery_id`는 dataset 안에서 유일해야 한다. 예: `xjtu`, `batch1_2C_battery-1`. 다른 dataset의 같은 배터리 이름은 별개다.

| soh.csv 열 | 정의 |
|---|---|
| `dataset`, `battery_id` | 데이터셋 및 배터리 식별자 |
| `cycle` | 원본에서의 cycle 번호. 추출·제외 후 다시 번호를 매기지 않는다. |
| `capacity_ah` | cycle별 용량, Ah 단위, float64 |
| `soh` | `capacity_ah / reference_capacity_ah`, 1 = 100%인 비율, float64. 1 초과 값도 보존한다. |
| `capacity_kind` | `reference_test`, `full_discharge`, `partial_discharge`, `unknown` |
| `is_observed` | 실제 측정값 여부. 보간·모델 추정값은 False로 구분한다. |
| `quality_flags` | 세미콜론으로 구분한 표시. 빈 문자열은 표시 없음. |
| `protocol_id` | 해당 cycle의 시험 조건 식별자 |
| `cycle_description` | 원본 cycle 설명 |

정규화 방법·기준 cycle·기준 용량은 `batteries.csv`에 기록한다. 서로 다른 정의의 SOH를 합쳐 비교하지 않는다. 온도처럼 정확한 수치를 모르는 정보는 빈 값으로 둔다. `room_temperature`는 수치 온도가 아니다.

## XJTU 2C 추출 규칙

- 입력은 `data/raw/XJTU/2C_battery-1.mat`부터 `2C_battery-8.mat`까지다. 3C와 다른 batch는 이 실행에 포함하지 않는다.
- `summary.discharge_capacity_Ah`를 추출한다. `data` cycle 수와 용량 배열 길이가 일치해야 한다.
- 초기 용량은 cycle 1의 Q1이다. 정격 2 Ah는 별도 metadata이며 SOH 분모로 사용하지 않는다.
- cycle 1은 `0.5C charge and 0.2C discharge [test capacity]`, 나머지는 `2C charge and 1C discharge`인지 확인한다. 다른 설명이면 `unknown`/`unexpected_protocol`로 표시하고 Audit을 실패로 기록한다.
- 모든 추출 가능한 cycle을 보존한다. 첫 cycle은 `reference_test`와 `initial_capacity_test`로 표시한다. cycle 2를 새 초기 용량으로 정규화하지 않는다.
- cycle 번호는 공개 MAT `data` 배열의 1-based 위치다. 제작자 코드와 같은 인덱스이지만 원래 장비 기록의 누락 여부까지 검증한 것은 아니다. `cycle_number_source`에 이 한계를 기록한다.
- `summary.cycle_life`는 metadata로 보존하며 행을 자르는 기준으로 사용하지 않는다.
- 용량 증가·SOH > 1은 각각 `capacity_increase`, `soh_above_one`으로 표시하고 보존한다. 보간, 평활화, clipping, 강제 단조 감소를 하지 않는다.
- 같은 시험 조건에서 직전 cycle 대비 SOH 변화의 절댓값이 0.01(1%p)을 초과하면 `large_soh_change`로 표시한다. 이는 설정 가능한 검토 기준이며 측정 오류 판정이나 예측 제외 기준이 아니다. 시험 조건 변경은 `protocol_transition`으로 구분한다. 주요 표시가 있는 관측은 보고서의 `xjtu_2c_quality_events.csv`에도 저장한다.
- NaN·무한대·0 이하 용량도 행을 유지하고 표시한다. 초기 기준 용량 자체가 유효하지 않으면 대체 기준을 자동 선택하지 않고 해당 파일을 실패로 기록한다.
- [제작자의 실험 설명](https://wang-fujin.github.io/)과 공개 MAT의 cycle 설명을 함께 사용한다. 다른 XJTU batch의 부분 방전·기준 시험 측정은 별도 추출 규칙이 필요하다.

## 실행과 결과

3C는 필수 확인과 SOH 곡선 중심으로 간소화했다. `summary.discharge_capacity_Ah`와 cycle 수 일치, 유효한 용량, 첫 cycle의 초기 시험 및 이후 3C 충전·1C 방전 조건, 공통 포맷의 기본키·Q/Q1 일관성을 확인한다. 용량 회복·급변은 곡선에서 확인하며 별도 이벤트 목록이나 변화량 임계값 Audit을 만들지 않는다. 모든 관측값을 보존하는 원칙과 초기 시험을 예측 window에서 제외하는 규칙은 2C와 같다. 기존 2C 결과는 유지한다.

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_xjtu_3c.py
```

3C 설정은 `configs/xjtu_3c_preprocess.json`, 전체 데이터는 `data/processed/XJTU/3C/`에 저장한다. 검토용 결과는 [3C SOH 곡선과 요약](../results/data_audit/xjtu_3c/xjtu_3c_report.md), `xjtu_3c_soh_curves.png`, `xjtu_3c_batteries.csv`, 간소화한 `xjtu_3c_audit.csv`만 보관한다. 정상적인 필수 확인 결과는 `passed`다. 전체 데이터 묶음의 스키마·해시는 로컬 manifest에 유지한다.

`results/data_audit/`의 검토용 파일은 `<dataset>_<group>_<내용>.<확장자>`로 이름을 붙인다. 예: `xjtu_2c_audit.csv`, `xjtu_3c_soh_curves.png`. 파일만 따로 열거나 복사해도 데이터셋과 시험 그룹을 구분할 수 있다. `data/processed/`의 공통 데이터 묶음은 기존 공통 파일명과 로더 규칙을 사용한다.

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_xjtu_2c.py
```

설정은 `configs/xjtu_2c_preprocess.json`, 전체 데이터는 `data/processed/XJTU/2C/`, 검토용 결과는 `results/data_audit/xjtu_2c/`에 저장한다. 전체 가공 데이터는 Git에서 제외하고 Audit·metadata·manifest·schema·미리보기·SOH 그림은 보관한다. 원본 파일은 읽기만 한다.

파일 하나라도 해석이나 Audit에 실패하면 manifest가 `audit_failed`가 되고 CLI가 실패 종료한다. `src.soh_data.load_bundle`도 이 상태의 데이터를 예측에 넘기지 않는다. `complete`일 때는 산출물 해시와 기본키·정규화 일관성을 다시 검사한다. 같은 입력·코드·설정으로 다시 실행하면 동일한 데이터 CSV를 생성한다.

## Zero-shot 입력 준비

예측용 급등 이상치 제거본은 다음 명령으로 별도 생성한다. [규칙·제거 수·전후 그림](../results/data_audit/outlier_cleaning/soh_outlier_cleaning_report.md)을 확인할 수 있다.

```powershell
.\.venv\Scripts\python.exe scripts\clean_soh_outliers.py
```

`configs/soh_outlier_cleaning.json`에 고정한 보수적 규칙을 사용한다. 이전 최대 10개 유효 관측이 최소 5개 있을 때, 현재 SOH가 1.2 초과이고 과거 중앙값보다 0.2(20%p) 초과 높으면 `capacity_outlier`로 표시한다. 배터리·protocol·segment·cycle 연속성이 바뀌면 과거 이력을 다시 시작한다. 이후 관측과 전체 수명 통계로 판정하지 않는다. 작은 회복·급격한 열화·순간 하락은 이 규칙의 자동 제거 대상이 아니다.

`data/processed/clean/<group>/soh.csv`는 해당 급등 관측을 제외한 데이터이며 기존 묶음은 그대로 보존한다. 같은 폴더의 `soh_flagged.csv`는 모든 행에 제외 표시를 남긴 비교용 데이터다. 정규화 기준·cycle 번호는 바꾸지 않고 보간하지 않는다. `observed_windows`는 제거로 생긴 cycle 간격을 건너뛰지 않는다. 표시본을 넘기면 `capacity_outlier`도 항상 제외하며, 기존 초기 시험·알 수 없는 protocol·수집 문제 제외 규칙도 유지한다.

```python
soh, batteries, manifest = load_bundle(Path("data/processed/clean/mit_batch1"))
windows = observed_windows(soh[soh.battery_id == "b1c18"], 64, 16)
```

현재 4개 관측과 이들을 포함한 210개 window를 제외했다. target의 품질 제외 여부는 그 측정값을 관측한 후 결정되며, 정제된 평가 구간과 원래 구간은 다르다. 제거 전후 성능을 비교할 때 같은 평가 구간을 사용하고 제외 수를 함께 보고한다. 기존 NASA 02·03 실험 결과를 재평가한 것은 아니다.

NASA·MIT·CALCE는 다음 명령으로 추출한다. dataset을 제한하려면 `--datasets mit calce`처럼 지정한다.

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_remaining_batteries.py
```

- NASA: discharge의 `data.Capacity`를 사용하고 기존 02·03 실험과 같은 첫 방전 Q1 기준으로 정규화한다. 정격 2 Ah는 별도 metadata다. 전체 MAT에서의 `source_record_index`도 보존한다. [제작자 설명](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets).
- MIT: HDF5의 `summary.QDischarge`와 `summary.cycle`만 읽는다. Batch 1의 cycle 1은 0인 미측정값으로 보존하고 `is_observed=False`로 표시한다. 기준은 명시적으로 cycle 2다. 나머지 신규 배터리는 첫 cycle 기준이다.
- MIT의 `b2c7/8/9/15/16`은 각각 `b1c0/1/2/3/4`의 연속 측정이다. 이 5개는 원래 기준 용량과 `physical_battery_id`를 공유하고 `cycle`에 앞 구간 길이를 더한다. 원래 번호는 `source_cycle`에 보존한다. 기록은 합치지 않으며 train/test 분리 시 동일 physical ID를 함께 묶어야 한다. [제작자 연결 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/Load%20Data.ipynb).
- MIT Batch 3의 `b3c2/23/32/37/42/43`은 제작자가 noisy channel로 지정했다. 모두 보존하고 `known_collection_issue`로 표시하며 기본 window에서 제외한다. 전체 수명이나 최종 용량을 기준으로 다른 배터리를 자동 제거하지 않는다.
- CALCE: `Discharge_Capacity(Ah)`는 누적값이다. cycle별로 방전 종료 누적값에서 방전 직전 누적값을 뺀다. 2.7 V까지 방전되지 않았거나 시작이 잘린 관측은 부분 방전으로 보존하고 예측에서 제외한다. 기준은 첫 완료 관측이다. 용량 증가·급변만으로 제거하지 않는다. [제작자 시험 조건](https://web.calce.umd.edu/batteries/data/).
- CALCE 원본 `Cycle_Index`는 날짜별로 다시 시작한다. 시간순 관측 번호를 `cycle`, 원본 번호를 `source_cycle`로 기록한다. 중복 측정 구간의 용량·상태가 같으면 한 번만 사용하며 원본 파일 이름을 모두 남긴다. 일치하지 않는 겹침은 실패 처리한다. 실제 장비 cycle 누락 여부는 이 번호만으로 확정할 수 없다.
- 선택 열 `segment_id`가 있으면 `observed_windows`는 같은 segment 안에서만 window를 만든다. CALCE는 원본 파일을 segment로 구분하며 최대 50 cycle이라 현재 64→16 window가 없다. 예측 전에 context를 줄이거나 파일 간 연속성을 확인한다. MIT는 원본 배터리 기록을 segment로 유지한다.

```python
from pathlib import Path
from src.soh_data import load_bundle, observed_windows

soh, batteries, manifest = load_bundle(Path("data/processed/XJTU/2C"))
rows = soh[soh.battery_id == "batch1_2C_battery-1"]
windows = observed_windows(rows, context_length=64, prediction_length=16)
```

`observed_windows`는 배터리 하나씩 처리한다. 첫 초기 시험, 비정상 용량, 알 수 없는 시험 조건은 예측 입력·target에서 제외한다. 부분 방전량과 보간값도 제외한다. 용량 증가와 SOH > 1은 제외하지 않는다.

80개 관측 전체가 연속된 cycle이고 실제 측정이어야 64→16 window를 만든다. 중간 행을 제거한 뒤 서로 다른 구간을 이어 붙이거나 누락 cycle을 보간하지 않는다. `forecast_start`와 `target_cycle`은 실제 source cycle 번호를 유지한다. 이번 XJTU 정책에서 첫 target은 cycle 66이다: context 2~65, target 66~81.

이 단계는 SOH 추출·Audit·윈도우 준비다. Chronos zero-shot 예측은 다음 단계에서 이 묶음을 읽어 실행한다.

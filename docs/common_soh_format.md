# 데이터셋 Audit 및 공통 SOH 포맷 v1

데이터셋마다 측정 조건을 보존하면서 cycle별 용량과 SOH를 같은 구조로 저장한다. 최초 구현은 XJTU Batch 1의 2C 충전 배터리 8개다. 현재 NASA 02·03의 데이터·실험 코드와 저장된 결과는 유지한다.

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
- 같은 시험 조건에서 직전 cycle 대비 SOH 변화의 절댓값이 0.01(1%p)을 초과하면 `large_soh_change`로 표시한다. 이는 설정 가능한 검토 기준이며 측정 오류 판정이나 예측 제외 기준이 아니다. 시험 조건 변경은 `protocol_transition`으로 구분한다. 주요 표시가 있는 관측은 보고서의 `quality_events.csv`에도 저장한다.
- NaN·무한대·0 이하 용량도 행을 유지하고 표시한다. 초기 기준 용량 자체가 유효하지 않으면 대체 기준을 자동 선택하지 않고 해당 파일을 실패로 기록한다.
- [제작자의 실험 설명](https://wang-fujin.github.io/)과 공개 MAT의 cycle 설명을 함께 사용한다. 다른 XJTU batch의 부분 방전·기준 시험 측정은 별도 추출 규칙이 필요하다.

## 실행과 결과

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_xjtu_2c.py
```

설정은 `configs/xjtu_2c_preprocess.json`, 전체 데이터는 `data/processed/XJTU/2C/`, 검토용 결과는 `results/data_audit/xjtu_2c/`에 저장한다. 전체 가공 데이터는 Git에서 제외하고 Audit·metadata·manifest·schema·미리보기·SOH 그림은 보관한다. 원본 파일은 읽기만 한다.

파일 하나라도 해석이나 Audit에 실패하면 manifest가 `audit_failed`가 되고 CLI가 실패 종료한다. `src.soh_data.load_bundle`도 이 상태의 데이터를 예측에 넘기지 않는다. `complete`일 때는 산출물 해시와 기본키·정규화 일관성을 다시 검사한다. 같은 입력·코드·설정으로 다시 실행하면 동일한 데이터 CSV를 생성한다.

## Zero-shot 입력 준비

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

# 원시 기록 기반 SOH 전처리 v2

현재 64→1 재실험은 [reviewed_native_v3](../results/preprocessing_reviewed_native_v3/report.md)를 사용한다. MIT의 native 용량 우선 정책과 원래 위치를 보존한 결측 입력을 적용한 별도 버전이며, 아래 strict_v2 bundle을 덮어쓰지 않았다.

> 2026-10-08 재검토: 이 문서는 기존 strict_v2의 규칙을 설명하며, 전체 수명 평가용 최종 확정본이 아니다. [전체 감사](../results/preprocessing_review_all/report.md)에서 NASA의 로그 부재 판정, MIT의 CV 적분 불일치·긴 기록 공백·복수 방전 처리, CALCE의 매우 낮은 창 커버리지를 확인했다. NASA 수정본은 별도 `nasa_review_v3`에 있으며 MIT/CALCE 마스크는 아직 변경하지 않았다. 아래 실행 명령으로 과거 strict_v2를 덮어쓰기 전에 새 출력 버전을 지정해야 한다.

목적은 정상적인 완전 충·방전 조건에서 측정한 용량으로 SOH 예측 데이터를 만드는 것이다. 곡선을 매끈하게 만들거나 R²를 높이는 것을 제외 기준으로 삼지 않는다. 원시 데이터와 기존 실험 결과는 변경하지 않는다.

## 산출물과 실행

- 새 공통 데이터: `data/processed/strict_v2/<group>/soh.csv`, `batteries.csv`, `manifest.json`.
- 결과·전후 곡선: `results/preprocessing_strict_v2/report.md`.
- 모든 cycle의 원시 검증 수치: `results/preprocessing_strict_v2/evidence/*_raw_features.csv.gz`.
- 제외 관측과 사유: `results/preprocessing_strict_v2/excluded_cycles.csv`.

```powershell
.\.venv\Scripts\python.exe scripts/preprocess_strict_batteries.py
```

원시 파일 체크섬을 기존 추출 시점과 대조하고, 모든 배터리의 모든 cycle에 원시 검증 결과를 1:1 대응한다. 중간 증거는 캐시해 재개할 수 있다. 원시 검증을 다시 하려면 `--stage features --force`, 판정·그림만 다시 만들려면 `--stage finalize`를 사용한다. 학습·추론은 실행하지 않는다.

기존 실험 스크립트의 `clean` 기본 경로는 과거 결과를 보존하기 위해 유지한다. 새 실험에서는 bundle을 명시적으로 지정한다. 예를 들어 모델을 실행하지 않고 새 데이터의 유효 창을 읽으려면:

```python
from src.soh_data import load_bundle, observed_windows
data, batteries, manifest = load_bundle("data/processed/strict_v2/xjtu_2c")
rows = data[data.battery_id == "batch1_2C_battery-1"]
windows = observed_windows(rows, context_length=64, prediction_length=16)
```

CALCE 엄격본은 원 파일 식별자(`source_segment_id`)와 관측 순서 구간(`segment_id`)을 따로 보존한다. 파일 경계를 다시 처리해도 같은 결과가 나오고, 제외된 cycle은 계속 창을 끊는다.

## 공통 원칙

1. `capacity_ah`, `soh=Q/Qref`, `cycle`을 유지한다. 단위나 카운터 오류가 의심돼도 근거 없이 용량을 추정해 덮어쓰지 않는다.
2. `quality_eligible=False`인 행은 보관하되 학습·검증·테스트에서 제외한다. 공통 `observed_windows`는 사용자 지정 제외 플래그와 관계없이 이 마스크와 `strict_quality_excluded`를 강제한다.
3. 제외된 cycle을 건너뛰어 64→16을 붙이지 않는다. 결측 보간, 이동평균, 단조 감소 강제, cycle 재번호 부여가 없다.
4. 충전 완료 및 방전 cutoff, 시간 순서, 카운터 일관성, 원시 용량과 추출값을 대조한다. 원시 전압·전류·시간이 없거나 불완전해 확인할 수 없으면 별도의 제외 이유를 남긴다.
5. 급락·회복 자체는 오류 증거가 아니다. 직전 SOH보다 절대 0.03 이상 변하면 `shape_review`를 표시하되 이 사실만으로 제외하지 않는다. 과거 대비 변화만 사용한다.
6. 같은 물리 배터리의 연속 기록을 독립 train/test 배터리로 취급하지 않는다. 기존 MIT `physical_battery_id`와 continuation 정보를 유지한다.

## 데이터셋별 검증

| 데이터 | 충전 확인 | 방전·용량 확인 |
|---|---|---|
| CALCE CS2 35–38 | CV step 4 존재, 마지막 충전 전압 ≥4.15V, 전류 ≤0.075A | 1.1A 방전, 2.7V 도달, 누적 방전 카운터 차분과 시간 적분 대조 |
| XJTU 2C/3C | 충전 종료 ≥4.15V 및 ≤0.15A | 초기 시험 0.4A / 일반 2A, 2.5V 도달, summary와 방전 단계 카운터 및 적분 대조 |
| NASA B0005/6/7/18 | 직전 charge 기록 존재, 종료 ≥4.15V 및 ≤0.12A | 2A 방전, Capacity 정의인 2.7V까지 적분; 실제 더 낮은 방전 cutoff와 혼동하지 않음 |
| MIT Batch 1–3 | 충전 종료 ≥3.55V 및 ≤0.10A | 방전 CC plateau 4.4A, 2.0V 도달; native QDischarge와 Qd 카운터 및 적분 대조 |

전압 cutoff는 표의 값에 +0.05V 허용오차를 둔다. NASA 이외 데이터의 마지막 방전 전압이 cutoff보다 0.10V 넘게 높으면 종료 기록 이상으로 제외한다. CC 전류 허용오차는 0.15A와 기대 전류의 15% 중 큰 값이다. 전압이 0.5V 미만 또는 MIT 3.9V / 나머지 4.5V 초과이면 원시 기록 이상으로 제외한다. 이 기준은 성능 점수로 탐색하지 않았다.

용량-카운터 허용오차는 0.005Ah와 용량의 1% 중 큰 값, 용량-시간적분 허용오차는 0.02Ah와 용량의 3% 중 큰 값이다. 후자는 검증을 위한 보수적 허용치이며, 오차 분포가 이 값 이하라는 통계적 보장은 없다. 초과 관측은 `capacity_integral_unverified`로 격리한다. 이는 용량이 반드시 잘못됐다는 단정이 아니다.

`quality_status`로 확실한 제외(`excluded`)와 확인 보류(`quarantined`)를 구분한다. 적분 대조 실패나 직전 충전 기록 부재만 있는 경우는 후자다. 두 상태 모두 엄격한 기본 데이터에서는 `quality_eligible=False`지만 원시값과 기존 결과는 보존한다. 특히 MIT Batch 3의 CV 구간에서는 전류 적분과 용량 카운터 간 체계적 차이가 발견되어 많은 관측이 보류됐다. 유효 데이터가 열화 후반을 덜 포함할 수 있으므로, 보류 이유를 해결하기 전 이 엄격본을 전체 수명 평가를 위한 최종 확정본으로 주장하지 않는다.

CALCE는 CC 시작 후 첫 기록까지의 `Step_Time(s)`도 적분에 포함한다. XJTU의 `relative_time_min`은 단계마다 초기화되므로 전체 cycle에서 시간이 감소한다는 이유로 제외하지 않고 방전 단계 내부에서 확인한다. NASA는 부하 제거 후 반등한 마지막 전압 대신 실제 부하가 걸린 구간을 확인한다.

MIT 공개 MAT의 I 값은 방전 plateau가 약 -4로 기록된다. 기존 구현은 1.1 배율을 적용했고 CC 구간에서는 원시 Qd 증가량과 일치하는 경향을 보였다. 그러나 독립 재검토에서 Batch 3의 CV 구간에는 같은 배율이 맞지 않았다. 따라서 이를 모든 구간에 검증된 A 단위 변환으로 단정할 수 없다. 배율은 evidence의 `current_scale_to_a`에 남아 있으며 native QDischarge나 SOH는 변환하지 않았다. 이 불일치를 해결하기 전 전류 적분 오차만으로 원본 용량이 잘못됐다고 확정하지 않는다.

온도 범위를 벗어나거나 방전 샘플 간격이 크게 벌어진 경우 검토 메모를 남긴다. 온도 센서 문제만으로 용량을 삭제하지 않는다. MIT의 원제작자 지정 noisy channel 6개, 미측정 placeholder, XJTU 초기 별도 프로토콜 등 기존 제외 조건도 유지한다. CALCE 중복 export는 용량뿐 아니라 충전과 trace 검증 수치까지 대조한다.

## 누수와 평가 범위

각 관측의 품질 판정은 해당 cycle 종료 시점까지의 원시 측정만 사용한다. NASA는 앞선 charge 기록을 연결한다. 미래 SOH 기반 이상치 보정·중앙 이동평균·정답 보간은 없다.

평가용 창은 context와 target 모두 유효한 측정으로 구성한다. 따라서 결과는 **정상적인 완전 충·방전 조건의 SOH 예측** 성능이며, 현실의 모든 불완전 운전 상황을 포함한 성능이 아니다. 제외율과 유효 창 수를 반드시 함께 보고한다. 정제 전후 평가 창이 달라지면 점수 차이는 전처리 효과만 나타내지 않으므로, 모델 비교는 동일한 창에서 다시 해야 한다.

특히 CALCE에서 CV 단계 누락이 자주 반복되면 정제 후 연속 80개 관측이 부족할 수 있다. 이를 해결하려고 제외 cycle을 단순히 삭제한 뒤 시간축을 압축하면 기존 64→16 문제 정의가 바뀐다. 결측을 다루는 입력 방식이나 별도 평가 설계는 후속 실험으로 다뤄야 한다.

## 제작자 자료

- [CALCE 시험 조건과 데이터 설명](https://web.calce.umd.edu/batteries/data/)
- [XJTU 데이터 제작자와 프로토콜](https://wang-fujin.github.io/)
- [NASA 원 데이터 설명](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets)
- [MIT 원 제작자 데이터 처리 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)

제작자 프로토콜은 기준 조건의 출처이며, 이 문서의 수치 허용오차와 제외 정책은 이번 검토에서 정한 구현상의 판단이다.

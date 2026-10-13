# 전체 배터리 원시 기록 기반 전처리

원본 용량·SOH·cycle 번호는 보존하고, 품질 기준을 통과한 관측만 예측에 사용한다. 보간·평활화·모델 실행 없음.

| 데이터셋 | 기록 수 | 전체 cycle | 기존 유효 | 정제 후 유효 | 추가 제외·보류 | 확인 보류 | 64→16 창: 전→후 |
|---|---:|---:|---:|---:|---:|---:|---:|
| nasa_aging | 4 | 636 | 636 | 633 | 3 | 3 | 320 → 83 |
| xjtu_2c | 8 | 3219 | 3211 | 3209 | 2 | 0 | 2579 → 2577 |
| xjtu_3c | 15 | 3742 | 3727 | 3727 | 0 | 0 | 2542 → 2542 |
| calce_cs2_1c | 4 | 3937 | 3911 | 3801 | 110 | 0 | 1914 → 69 |
| mit_batch1 | 46 | 38811 | 38765 | 38728 | 37 | 18 | 35131 → 34325 |
| mit_batch2 | 48 | 24920 | 24920 | 24898 | 22 | 1 | 21128 → 19635 |
| mit_batch3 | 46 | 51007 | 41240 | 31371 | 9869 | 9867 | 38080 → 25398 |

![주요 전처리 전후](key_before_after.png)


## 사용과 해석

- 새 데이터는 `data/processed/strict_v2/<group>`에 저장했다. 기존 데이터·모델 결과는 유지했다.
- `soh.csv`는 모든 관측을 보존한다. **quality_eligible=True만 사용**하며, 공통 `observed_windows`가 이를 강제한다. 제외 행을 삭제한 뒤 cycle을 다시 번호 매기면 안 된다.
- 원래의 64→16 연속 창에서 제외 cycle을 건너뛰거나 채우지 않는다. 유효 창 감소는 실제 품질 제약이다.
- `quality_status=quarantined`는 **오류 확정이 아니라 확인 보류**다. 엄격한 기본 마스크에서는 제외하지만, 삭제·보정하지 않는다. 빨강은 제외, 주황은 확인 보류다.
- 충전 미완료, 방전 종료 미도달, 비정상 전압/전류, 시간·카운터 오류, 용량 대조 실패를 구분했다. 재구성 근거가 부족한 용량은 임의로 수정하지 않았다.
- 형태만 튀는 점은 삭제하지 않았다. 과거 SOH 대비 0.03 이상 변한 점은 검토 표시만 한다. 온도·휴지시간·조건 변화로 인한 실제 용량 변동이 남을 수 있다.
- 미래 SOH를 이용한 보정과 정답 평활화는 없다. 단, 평가 대상은 미래 16개 cycle의 측정 품질까지 확인된 창이므로 '유효한 완전 충·방전 조건의 SOH 예측' 평가임을 명시해야 한다.
- 이전 결과와 정제 후 결과는 평가 창이 달라 직접 개선율을 비교할 수 없다. 재평가 시 모든 모델에 동일한 정제 창을 사용해야 한다.
- MIT 연속 기록 5쌍은 기존 physical_battery_id를 유지한다. train/validation/test에서 같은 물리 배터리를 분리하면 안 된다.

## 근거와 규칙

- [상세 규칙](../../docs/strict_preprocessing.md)
- [배터리별 요약](battery_summary.csv), [제외 cycle 및 이유](excluded_cycles.csv), [이유별 집계](exclusion_counts.csv)
- `evidence/`에 모든 cycle의 충전 종료·방전 종료·적분·카운터·원본 대응 정보를 보관했다.

## 핵심 확인 사항

- CALCE의 표시된 급락 중 cycle 127·146·157·169·708·716·738(CS2_35)에서는 CV step 4가 없고 충전이 약 0.55A에서 종료됐다. 정상 인접 cycle은 약 0.05A까지 감소한다. 같은 완전 충전 용량으로 비교하기 어려워 제외했다.
- NASA B0005/6/7의 cycle 90은 충전 기록 없이 다음 방전이 시작된다. 용량 오류로 단정하지 않고 확인 보류했다. 이 한 점 때문에 각 배터리에서 연속 80개 관측 창이 크게 줄어든다. B0018과 다른 검증된 용량 회복은 유지했다.
- MIT Batch 3에서는 native QDischarge와 Qd 카운터가 맞아도 전류 시간적분이 수 % 더 큰 관측이 다수 있다. 표본을 단계별로 확인하면 주로 CV 방전에서 차이가 커진다. 적분 차이만 있는 관측은 오류라고 단정하지 않고 보류했으며, **실제 열화 후반 데이터까지 빠질 수 있어 이 엄격본의 성능을 전체 수명 성능으로 해석하면 안 된다.** 원 배포본의 CV 기록·카운터 정의를 확인하기 전 대규모 삭제나 재적분 대체는 하지 않았다.
- MIT 확인 보류까지 제외한 것은 보수적인 검증 정책이다. 보류 관측의 재사용 여부를 결정하기 전에는 원래 MIT 결과를 정제 완료 결과로 대체하거나 모델 우열을 다시 결론내리지 않는다.

기준점 검토 표시가 있는 기록 6개(기존 noisy channel 포함). `battery_summary.csv`의 reference_review를 확인한다. 기준 용량은 임의 변경하지 않았다.

## nasa_aging

![전처리 전후](nasa_aging_before_after_01.png)

## xjtu_2c

![전처리 전후](xjtu_2c_before_after_01.png)

## xjtu_3c

![전처리 전후](xjtu_3c_before_after_01.png)

![전처리 전후](xjtu_3c_before_after_02.png)

## calce_cs2_1c

![전처리 전후](calce_cs2_1c_before_after_01.png)

## mit_batch1

![전처리 전후](mit_batch1_before_after_01.png)

![전처리 전후](mit_batch1_before_after_02.png)

![전처리 전후](mit_batch1_before_after_03.png)

![전처리 전후](mit_batch1_before_after_04.png)

## mit_batch2

![전처리 전후](mit_batch2_before_after_01.png)

![전처리 전후](mit_batch2_before_after_02.png)

![전처리 전후](mit_batch2_before_after_03.png)

![전처리 전후](mit_batch2_before_after_04.png)

## mit_batch3

**곡선 전체가 빨간색인 b3c2·b3c23·b3c32·b3c37·b3c42·b3c43은 원 제작자가 noisy channel로 제외한 6개 기록이다. 이번 처리에서 새로 통째로 제외한 것이 아니다. 주황색은 별도로 이번 검사에서 확인 보류한 관측이다.**

[원 제작자의 제외 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/Load%20Data.ipynb)

![전처리 전후](mit_batch3_before_after_01.png)

![전처리 전후](mit_batch3_before_after_02.png)

![전처리 전후](mit_batch3_before_after_03.png)

![전처리 전후](mit_batch3_before_after_04.png)

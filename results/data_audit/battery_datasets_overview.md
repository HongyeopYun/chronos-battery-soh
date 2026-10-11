# 배터리 데이터 가공 현황

raw 폴더의 데이터에서 cycle별 용량과 SOH를 공통 포맷으로 추출했다. 검토 결과는 필수 오류 확인과 관측 곡선이다.

| 그룹 | 배터리 기록 | 저장 cycle | 64→16 windows | 곡선·요약 |
|---|---:|---:|---:|---|
| xjtu_2c | 8 | 3,219 | 2,579 | [보기](xjtu_2c/xjtu_2c_report.md) |
| xjtu_3c | 15 | 3,742 | 2,542 | [보기](xjtu_3c/xjtu_3c_report.md) |
| nasa_aging | 4 | 636 | 320 | [보기](nasa_aging/nasa_aging_report.md) |
| mit_batch1 | 46 | 38,811 | 35,131 | [보기](mit_batch1/mit_batch1_report.md) |
| mit_batch2 | 48 | 24,920 | 21,128 | [보기](mit_batch2/mit_batch2_report.md) |
| mit_batch3 | 46 | 51,007 | 38,080 | [보기](mit_batch3/mit_batch3_report.md) |
| calce_cs2_1c | 4 | 3,937 | 0 | [보기](calce_cs2_1c/calce_cs2_1c_report.md) |

총 **126,272행**을 저장했다. 아직 새로운 데이터셋의 모델 예측은 실행하지 않았다.

- MIT 140개 기록은 연속 측정 구간 5개를 포함한 물리 배터리 135개다. 분할 시 `physical_battery_id`를 함께 묶는다.R
- CALCE는 중복 내보내기를 제거하고 파일별 측정 구간을 구분했다. 파일당 최대 50 cycle이므로 현재 64→16 window는 없다. 예측 시 context를 줄이거나 파일 간 연속성을 확인해야 한다.
- NASA 공통 포맷은 기존 02·03 실험과 같은 Q/Q1 기준이다.
- 전체 CSV는 `data/processed/<dataset>/<group>/`에 저장하며 Git에서 제외한다. 검토 파일명에는 dataset과 group을 넣었다.

예측용 급등 이상치 제거본: `data/processed/clean/<group>/`. [제거 목록·전후 그림·window 수](outlier_cleaning/soh_outlier_cleaning_report.md). 위 표는 기존 가공본 기준이며 제거본 수치는 해당 보고서에 기록했다.

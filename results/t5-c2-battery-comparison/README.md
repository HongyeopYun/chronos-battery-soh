# T5 / Chronos-2 battery zero-shot pilot

- [t5-battery/mit_batch3](../t5-battery/pilot/mit_batch3/t5-battery_report.md)
- [t5-battery/xjtu_3c](../t5-battery/pilot/xjtu_3c/t5-battery_report.md)
- [c2-battery/mit_batch3](../c2-battery/pilot/mit_batch3/c2-battery_report.md)
- [c2-battery/xjtu_3c](../c2-battery/pilot/xjtu_3c/c2-battery_report.md)

동일한 일부 배터리·64→16·stride 1 비교다. Mean window R²와 관측 기간 초·중·후반 점수를 사용한다. 창별 R²를 단순평균한 뒤 배터리를 동일 가중 평균한다. 전체 T5 결과도 같은 정의로 갱신한다. 완료된 그룹만 CSV에 포함되며 데이터셋 사이 점수를 합치지 않는다.

두 모델의 모든 pilot 그룹이 완료됐고 입력 조건과 단순 기준 점수의 일치 여부를 확인했다.

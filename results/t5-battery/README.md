# 모든 공통 SOH 데이터의 zero-shot

- [nasa_aging: 64→16](nasa_aging/report.md)
- [calce_cs2_1c: 32→8](calce_cs2_1c/report.md)
- [xjtu_2c: 64→16](xjtu_2c/report.md)
- [xjtu_3c: 64→16](xjtu_3c/report.md)
- [mit_batch2: 64→16](mit_batch2/report.md)
- [mit_batch1: 64→16](mit_batch1/report.md)
- [mit_batch3: 64→16](mit_batch3/report.md)

그룹별 모델 점수는 summary.csv에 있다. 배터리 기록에 같은 가중치를 주며, 그룹 간 점수를 하나로 합치지 않는다. R²는 한 창의 실제 미래 평균을 기준으로 계산하고 창별 점수를 단순평균한 뒤 seed·배터리를 동일 가중 평균한다. r2_scope=representative_windows인 T5 MIT batch1·3은 대표 창만 평가했으며 나머지 지표는 기존 전체 값이다. C2는 전체 창 평가를 유지한다. CALCE는 결과 경로별 context/horizon을 확인한다.

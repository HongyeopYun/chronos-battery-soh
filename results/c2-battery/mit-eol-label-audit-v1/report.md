# MIT EOL 원본 라벨 빠른 검토

기존 데이터·모델·성능표는 변경하지 않았다. 이 폴더는 신규 학습 결과가 아니라 원본 수명 라벨 감사 결과다.

## 결론

- 원 논문의 EOL은 정격용량 1.1Ah의 80%, 즉 0.88Ah이다. 기존 실험은 초기 측정용량 Q_ref의 80%였으므로 서로 다른 목표를 예측했다.
- 제작자 처리 규칙을 적용한 수명 라벨은 Batch1 41셀, Batch2 43셀, Batch3 40셀, 총124개 물리 셀에서 사용할 수 있다.
- Batch1·3의 유효81셀은 제공된 cycle_life가 마지막 저장 cycle보다 정확히1 크다. 저장된 용량 곡선에는 0.88Ah 교차가 없다. 관측 교차가 없다는 사실을 곧바로 원본 수명 라벨 부재로 해석하면 안 된다.
- 마지막 cycle+1을 새 정답으로 추측하지 않는다. 별도 제공된 cycle_life를 읽고 제작자의 예외·연속 측정 보정을 적용한다. 미도달5셀도 숫자상 last+1을 가지므로 단순 덧셈은 잘못된 정답을 만든다.
- Batch2 43셀 모두 원본 cycle_life와 원시 QDischarge<=0.88 최초 cycle이 정확히 일치한다. 기존 초기용량80% 교차는 공식라벨보다 8~30cycle 늦다(중앙값 13).
- b2c40·b2c46은 현재 QC가 최초 교차 관측을 제외해 적격 관측으로 재계산하면 1cycle 늦어진다. 공식 라벨과 입력 관측 QC는 별도 관리해야 한다. 입력에서 제외된 점을 복원하거나 라벨을 임의로 바꾸지 않았다.
- Batch1의 b1c8/10/12/13/22는 제작자가 80% 미도달로 지정했다. 수명 숫자가 들어 있어도 exact EOL로 사용하지 않는다. 원래 추적 기록을 유지하고 검열 셀로 따로 취급한다.
- Batch3의 b3c2/23/32/37/42/43은 제작자 noisy channel 제외 규칙을 유지한다.
- b1c0..4에 b2c7/8/9/15/16을 합치고 cycle_life에는 각각662/981/1060/208/482를 더한다. 보정 EOL은1852/2160/2237/1434/1709이며 독립적인 Batch2 셀로 중복 사용하지 않는다.

## 후속 실험에 적용할 정의

- MIT SOH_nominal=Q/1.1. EOL은 검증된 제작자 cycle_life, RUL(t)=EOL-t.
- 원래 cycle 위치와 적격 입력 마스크를 유지하고 t<EOL만 평가한다. Batch2의 EOL 이후 기록은 RUL 학습·평가에서 제외한다.
- 초기용량 정규화를 입력 표현으로 유지할 수도 있지만, 그때는 궤적의 임계값을 셀별0.88/Q_ref로 바꿔야 한다. 무조건0.8을 적용하면 같은 EOL이 아니다.
- 별도 SOH 용량 교차가 저장되지 않은 Batch1·3에서도 공식 EOL 라벨로 직접 RUL과 궤적의 EOL 예측을 평가할 수 있다. 미래 SOH 곡선 자체는 관측된 범위까지만 평가한다.
- 배치별 성능과 교차배치 일반화를 분리하고, 같은 물리 셀을 학습·검증·시험에 중복 배치하지 않는다.
- 정격80% 기준으로 바꾸면 목표와 데이터 범위가 바뀐다. 기존 결과는 초기용량80% 실험으로 보존하고 새 실행으로 비교해야 한다.

## 근거

- [원 논문](https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf)
- [제작자 Load Data](https://raw.githubusercontent.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/master/Load%20Data.ipynb)
- 로컬 data/raw/MIT/MIT_Batch_1.mat, MIT_Batch_2.mat, MIT_Batch_3.mat의 cycle_life·summary를 직접 대조했다.
- [전체135개 물리 셀 감사](all_physical_cells_audit.csv) · [사용 가능124개 라벨](usable_official_eol_labels.csv) · [QC 차이](qc_crossing_differences.csv) · [검증](verification.json)

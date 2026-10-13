# mit_batch3 SOH 추출

기록 **46개**, cycle **51,007개**. 필수 확인: **complete**.

용량·cycle 배열, 중복·순서, 정규화 계산을 확인했다. 유효하지 않거나 불완전한 관측은 표시해서 예측 window에서 제외한다. 고유 관측과 원래 capacity는 보존하며 보간·평활화·이상치 자동 제거를 하지 않았다.

Batch 1의 cycle 1은 0인 미측정값으로 보존하고 SOH 기준은 cycle 2 용량으로 명시했다. batch를 이어 측정한 5개 기록은 원래 배터리의 기준 용량과 physical_battery_id를 공유한다. train/test 분리 시 이 ID를 함께 묶어야 한다. 제작자가 noisy channel로 표시한 Batch 3의 6개 기록은 곡선과 데이터에 보존하고 예측 window에서는 제외한다. 기록 140개는 물리 배터리 135개에 해당한다.

![mit_batch3 SOH curves](mit_batch3_soh_curves_01.png)

![mit_batch3 SOH curves](mit_batch3_soh_curves_02.png)

| Battery | 관측 수 | 예측 제외 | 64→16 windows | 마지막 SOH | 상태 |
|---|---:|---:|---:|---:|---|
| b3c0 | 1008 | 0 | 929 | 0.8255 | passed |
| b3c1 | 1062 | 0 | 983 | 0.8272 | passed |
| b3c2 | 1266 | 1266 | 0 | 0.8303 | passed_with_exclusions |
| b3c3 | 1114 | 0 | 1035 | 0.8290 | passed |
| b3c4 | 1047 | 0 | 968 | 0.8211 | passed |
| b3c5 | 827 | 0 | 748 | 0.8387 | passed |
| b3c6 | 666 | 0 | 587 | 0.8291 | passed |
| b3c7 | 1835 | 0 | 1756 | 0.8255 | passed |
| b3c8 | 827 | 0 | 748 | 0.8282 | passed |
| b3c9 | 1038 | 0 | 959 | 0.8238 | passed |
| b3c10 | 1077 | 0 | 998 | 0.8301 | passed |
| b3c11 | 816 | 0 | 737 | 0.8256 | passed |
| b3c12 | 931 | 0 | 852 | 0.8208 | passed |
| b3c13 | 815 | 0 | 736 | 0.8271 | passed |
| b3c14 | 857 | 0 | 778 | 0.8218 | passed |
| b3c15 | 875 | 0 | 796 | 0.8247 | passed |
| b3c16 | 1637 | 0 | 1558 | 0.8299 | passed |
| b3c17 | 1314 | 0 | 1235 | 0.8265 | passed |
| b3c18 | 1145 | 0 | 1066 | 0.8244 | passed |
| b3c19 | 1154 | 0 | 1075 | 0.8220 | passed |
| b3c20 | 812 | 0 | 733 | 0.8278 | passed |
| b3c21 | 771 | 0 | 692 | 0.8242 | passed |
| b3c22 | 1001 | 0 | 922 | 0.8349 | passed |
| b3c23 | 2189 | 2189 | 0 | 0.8798 | passed_with_exclusions |
| b3c24 | 824 | 0 | 745 | 0.8422 | passed |
| b3c25 | 988 | 0 | 909 | 0.8317 | passed |
| b3c26 | 1027 | 0 | 948 | 0.8219 | passed |
| b3c27 | 849 | 0 | 770 | 0.8308 | passed |
| b3c28 | 540 | 0 | 461 | 0.8372 | passed |
| b3c29 | 857 | 0 | 778 | 0.8262 | passed |
| b3c30 | 934 | 0 | 855 | 0.8246 | passed |
| b3c31 | 730 | 0 | 651 | 0.8247 | passed |
| b3c32 | 2237 | 2237 | 0 | 0.9088 | passed_with_exclusions |
| b3c33 | 1283 | 0 | 1204 | 0.8314 | passed |
| b3c34 | 1157 | 0 | 1078 | 0.8302 | passed |
| b3c35 | 1092 | 0 | 1013 | 0.8283 | passed |
| b3c36 | 922 | 0 | 843 | 0.8328 | passed |
| b3c37 | 1389 | 1389 | 0 | 0.8394 | passed_with_exclusions |
| b3c38 | 1934 | 0 | 1855 | 0.8231 | passed |
| b3c39 | 1155 | 0 | 1076 | 0.8414 | passed |
| b3c40 | 795 | 0 | 716 | 0.8323 | passed |
| b3c41 | 785 | 0 | 706 | 0.8393 | passed |
| b3c42 | 1641 | 1641 | 0 | 0.8261 | passed_with_exclusions |
| b3c43 | 1045 | 1045 | 0 | 0.8236 | passed_with_exclusions |
| b3c44 | 939 | 0 | 860 | 0.8239 | passed |
| b3c45 | 1800 | 0 | 1721 | 0.8238 | passed |

전체 공통 데이터: `data/processed/MIT/mit_batch3`.
아직 모델 예측은 실행하지 않았다.

[데이터 제작자의 설명](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_remaining_batteries.py
```

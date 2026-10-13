# mit_batch2 SOH 추출

기록 **48개**, cycle **24,920개**. 필수 확인: **complete**.

용량·cycle 배열, 중복·순서, 정규화 계산을 확인했다. 유효하지 않거나 불완전한 관측은 표시해서 예측 window에서 제외한다. 고유 관측과 원래 capacity는 보존하며 보간·평활화·이상치 자동 제거를 하지 않았다.

Batch 1의 cycle 1은 0인 미측정값으로 보존하고 SOH 기준은 cycle 2 용량으로 명시했다. batch를 이어 측정한 5개 기록은 원래 배터리의 기준 용량과 physical_battery_id를 공유한다. train/test 분리 시 이 ID를 함께 묶어야 한다. 제작자가 noisy channel로 표시한 Batch 3의 6개 기록은 곡선과 데이터에 보존하고 예측 window에서는 제외한다. 기록 140개는 물리 배터리 135개에 해당한다.

![mit_batch2 SOH curves](mit_batch2_soh_curves_01.png)

![mit_batch2 SOH curves](mit_batch2_soh_curves_02.png)

| Battery | 관측 수 | 예측 제외 | 64→16 windows | 마지막 SOH | 상태 |
|---|---:|---:|---:|---:|---|
| b2c0 | 326 | 0 | 247 | 0.7738 | passed |
| b2c1 | 170 | 0 | 91 | 0.7731 | passed |
| b2c2 | 464 | 0 | 385 | 0.7705 | passed |
| b2c3 | 361 | 0 | 282 | 0.7760 | passed |
| b2c4 | 471 | 0 | 392 | 0.7722 | passed |
| b2c5 | 507 | 0 | 428 | 0.7753 | passed |
| b2c6 | 545 | 0 | 466 | 0.7751 | passed |
| b2c7 | 662 | 0 | 583 | 0.8245 | passed |
| b2c8 | 981 | 0 | 902 | 0.8184 | passed |
| b2c9 | 1060 | 0 | 981 | 0.8155 | passed |
| b2c10 | 596 | 0 | 517 | 0.7769 | passed |
| b2c11 | 505 | 0 | 426 | 0.7725 | passed |
| b2c12 | 491 | 0 | 412 | 0.7739 | passed |
| b2c13 | 520 | 0 | 441 | 0.7670 | passed |
| b2c14 | 509 | 0 | 430 | 0.7691 | passed |
| b2c15 | 208 | 0 | 129 | 0.8151 | passed |
| b2c16 | 482 | 0 | 403 | 0.8160 | passed |
| b2c17 | 525 | 0 | 446 | 0.7754 | passed |
| b2c18 | 523 | 0 | 444 | 0.7776 | passed |
| b2c19 | 489 | 0 | 410 | 0.7899 | passed |
| b2c20 | 531 | 0 | 452 | 0.7665 | passed |
| b2c21 | 520 | 0 | 441 | 0.7710 | passed |
| b2c22 | 540 | 0 | 461 | 0.7688 | passed |
| b2c23 | 562 | 0 | 483 | 0.7708 | passed |
| b2c24 | 524 | 0 | 445 | 0.7765 | passed |
| b2c25 | 487 | 0 | 408 | 0.7763 | passed |
| b2c26 | 498 | 0 | 419 | 0.7694 | passed |
| b2c27 | 495 | 0 | 416 | 0.7775 | passed |
| b2c28 | 541 | 0 | 462 | 0.7740 | passed |
| b2c29 | 522 | 0 | 443 | 0.7728 | passed |
| b2c30 | 507 | 0 | 428 | 0.7768 | passed |
| b2c31 | 519 | 0 | 440 | 0.7726 | passed |
| b2c32 | 544 | 0 | 465 | 0.7721 | passed |
| b2c33 | 546 | 0 | 467 | 0.7677 | passed |
| b2c34 | 526 | 0 | 447 | 0.7765 | passed |
| b2c35 | 486 | 0 | 407 | 0.7754 | passed |
| b2c36 | 560 | 0 | 481 | 0.7692 | passed |
| b2c37 | 502 | 0 | 423 | 0.7717 | passed |
| b2c38 | 487 | 0 | 408 | 0.7741 | passed |
| b2c39 | 480 | 0 | 401 | 0.7701 | passed |
| b2c40 | 530 | 0 | 451 | 0.7695 | passed |
| b2c41 | 452 | 0 | 373 | 0.7762 | passed |
| b2c42 | 492 | 0 | 413 | 0.7687 | passed |
| b2c43 | 487 | 0 | 408 | 0.7726 | passed |
| b2c44 | 479 | 0 | 400 | 0.7744 | passed |
| b2c45 | 514 | 0 | 435 | 0.7738 | passed |
| b2c46 | 449 | 0 | 370 | 0.7742 | passed |
| b2c47 | 745 | 0 | 666 | 0.7957 | passed |

전체 공통 데이터: `data/processed/MIT/mit_batch2`.
아직 모델 예측은 실행하지 않았다.

[데이터 제작자의 설명](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_remaining_batteries.py
```

# 02·03 갈무리와 04 비교 기준

이 문서는 2026-09-29 진단을 갈무리한 기록이다. 이후 03에는 EOS loss 제외·validation 유지·학습률 3개·best checkpoint 프로토콜을 적용했다. 현재 실행 방식은 [새 프로토콜](lora_checkpoint_protocol.md), 결과는 [학습률 비교 보고서](../results/checkpoint_lora/report.md)를 참고한다. 아래 수치는 당시 결과로 보존한다.

02는 zero-shot 기준선, 03은 기존 LoRA 결과와 EOS loss 분리 검증까지 정리했다. 04 Full fine-tuning은 다음 단계이며 아직 구현하거나 실행하지 않았다.

## 무엇을 비교했나

`amazon/chronos-t5-base`의 같은 revision을 사용해 NASA 배터리 B0005·B0006·B0007·B0018을 평가했다. SOH는 첫 방전 용량 대비 비율 `Qk/Q1`이다. 매 예측 시점의 실제 관측 64개로 미래 16개를 예측하며, 첫 64개만 알고 수명 전체를 예측하는 실험은 아니다.

LoRA는 q/v에 rank 8, alpha 16, dropout 0.05를 적용했다. 평가 배터리는 학습과 epoch 선택에서 제외하는 LOBO 방식이다. 나머지 세 배터리의 시간순 validation으로 epoch를 고른 뒤, 기존 실험에서는 세 배터리 전체로 같은 epoch만큼 새로 학습했다.

| 결과 | 저장 위치 | 역할 |
|---|---|---|
| Zero-shot | `results/runs/20260923T063348Z_zeroshot_b57856` | 02 기준선 |
| 기존 LoRA | `results/runs/20260923T063628Z_lora_2df6cb` | 출력 EOS 포함, validation 후 full-data refit |
| EOS loss 제외 LoRA | `results/diagnostics/20260929/training/B0005_without_eos`, `confirm_eos/B*_without_eos` | 같은 기존 학습량에서 EOS 항만 바꾼 후속 진단 |

마지막 행의 `confirm_eos/`도 `results/diagnostics/20260929/` 아래에 있다. 가중치는 로컬에 있고 Git에서 제외된다. 원시 예측·지표·진단 보고서는 저장돼 있다.

## 정리된 결과

각 window에서 100 samples를 생성하고 추론 seed 42/43/44의 점수를 평균했다. 배터리 네 개에 같은 가중치를 준 macro 결과이며, LoRA 학습 seed는 42 하나다. 값은 [집계 CSV](../results/diagnostics/20260929/seed_confirmation/macro_summary.csv)에 근거한다.

| 모델 | 중앙값 RMSE ↓ | 평균 예측 RMSE ↓ | CRPS ↓ | 80% 구간 실제 포함률 |
|---|---:|---:|---:|---:|
| Zero-shot | 0.016307 | 0.018540 | 0.008769 | 80.32% |
| 기존 LoRA | 0.016176 | 0.294040 | 0.017684 | 81.20% |
| EOS loss 제외 LoRA | 0.014894 | 0.015011 | 0.007552 | 74.46% |

중앙값/평균은 각각 100개 sample의 중앙값/평균으로 만든 점예측을 뜻한다. CRPS는 예측 분포 전체의 점수다. 실제 포함률은 높을수록 무조건 좋은 지표가 아니며, 명목값 80%와 구간 폭을 함께 봐야 한다.

- **02:** 비교 가능한 zero-shot 기준선을 확보했다.
- **03 기존 LoRA:** 중앙값 RMSE 개선은 약 0.8%지만 예측 분포가 크게 악화됐다. 전체적으로 성공적인 튜닝으로 결론내리지 않는다.
- **03 EOS 진단:** 마지막 출력 EOS loss만 제외하자 중앙값 RMSE는 zero-shot보다 8.66%, CRPS는 13.88% 감소했다. 입력 EOS와 미래값 토큰의 loss 가중치는 유지했다. 원시 예측을 clipping하지 않았다.
- **남은 한계:** B0006 중앙값은 zero-shot보다 0.77% 나쁘다. B0007 중앙값은 기존 LoRA보다 나빠졌고 80% 구간 포함률은 약 65.29%다. EOS 제외 후에도 드문 극단값은 남는다.

## 성능 저하 원인에 대한 판정

| 요인 | 확인한 내용 | 해석 범위 |
|---|---|---|
| 출력 EOS loss | 해당 항만 제외하면 네 fold 모두 평균 예측 RMSE·CRPS가 회복 | 현재 데이터·학습 설정에서의 인과적 영향 |
| 재학습 업데이트 증가 | B0005 같은 학습 경로의 81→135회에서 평균 예측 RMSE 0.041309→0.551199 | 다른 fold의 step 효과까지 검증한 것은 아님 |
| 학습률 | B0005에서 1e-4→3e-5만 바꾸어도 폭주 완화 | 최적 학습률을 확정한 것은 아님 |
| PEFT wrapper / bf16 추론 | B0005 앞 32 windows에서 wrapper 우회는 동일 결과, FP32 추론도 폭주 | FP32 재학습 미검증; cache-off는 XPU 오류로 미완료 |
| 데이터 부족 / LoRA rank | 이번 실험으로 원인을 확정할 수 없음 | Full FT 결과만으로도 단독 원인 확정은 어려움 |

상세한 고정 조건, 학습 loss 분해, 통제 실험과 실패 기록은 [진단 보고서](../results/diagnostics/20260929/diagnosis.md)에 있다. 이번 결과는 이미 관찰한 네 배터리에 대한 사후 진단이다. 추론 seed 세 개와 겹치는 window를 독립적인 데이터 반복으로 취급하지 않는다.

## 노트북을 읽거나 다시 실행할 때

- [02](../notebooks/02_chronos_zeroshot.ipynb)의 마지막 절에 고정 zero-shot 지표가 있다.
- [03](../notebooks/03_chronos_lora_lobo.ipynb)의 앞부분은 기존 EOS 포함 + refit 실험, 마지막 절은 저장된 EOS 제외 진단이다. 마지막 절은 CSV/PNG를 읽으며 새 학습을 하지 않는다.
- `FORCE_RERUN=True` 또는 `scripts/run_experiment.py lora --force`는 **기존 LoRA 프로토콜**을 재실행한다. EOS 제외 실험으로 바뀌지 않는다. 후속 학습 명령은 진단 보고서에 있다.
- 노트북 마지막 절은 2026-09-29의 고정 결과다. 앞부분에서 설정을 바꾸어 새 run을 만들면 둘은 다른 실험이 된다.
- `scripts/build_notebooks.py`가 노트북의 원본 템플릿이다. 동일한 셀의 기존 출력은 유지하며, 내용이 바뀐 코드 셀은 출력을 비운다.
- `cuda-fullft`의 CUDA·KL·검증 정책 실험은 별도 브랜치의 다른 프로토콜이다. 그 결과와 여기의 개선율을 직접 비교하지 않는다.

## 04에서 확인할 질문과 비교 조건

**질문: 같은 조건에서 전체 가중치를 학습하면 EOS 제외 LoRA보다 추가 이득이 있는가?** 성능 향상을 전제하지 않고 정확도·분포·계산 비용을 함께 비교한다.

1. 모델 ID/revision, SOH 정의, outer LOBO, context 64/horizon 16, 평가 window, 장치와 추론 설정을 맞춘다. 모델 크기를 바꾸면 별도 실험으로 기록한다.
2. Zero-shot / LoRA / Full FT를 비교한다. 학습 모델은 동일한 출력 EOS loss 정책과 시간순 내부 검증을 사용한다. 미래값 토큰의 loss 가중치도 맞춘다.
3. 학습률은 LoRA와 Full FT 각각 훈련 배터리 내부 검증으로 선택한다. checkpoint 선택 기준과 탐색 범위를 test 평가 전에 고정한다. 검증한 checkpoint를 그대로 평가하고 별도 full-data refit은 기본 비교에 넣지 않는다.
4. 현재 EOS 진단은 기존 epoch를 유지한 full-data 학습이다. 위 프로토콜과 다르므로 **04의 LoRA 대조군은 같은 검증·학습 방식으로 다시 평가**한다. 이 문서의 0.014894를 새 Full FT와 바로 비교하지 않는다.
5. 중앙값/평균 예측 RMSE, CRPS, 80% 구간 포함률·폭, 이상 sample 비율, 학습 시간·최대 메모리를 남긴다. 학습 seed 반복과 추론 seed 반복을 구분한다.
6. 첫 fold는 메모리와 실행 안정성을 확인한다. 실행 가능하면 그 fold의 성능이 좋든 나쁘든 네 fold를 모두 진행한다. 같은 네 배터리를 재사용한 비교이므로 독립적인 새 벤치마크라고 부르지 않는다.

04의 코드·학습률 후보·실행 예산은 다음 작업에서 구체화한다. 이번 갈무리에서는 기존 학습 소스와 설정을 변경하지 않았다.

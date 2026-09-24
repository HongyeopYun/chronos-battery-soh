# 02 / 03 노트북 검토

검토일: 2026-09-23. 대상은 현재 작업 디렉터리의 수정 중인 노트북이다. 기존 노트북, 예측 파일, CSV는 변경하지 않았다. 셀 번호는 위에서부터 1부터 센 번호이며 Jupyter 실행 번호와 다르다.

## 판단과 검증 범위

**Zero-shot과 LoRA LOBO의 핵심 구현은 정상이다.** 확인한 코드에서는 평가 배터리의 미래 정답을 추론 입력이나 해당 fold의 학습 데이터에 넣는 누수가 발견되지 않았다. 다만 재현성, 결과 저장, 검증 세트 설계는 보완해야 한다. 현재 결과는 탐색 실험으로 해석하는 것이 적절하다.

실제로 수행한 확인:

- 두 노트북의 모든 코드 셀 구문 검사. 저장된 오류 출력 없음.
- 원본 MAT에서 방전 용량과 SOH를 재구성하고 64-context / 16-horizon 윈도우 재생성.
- 저장된 zero-shot NPZ의 크기·유한값 확인 및 RMSE/MAE 재계산. Last-value, Linear 기준 모델도 재계산. CSV의 해당 수치와 모두 소수점 6자리 반올림 정밀도 내 일치.
- 로컬 캐시 모델과 실제 Intel Arc B390 XPU에서 2개 윈도우로 zero-shot 추론, LoRA 학습 1 step, LoRA 추론 실행. 양쪽 추론 shape `(2, 20, 16)`, 유한값 확인.
- 학습용·추론용 tokenizer의 context token, mask, scale이 동일함을 확인.
- 학습 가능한 파라미터 884,736개가 모두 LoRA 파라미터이고 float32임을 확인. gradient가 유한하고, LoRA B 가중치가 갱신되며, frozen 파라미터의 gradient가 없음을 확인. 별도로 비교한 frozen 파라미터도 불변.

**전체 4-fold × 10-epoch 재학습은 수행하지 않았다.** 소규모 실행은 현재 환경의 실행 경로 확인이며, 기존 LoRA 점수의 재현 실험이 아니다. 기존 LoRA 예측 배열과 adapter checkpoint가 디스크에 없어 해당 점수를 원시 예측에서 독립적으로 재계산할 수 없다.

## 결과

| 배터리 | Last-value RMSE | Linear RMSE | Zero-shot RMSE | LoRA RMSE |
|---|---:|---:|---:|---:|
| B0005 | 0.022004 | 0.019174 | 0.013641 | 0.011835 |
| B0006 | 0.024485 | 0.026631 | 0.019418 | 0.019555 |
| B0007 | 0.017806 | 0.016162 | 0.012504 | 0.009840 |
| B0018 | 0.022700 | 0.030008 | 0.023311 | 0.021315 |
| 배터리별 평균 | 0.021749 | 0.022994 | 0.017219 | 0.015636 |

LoRA 열은 저장된 셀 출력 및 CSV의 보고값이다. 나머지 모델은 원시 데이터와 저장된 예측으로 검산했다. 보고값 기준 LoRA의 macro RMSE 개선은 약 9.19%, macro MAE 개선은 약 12.11%다. B0006의 RMSE는 약 0.71% 악화되었고 MAE는 개선되었다. 따라서 모든 배터리와 모든 지표가 개선되었다고 해석하면 안 된다.

## 정상적으로 구현된 부분

1. **데이터와 정규화:** MAT의 discharge 용량만 사용한다. B0005/6/7은 각 168개, B0018은 132개이며 유한값이다. `Qk / Q1`은 첫 관측 용량 기준 SOH로 일관적이며 미래 전체 통계로 정규화하지 않는다. 단, 정격 용량 기준 SOH와 정의가 다르므로 보고서에 명시해야 한다.
2. **Zero-shot:** 사전학습 `amazon/chronos-t5-base`를 추가 학습 없이 사용한다. 64개 관측만 입력하고 이후 16개를 예측하며 실제 미래와 비교한다. 89/89/89/53개 평가 윈도우가 맞다.
3. **LOBO:** 평가 배터리를 제외한 3개 배터리의 윈도우만 학습한다. 매 fold에서 사전학습 모델을 새로 불러온다. B0005 초기 1-epoch 시연 뒤에도 새 모델을 생성하므로 현재 코드 순서상 최종 10-epoch에 시연 학습이 누적되지 않는다.
4. **학습 label:** context로 계산한 scale을 future label에도 전달한다. EOS 추가와 mask의 `-100` 처리가 맞다. T5의 `labels=`에 의한 teacher forcing과 CE 학습은 정상적인 절차다. [Chronos 공식 학습 코드](https://github.com/amazon-science/chronos-forecasting/blob/main/scripts/training/train.py)
5. **LoRA 연결:** q/v에 `r=8`, alpha=16을 적용하고 학습 가능한 파라미터만 AdamW로 갱신한다. 평가 전 `eval()` 및 `pipeline.model.model = lora_model`을 수행하여 학습된 adapter가 추론에 연결된다.
6. **64/16 설정:** 사전학습 기본 설정 512/64보다 짧은 context/horizon 사용은 오류가 아니다. 학습 tokenizer의 길이만 64/16으로 조정해도 이 입력에서 추론 tokenizer와 동일한 토큰 및 scale을 만든다.
7. **장치:** CUDA가 False여도 실제 모델은 `xpu:0`에서 실행된다. 이번 소규모 실행에서도 XPU 동작을 확인했다.

## 수정 우선순위

### 1. 비교표의 하드코딩 제거 및 LoRA 산출물 저장 — 높음

- 위치: 03의 33~34번째 셀, `comparison = pd.DataFrame(...)`와 CSV 저장. 02의 17번째 셀도 B0005 metric을 상수로 입력한다.
- 현재 CSV는 `lora_results`에서 생성되지 않는다. 재학습 결과가 달라져도 예전 수치를 계속 기록할 수 있다. 현재 값이 틀렸다는 뜻은 아니며, 저장된 출력과 표는 일치한다.
- 모든 모델의 최종 예측과 동일한 `y_true`에서 metric을 자동 계산하고, 그 결과로 표와 CSV를 만든다.
- fold별 LoRA adapter, 예측 NPZ, target 또는 target 재구성 정보, epoch loss, train/test 배터리, seed, 모델 revision, 실행 설정을 저장한다. 저장 위치는 기존 결과와 구분되는 실행별 디렉터리가 적절하다.

### 2. 시드를 모델 생성 전에 고정하고 fold 실행을 통일 — 높음

- 위치: 03의 15~17번째 셀, 23~26번째 셀, 31번째 셀의 `run_lobo_fold()`.
- `create_fresh_lora_model()` 이후에 `torch.manual_seed(SEED)`를 호출한다. B0006은 학습 직전 전역 seed 재설정도 없다.
- LoRA A는 무작위로 초기화되므로 생성 뒤의 seed 설정은 초기 가중치를 고정하지 못한다. DataLoader의 generator seed만으로는 충분하지 않다. [PEFT 초기화 문서](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora#initialization)
- 각 fold 시작 시 seed를 먼저 설정한 다음 모델·adapter·DataLoader를 생성한다. B0005/6 별도 학습 코드와 B0007/18 함수 실행을 하나의 함수로 통일한다.
- 설치된 PyTorch의 `torch.manual_seed()`는 XPU에도 seed를 설정한다. 문제의 핵심은 XPU seed 호출 누락이 아니라 **호출 시점과 셀 실행 상태**다.

### 3. 학습용 배터리 안에서 validation 구성 — 연구 결론 강화에 중요

- 현재 10 epoch 동안 train CE만 추적하고 마지막 epoch를 평가한다. CE 감소는 학습이 진행된 증거지만 일반화 성능이나 최적 epoch의 증거는 아니다.
- 10 epoch를 평가 전에 정한 고정 프로토콜이면 validation 부재 자체가 누수는 아니다. 평가 배터리 결과를 보면서 epoch/rank/lr를 선택했다면 그 배터리는 모델 선택에 사용된 것이다.
- outer LOBO의 평가 배터리는 최종 평가에만 사용한다. 나머지 3개 안에서 배터리 단위 inner validation으로 epoch/rank/lr를 고른 뒤 3개 전체로 재학습한다. 비용을 줄이는 설계라면 배터리별 시간 블록 분할을 사용하되 검증 target이 train context/target에 들어가지 않도록 경계를 먼저 정한다.
- 겹치는 윈도우를 무작위로 train/validation 분리하는 방식은 피한다.

### 4. 샘플 집계 방식 명시 및 반복 평가 — 중간

- 위치: 02와 03의 `.median(dim=1).values`, `NUM_SAMPLES = 20`.
- PyTorch는 짝수 샘플에서 두 중앙값 중 작은 값을 반환한다. 20개에서는 정렬된 10번째 값이며 두 중앙값 평균과 다르다. 유효한 중앙값 정의이지만 일반적인 보간 중앙값보다 낮거나 같다. [PyTorch median 문서](https://docs.pytorch.org/docs/2.12/generated/torch.median.html)
- 일반적인 보간 중앙값을 원하면 `torch.quantile(forecast.float(), 0.5, dim=1)`을 사용한다. RMSE용 sample mean도 함께 비교할 수 있다. 두 모델에 동일한 집계 규칙을 적용해야 한다.
- 20 samples의 Monte Carlo 변동을 줄이려면 예를 들어 100 samples와 복수 추론 seed를 검토한다. LoRA 학습 seed도 3~5개 반복하여 평균·표준편차를 보고하면 초기화 변동과 추론 변동을 구분할 수 있다.
- 02의 나머지 배터리 추론이 18번째와 19번째 셀에서 반복된다. 한 번 생성한 예측으로 metric·plot·저장을 처리하면 계산 낭비와 실행 결과 혼용 가능성을 줄인다.

### 5. 실행 경로 및 환경 기록 — 중간

- 두 노트북의 `PROJECT_ROOT = Path("..").resolve()`는 CWD가 `notebooks`일 때 맞다. 프로젝트 루트에서 실행하면 상위 `Lab`을 가리켜 데이터 로딩에 실패한다. 실제 두 경로를 확인했다.
- CWD와 그 부모들 중 `data/raw` 및 `notebooks`가 있는 디렉터리를 찾아 루트를 정하고, 결과 저장 디렉터리도 생성하도록 한다.
- requirements.txt에는 torch, chronos-forecasting, transformers, peft, accelerate가 없다. 현재 환경을 새로 만들 수 있도록 버전과 XPU 설치 방법을 기록한다.
- 검사 환경: Python 3.12.10, torch 2.14.0+xpu, chronos-forecasting 2.3.2, transformers 5.17.0, peft 0.21.0, accelerate 1.15.0. 로컬 모델 캐시 snapshot은 `ad294eaacead15db499b740ea4122266dd2a81a2`다.
- `torch_dtype` 경고가 있는 호출은 현재 환경에서 권장되는 `dtype`으로 통일할 수 있다. 기존 실행 실패 원인은 아니다.

## 평가 해석과 추가 개선

- **Rolling-origin 평가임을 명시:** 매 예측 시점마다 실제 최근 64개 SOH를 새로 제공한다. 첫 64개만 알고 배터리 수명 끝까지 연속 예측한 실험은 아니다. 이 방식 자체는 정상이며 장기 열화/RUL 예측은 별도 프로토콜이 필요하다.
- **중첩 윈도우는 독립 표본이 아니다:** 한 실제 target cycle이 최대 16개 윈도우에 포함된다. 현재 평균은 예측 시작점과 horizon 조합을 평가한다. 320개 윈도우를 독립 배터리 표본처럼 취급해 신뢰구간이나 유의성을 계산하면 안 된다. 실험 대상은 4개 배터리이고 fold별 학습 데이터도 중첩된다.
- **전체 RMSE와 window별 RMSE 평균 구분:** `sqrt(mean(error²))`와 `mean(sqrt(mean(error², axis=1)))`는 다르다. B0006에서 window별 평균 개선이 양수인데 전체 RMSE가 나빠진 것은 모순이 아니다. 일부 큰 오차의 제곱 영향이 원인일 수 있다.
- **horizon별 평가 확대:** 02는 B0005의 horizon별 오차를 계산하지만, 양 모델과 전 배터리에서 h=1~16 오차를 비교하면 단기/장기 개선 여부가 드러난다. 평균뿐 아니라 최악 구간과 예측구간 coverage도 살펴볼 수 있다.
- **계단형 예측:** T5 Chronos의 이산 토큰 bin 때문에 발생한다. 현재 context scale에서 평균 SOH bin 간격은 약 0.0056~0.0065, 즉 0.56~0.65%p다. 소수점 표시나 그래프 코드 문제가 아니다. [Chronos tokenizer 구현](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos.py)
- **양자화가 모든 오차를 설명하지는 않는다:** 03의 quantization oracle RMSE 출력은 약 0.0016~0.0019로 실제 예측 RMSE보다 훨씬 작다. 이 oracle은 실제 미래를 가장 가까운 bin에 투영한 진단값이며 실현 가능한 예측 성능이 아니다. bin 값만 반환하는 현재 집계의 제약을 보여주며, sample mean이나 보간 quantile까지 동일한 하한으로 해석하면 안 된다.
- **SOH 정의:** 현재 `Qk / Q1`을 계속 쓰는 것은 일관적이다. 정격 용량 기준 SOH와 비교하거나 EOL threshold를 적용할 때는 기준 용량 차이를 반영해야 한다.
- **데이터 규모와 용도:** 3개 배터리에서 만든 수백 개 중첩 윈도우는 수백 개 독립 배터리 데이터가 아니다. 복수 seed 평가와 추가 배터리/실험 조건 검증이 필요하다. 관측 SOH가 이미 있는 상태에서 미래 SOH를 예측한 실험이며 전압·전류로 현재 SOH를 추정한 실험과 구분한다.
- **작은 집계 버그 가능성:** 03의 39번째 셀은 `delta > 0`, `delta < 0`, `isclose(delta, 0)`를 각각 세므로 거의 0인 값이 improved/worsened와 tied에 중복 집계될 수 있다. 현재 저장 출력의 tied는 0이라 실제 수치에는 영향이 없었다. 허용오차를 먼저 정해 세 범주를 배타적으로 만든다.

우선 하드코딩·저장·seed·실행 함수 통일을 해결하고, validation 및 반복 평가를 추가하는 순서가 적절하다. LoRA rank나 epoch를 무조건 늘리는 것보다 현재 개선이 재현되는지 먼저 확인하는 편이 실험 해석에 도움이 된다.

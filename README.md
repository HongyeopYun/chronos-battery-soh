# Battery SOH / RUL — Chronos-2

**시작:** [전체 결과 탐색](results/index.html) · [폴더 구조](docs/workspace_layout.md) · [실행 파일 안내](scripts/README.md) · [공통 코드 안내](src/README.md) · [발표 자료](presentations/README.md)

최신 NASA 직접 RUL few-shot 개선은 [개선 전후 그래프와 검증](results/c2-battery/rul-nasa-fewshot-report-v1/index.html)에 있다. 동결한 Chronos 백본과 기존 보조 특징 3개를 사용하는 별도 RUL 회귀 Head의 실험이다.

최신 실험은 **현재 SOH 한 점 예측**, **미래 SOH 경로 기반 RUL zero-shot**, **직접 RUL Head-only·Full·LoRA**다.
전체 비교는 [v4 GPU 통합 보고서](results/c2-battery/v4-gpu-suite-v2/report.md), 후속 데이터 양 진단은 [RUL 학습곡선 보고서](results/c2-battery/rul-learning-curve-v1/report.md)에 있다. SSH `home-pc`의 RTX 3080에서 실행했다.
기준 데이터는 `data/processed/reviewed_v4/forecast_safe/`이며 NASA·CALCE·MIT 3개 batch·XJTU 2C/3C의 7개 그룹을 사용한다. [전처리 변경과 검증](results/preprocessing_batterylife_v4/report.md)을 먼저 확인한다. 과거 `reviewed_native_v3` 결과는 그대로 보존한다.
예전 T5·LoRA·strict-v2 모델 실험은 [archive/](archive/README.md)에 별도 보관했다.

## 결과 바로 보기

| 실험 | 결과 | 상태 |
|---|---|---|
| RUL 학습 셀 수·정답 수 확대, Head-only | [후속 보고서](results/c2-battery/rul-learning-curve-v1/report.md) · [정답 수 비교](results/c2-battery/rul-learning-curve-v1/label_learning_curves.png) · [MIT 셀 수 비교](results/c2-battery/rul-learning-curve-v1/mit_cell_learning_curves.png) | 완료: CUDA, 144개 Head, 3seed, 평균·중앙값·Ridge 기준 |
| v4 현재 SOH 2·4·8·16·32·64 / 궤적 RUL / 직접 RUL 3방식 | [통합 보고서](results/c2-battery/v4-gpu-suite-v2/report.md) · [SOH 평균 감소율](results/c2-battery/v4-gpu-suite-v2/soh_input_reduction.png) · [직접 RUL 비교](results/c2-battery/v4-gpu-suite-v2/rul_direct_representatives.png) | 완료: CUDA, 117개 모델 선택·저장·검증 |
| BatteryML / BatteryLife 규칙 반영 전처리 v4 | [정제 보고서](results/preprocessing_batterylife_v4/report.md) · [적격 관측 그래프](results/preprocessing_batterylife_v4/eligible_trajectories.png) | 완료: 예측용·사후 정제용 분리 |
| 현재 SOH 입력 2·4·8·16·32·64 → 1 | [통합 비교](results/c2-battery/one-step-2-reviewed-v3/report.md) · [입력별 평균 MSE 감소율](results/c2-battery/one-step-2-reviewed-v3/input_mean_mse_reduction.png) | 완료 |
| v3 RUL SOH 80% 기준 재평가 | [과거 예측 재채점](results/c2-battery/rul-nasa-soh80-rescore-v1/report.md) · [예측 경로](results/c2-battery/rul-nasa-soh80-rescore-v1/trajectories.png) | 완료: 기존 예측 재채점 |
| RUL zero-shot, 입력 64 → 미래 256 | [첫 파일럿](results/c2-battery/rul-nasa-l64-h256-zero-v1/report.md) | 완료 |
| RUL head-only | [Zero-shot 대비](results/c2-battery/rul-nasa-l64-h256-head-v1/report.md) | 완료 |
| RUL 입력 16·32·64 비교 | [동일 CUDA 비교](results/c2-battery/rul-nasa-context-sweep-cuda-v1/report.md) | 완료 |
| RUL 학습 시작 시점 확대 | [t64 vs t64·80·96](results/c2-battery/rul-nasa-multi-origin-xpu-v1/report.md) · [예측 경로](results/c2-battery/rul-nasa-multi-origin-xpu-v1/trajectories.png) | 완료 |

v4 결과·원시 예측·프로토콜·학습 이력·선택 가중치는 `results/c2-battery/v4-gpu-suite-v2/`에 있다. 실행 코드 사본은 `source_snapshot.zip`에 보존했다.
과거 현재 SOH 결과는 `results/c2-battery/one-step-{L}-reviewed-v3/`에 있다.
v4 RUL은 SOH 80% 기준의 신규 학습·예측이다. `rul-nasa-soh80-rescore-v1`은 과거 예측의 재채점이며, 그 이전 RUL 보고서는 당시의 1.4Ah 기준을 보존한다.

## 실험을 해석하는 기준

- **현재 SOH:** 최근 실측 L개로 바로 다음 한 점을 예측한다. 평가 시점마다 실제 관측으로 입력을 갱신한다.
- **현재 SOH R²:** 각 배터리의 여러 한 점 예측을 모아 temporal R²를 계산한다. 한 점짜리 창 각각의 R²가 아니다.
- **MSE 감소율:** 배터리별 `1 − MSE_model / MSE_LastValue`를 계산한 뒤 평균한다. 통합 그림은 7개 데이터셋 평균에 같은 가중치를 준다.
- **v4 입력 길이 비교:** 145개 시험 셀, 길이마다 94,516개 공통 평가 시점이다. 6개 길이 모두 같은 CUDA 환경이다. MIT 물리 셀 병합 후 분할도 새로 고정했으므로 v3 점수와의 차이를 전처리만의 개선 효과로 해석하지 않는다.
- **v4 궤적 RUL:** 최근 32개 기록 위치를 입력하고 t=32부터 16개 위치 간격으로 미래 SOH 1,024개를 예측한다. 셀별 Q_ref로 정규화한 SOH ≤ 0.8의 최초 적격 관측을 EOL로 정한다. 미래 실측을 입력하지 않는다. 미도달(`>1024`)과 조건부 MAE·전체 MAE 하한을 함께 보고한다.
- **v4 직접 RUL:** 동일한 8개 학습 RUL 정답·4개 검증 정답으로 Head-only / Full / LoRA를 비교한다. 새 RUL 회귀 head를 사용하며 순정 C2의 직접 RUL zero-shot으로 부르지 않는다. 입력에는 SOH 이력만 사용한다.
- **v4 RUL 평가 범위:** 사용 가능 160셀 중 57셀에서 SOH 80% 도달을 관측했다. 직접 모델 시험은 157셀, 방법당 6,816개 원점이다. MIT Batch 1·3은 MIT Batch 2 모델, XJTU 3C는 XJTU 2C 모델을 적용한 전이 예측으로 별도 표시한다. 미도달 셀의 기록 종료를 EOL로 대체하지 않는다.
- **과거 RUL과 비교:** 기존 NASA 실험은 t=64·H=256 또는 1.4Ah 기준이다. v4는 데이터·시험 시점·horizon·head 목적이 바뀌었으므로 숫자 차이를 단일 요소의 개선으로 해석하지 않는다.
- **학습 분리:** 학습·검증·평가 배터리는 분리한다. 평가 셀의 미래로 모델이나 체크포인트를 선택하지 않는다.
- **RUL 후속 학습곡선:** 셀당 정답4·16·전체, MIT 학습 셀2·4·8·16을 비교한다. MIT는 ID와 고정 RNG로 시험20개를 새로 고정한 탐색이며, 기존 시험40개 점수와 직접 비교하지 않는다. 다른 세 그룹은 기존 LOBO를 유지한다. 모든 조건의 최적화·검증 예산을 맞추고, 같은 SOH32 입력의 Ridge와 비교한다. 전체 정답 조건은 few-shot이 아니다.
- **그림:** 제외된 관측점은 표시하지 않고 원래 cycle 위치와 결측 구간을 유지한다. 평활화·보간·단조화는 하지 않는다.
- **v4 전처리:** CALCE 용량 적분·MIT 연속 셀 병합과 지정 예외·XJTU 측정 조건을 반영했다. `forecast_safe`를 사용했다. CALCE 양방향 중앙값과 MIT 미도달 셀 제외는 미래 정보에 의존하므로 `retrospective`에서만 적용한다. `src.preprocessing_v4.load_v4_bundle`로 읽고 물리 셀 단위 새 분할을 고정했다.

## 작업 폴더

| 경로 | 역할 |
|---|---|
| `presentations/published/` | 발표 완성본 |
| `artifacts/` | 발표 빌드·원격 전송 묶음·결과 ZIP·정리 기록 |
| `archive/` | 과거 실험·모델·정리 전 복구본 |
| `scripts/` | 최신 C2 실행·보고서·필요한 데이터 전처리 |
| `src/` | 모델 학습·평가·정제 공통 함수 |
| `tests/` | 현재 코드와 데이터 처리 테스트 |
| `configs/` | C2 및 공통 함수가 사용하는 설정 |
| `data/raw/` | 원본 데이터. 정리 대상에서 제외 |
| `data/processed/` | 최신 v4·기존 v3와 중간 데이터. 정리 대상에서 제외 |
| `results/c2-battery/` | 최신 SOH·RUL 결과 |
| `results/preprocessing_*/`, `results/data_audit/` | NASA 복구·다른 데이터셋 정제 근거 |
| `models/` | 공통 Chronos-2 캐시와 최신 실험 가중치 |

일부 공통 모듈에는 예전 T5 이름이 남아 있지만 최신 C2 코드가 직접 사용하므로 보존했다.
`results/c2-battery/strict-v2-zero-head/protocol.json`은 고정 배터리 분할,
`results/c2-battery/one-step-64/coverage.csv`는 전처리 전후 비교를 위해 남긴 최소 참조 파일이다.
이 두 폴더의 과거 모델 결과는 별도 보관했다.

최신 직접 RUL 입력 길이 비교: [전체 셀 갤러리](results/c2-battery/rul-context-nominal80-v1/index.html) · [성능표](results/c2-battery/rul-context-nominal80-v1/summary.csv) · [검증](results/c2-battery/rul-context-nominal80-v1/verification.json). MIT는 공식 정격80% EOL을 적용했으며 기존 초기용량80% 결과와 별도 보관한다.

## 실행 환경과 진입점

기존 `.venv`를 사용한다. 환경과 설치 패키지는 이번 정리에서 변경하지 않았다.
새 환경에서는 장치에 맞는 PyTorch를 설치한 후 `requirements.txt`를 설치한다.

| 작업 | 진입점 |
|---|---|
| 직접 RUL context / MIT 정격80 | `scripts/run_c2_rul_context_nominal.py --out <새 결과 폴더> --cache-root <캐시>` — CUDA, 201 Head, NASA L8/16/32 및 나머지 L8/16/32/64/128, 동일 학습·평가 시점 |
| 직접 RUL context 검증·그래프 | `scripts/report_c2_rul_context_nominal.py --out <결과 폴더>` — 공식 MIT 라벨, 셀 분리, 저장 가중치, 동일 시점 및 기준 점수 검증 후 전체 셀 갤러리 |
| RUL 학습곡선 실행 | `scripts/run_c2_rul_learning_curve.py --out <새 결과 폴더> --cache-root <캐시>` — CUDA, 144개 Head, `--prepare-only` 지원 |
| RUL 학습곡선 검증·보고서 | `scripts/report_c2_rul_learning_curve.py --out <결과 폴더> --gpu-verify` — 저장 Head 재로딩 및 native 예측 표본 검증, 셀 분리·해시·기준 점수 재검산 |
| RUL 학습곡선 내보내기 | `scripts/export_c2_rul_learning_curve.py --out <검증된 결과 폴더> --archive <ZIP 경로>` — 144개 가중치·예측·소스 스냅샷·파일 해시 포함 |
| v4 GPU 전체 실험 | `scripts/run_c2_v4_suite.py --out <새 출력 폴더> --cache-root <캐시>` — CUDA 필수, `--smoke-only` 또는 `--prepare-only` 지원 |
| v4 점수 독립 검증·보고서 | `scripts/verify_report_c2_v4_suite.py --out <결과 폴더>` — `--read-only`는 기존 결과를 보존하고 별도 로컬 검증 기록 생성 |
| v4 검증 결과 내보내기 | `scripts/export_c2_v4_suite.py --out <검증된 결과 폴더>` — 결과와 가중치를 별도 ZIP으로 보존 |
| SOH 64 → 1 | `scripts/run_c2_one_step_context.py --reviewed` |
| SOH 나머지 입력 길이 | `scripts/run_c2_context32.py`, `run_c2_context16.py`, `run_c2_context8.py`, `run_c2_context4.py`, `run_c2_context2.py` |
| SOH 그림만 갱신 | `scripts/render_c2_one_step_predictions.py --out results/c2-battery/one-step-2-reviewed-v3` |
| RUL zero-shot / head | `scripts/run_c2_rul_pilot.py`, `scripts/run_c2_rul_head.py` |
| RUL 입력 길이 비교 | `scripts/run_c2_rul_context_sweep.py` — 데스크톱 CUDA 조건 |
| RUL 학습 시점 비교 | `scripts/run_c2_rul_multi_origin.py` — XPU 조건 |
| RUL SOH 80% 재평가 | `scripts/rescore_c2_rul_soh80.py` — 기존 4개 실행의 저장 예측 사용; 완료된 출력 폴더가 있으면 덮어쓰지 않고 중단 |
| v3 데이터 구축 | `scripts/build_reviewed_soh.py` |
| v4 데이터 구축 | `scripts/build_preprocessing_v4.py --output <새 출력 경로> --report <새 보고서 경로>` — 완료된 결과 덮어쓰기 금지 |
| 저장된 v4 독립 검증 | `scripts/verify_preprocessing_v4.py` |

완료된 결과의 입력·가중치·실행 조건은 고정되어 있다. 새로운 실험은 별도 결과 경로를 사용한다.
과거 v3의 CUDA 16·8·4·2 및 RUL 입력 길이 실험은 기존 데스크톱 실행 경로와 조건을 유지한다.
코드를 복사한 실험 패키지와 본 프로젝트는 구분한다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
```

결과 폴더에는 점수·원시 예측·학습 조건·최소 검증 기록을 함께 남겼다.
과거 결과의 소스 해시가 가리키는 실행 파일은 이번 정리에서 수정하지 않았다.

## 이전 실험 보관

2026-10-13 정리 전 코드·문서·결과·가공 데이터는 당시 프로젝트 밖에 복사해 검증했다. 2026-10-15에 해당 보관본 전체를 프로젝트의 `archive/cleanup-20261013/`로 이동했다.
과거 실행 스크립트·노트북·결과·모델은 영구 삭제하지 않고 별도 폴더로 옮겼다.

[별도 보관 및 복구 안내](archive/README.md)

이 보관 경로는 이 PC의 로컬 경로이며 Git에는 포함되지 않는다.

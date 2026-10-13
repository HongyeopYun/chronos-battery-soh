# Battery SOH / RUL — Chronos-2

최신 실험은 **현재 SOH 한 점 예측**과 **미래 SOH 경로 기반 RUL**이다.
기준 데이터는 `data/processed/reviewed_native_v3/`이며 NASA·CALCE·MIT 3개 batch·XJTU 2C/3C의 7개 그룹을 사용한다.
예전 T5·LoRA·strict-v2 모델 실험은 프로젝트 밖에 별도 보관했다.

## 결과 바로 보기

| 실험 | 결과 | 상태 |
|---|---|---|
| 현재 SOH 입력 2·4·8·16·32·64 → 1 | [통합 비교](results/c2-battery/one-step-2-reviewed-v3/report.md) · [입력별 평균 MSE 감소율](results/c2-battery/one-step-2-reviewed-v3/input_mean_mse_reduction.png) | 완료 |
| RUL zero-shot, 입력 64 → 미래 256 | [첫 파일럿](results/c2-battery/rul-nasa-l64-h256-zero-v1/report.md) | 완료 |
| RUL head-only | [Zero-shot 대비](results/c2-battery/rul-nasa-l64-h256-head-v1/report.md) | 완료 |
| RUL 입력 16·32·64 비교 | [동일 CUDA 비교](results/c2-battery/rul-nasa-context-sweep-cuda-v1/report.md) | 완료 |
| RUL 학습 시작 시점 확대 | [t64 vs t64·80·96](results/c2-battery/rul-nasa-multi-origin-xpu-v1/report.md) · [예측 경로](results/c2-battery/rul-nasa-multi-origin-xpu-v1/trajectories.png) | 완료 |

현재 SOH의 개별 결과는 `results/c2-battery/one-step-{L}-reviewed-v3/`에 있다.
RUL 내부 가중치/LoRA 미세조정은 다음 후보이며 **아직 실행하지 않았다**.

## 실험을 해석하는 기준

- **현재 SOH:** 최근 실측 L개로 바로 다음 한 점을 예측한다. 평가 시점마다 실제 관측으로 입력을 갱신한다.
- **현재 SOH R²:** 각 배터리의 여러 한 점 예측을 모아 temporal R²를 계산한다. 한 점짜리 창 각각의 R²가 아니다.
- **MSE 감소율:** 배터리별 `1 − MSE_model / MSE_LastValue`를 계산한 뒤 평균한다. 통합 그림은 7개 데이터셋 평균에 같은 가중치를 준다.
- **입력 길이 비교:** 2개 입력이 모두 결측인 CALCE 3점을 제외한 97,865개 공통 평가 시점을 사용한다. 64·32는 XPU, 16·8·4·2는 데스크톱 CUDA 결과이므로 장치 차이가 있다.
- **RUL:** t=64에서 미래 SOH 256개를 예측한 뒤, 용량 1.4Ah 이하에 최초 도달하는 시점으로 계산한다. 예측 도중 미래 실측을 넣지 않는다.
- **RUL 평가:** NASA 4셀 중 실제 EOL 관측은 3셀이다. B0007은 우측 검열이고, 예측 미도달은 `>256`으로 남긴다. 미도달 수와 조건부 MAE를 함께 읽는다.
- **학습 분리:** 학습·검증·평가 배터리는 분리한다. 평가 셀의 미래로 모델이나 체크포인트를 선택하지 않는다.
- **그림:** 제외된 관측점은 표시하지 않고 원래 cycle 위치와 결측 구간을 유지한다. 평활화·보간·단조화는 하지 않는다.

## 작업 폴더

| 경로 | 역할 |
|---|---|
| `scripts/` | 최신 C2 실행·보고서·필요한 데이터 전처리 |
| `src/` | 모델 학습·평가·정제 공통 함수 |
| `tests/` | 현재 코드와 데이터 처리 테스트 |
| `configs/` | C2 및 공통 함수가 사용하는 설정 |
| `data/raw/` | 원본 데이터. 정리 대상에서 제외 |
| `data/processed/` | 현재 v3와 이를 만드는 중간 데이터. 정리 대상에서 제외 |
| `results/c2-battery/` | 최신 SOH·RUL 결과 |
| `results/preprocessing_*/`, `results/data_audit/` | NASA 복구·다른 데이터셋 정제 근거 |
| `models/` | 공통 Chronos-2 캐시와 최신 실험 가중치 |

일부 공통 모듈에는 예전 T5 이름이 남아 있지만 최신 C2 코드가 직접 사용하므로 보존했다.
`results/c2-battery/strict-v2-zero-head/protocol.json`은 고정 배터리 분할,
`results/c2-battery/one-step-64/coverage.csv`는 전처리 전후 비교를 위해 남긴 최소 참조 파일이다.
이 두 폴더의 과거 모델 결과는 별도 보관했다.

## 실행 환경과 진입점

기존 `.venv`를 사용한다. 환경과 설치 패키지는 이번 정리에서 변경하지 않았다.
새 환경에서는 장치에 맞는 PyTorch를 설치한 후 `requirements.txt`를 설치한다.

| 작업 | 진입점 |
|---|---|
| SOH 64 → 1 | `scripts/run_c2_one_step_context.py --reviewed` |
| SOH 나머지 입력 길이 | `scripts/run_c2_context32.py`, `run_c2_context16.py`, `run_c2_context8.py`, `run_c2_context4.py`, `run_c2_context2.py` |
| SOH 그림만 갱신 | `scripts/render_c2_one_step_predictions.py --out results/c2-battery/one-step-2-reviewed-v3` |
| RUL zero-shot / head | `scripts/run_c2_rul_pilot.py`, `scripts/run_c2_rul_head.py` |
| RUL 입력 길이 비교 | `scripts/run_c2_rul_context_sweep.py` — 데스크톱 CUDA 조건 |
| RUL 학습 시점 비교 | `scripts/run_c2_rul_multi_origin.py` — XPU 조건 |
| v3 데이터 구축 | `scripts/build_reviewed_soh.py` |

완료된 결과의 입력·가중치·실행 조건은 고정되어 있다. 새로운 실험은 별도 결과 경로를 사용한다.
CUDA에서 실행한 16·8·4·2 및 RUL 입력 길이 실험은 기존 데스크톱 실행 경로와 조건을 유지한다.
코드를 복사한 실험 패키지와 본 프로젝트는 구분한다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
```

결과 폴더에는 점수·원시 예측·학습 조건·최소 검증 기록을 함께 남겼다.
과거 결과의 소스 해시가 가리키는 실행 파일은 이번 정리에서 수정하지 않았다.

## 이전 실험 보관

2026-10-13 정리 전 코드·문서·결과·가공 데이터는 프로젝트 밖에 복사해 검증했다.
과거 실행 스크립트·노트북·결과·모델은 영구 삭제하지 않고 별도 폴더로 옮겼다.

[별도 보관 및 복구 안내](C:/Users/dbsgh/Documents/CodexBackups/chronos-battery-soh-20261013-114645/README.md)

이 보관 경로는 이 PC의 로컬 경로이며 Git에는 포함되지 않는다.

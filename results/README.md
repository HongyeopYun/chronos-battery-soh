# 실험 결과 탐색

[브라우저 탐색 페이지](index.html) · [프로젝트 구조](../docs/workspace_layout.md)

아래는 저장된 결과의 위치 안내다. 파일 존재와 실험 검증 완료는 구분하며 각 실행의 조건·검증 기록을 확인한다.

## NASA 직접 RUL 개선

- **NASA 직접 RUL 개선 · 요약 및 전후 그래프** — [그래프 보기](c2-battery/rul-nasa-fewshot-report-v1/index.html) · [성능표](c2-battery/rul-nasa-fewshot-report-v1/summary.csv) · [감사 기록](c2-battery/rul-nasa-fewshot-report-v1/audit.json)
- **NASA 개선 Head 이후 전체 가중치 학습** — [검증 기록](c2-battery/rul-nasa-fewshot-full-v1/verification.json) · [실험 조건](c2-battery/rul-nasa-fewshot-full-v1/protocol.json)
- **NASA Head 개선 · 탐색 및 저장 예측** — [성능표](c2-battery/rul-nasa-fewshot-improvement-v1/summary.csv) · [검증 기록](c2-battery/rul-nasa-fewshot-improvement-v1/verification.json) · [실험 조건](c2-battery/rul-nasa-fewshot-improvement-v1/protocol.json)
- **NASA 개선 원인 · 셀 분할 반복 검증** — [검증 기록](c2-battery/rul-nasa-fewshot-robustness-v1/verification.json) · [실험 조건](c2-battery/rul-nasa-fewshot-robustness-v1/protocol.json)

## 직접 RUL · context 및 학습량

- **rul-context-nominal80-adaptation-xpu-v1** — [폴더](c2-battery/rul-context-nominal80-adaptation-xpu-v1/)
- **직접 RUL 입력 길이 · MIT 정격 80%** — [그래프 보기](c2-battery/rul-context-nominal80-v1/index.html) · [보고서](c2-battery/rul-context-nominal80-v1/report.md) · [성능표](c2-battery/rul-context-nominal80-v1/summary.csv) · [검증 기록](c2-battery/rul-context-nominal80-v1/verification.json) · [실험 조건](c2-battery/rul-context-nominal80-v1/protocol.json)
- **직접 RUL 학습 셀 수 · 정답 수** — [보고서](c2-battery/rul-learning-curve-v1/report.md) · [성능표](c2-battery/rul-learning-curve-v1/summary.csv) · [실험 조건](c2-battery/rul-learning-curve-v1/protocol.json)

## 현재 SOH

- **one-step-16-reviewed-v3** — [보고서](c2-battery/one-step-16-reviewed-v3/report.md) · [성능표](c2-battery/one-step-16-reviewed-v3/summary.csv) · [검증 기록](c2-battery/one-step-16-reviewed-v3/verification.json) · [실험 조건](c2-battery/one-step-16-reviewed-v3/protocol.json)
- **one-step-2-reviewed-v3** — [보고서](c2-battery/one-step-2-reviewed-v3/report.md) · [성능표](c2-battery/one-step-2-reviewed-v3/summary.csv) · [검증 기록](c2-battery/one-step-2-reviewed-v3/verification.json) · [실험 조건](c2-battery/one-step-2-reviewed-v3/protocol.json)
- **one-step-32-reviewed-v3** — [보고서](c2-battery/one-step-32-reviewed-v3/report.md) · [성능표](c2-battery/one-step-32-reviewed-v3/summary.csv) · [검증 기록](c2-battery/one-step-32-reviewed-v3/verification.json) · [실험 조건](c2-battery/one-step-32-reviewed-v3/protocol.json)
- **one-step-4-reviewed-v3** — [보고서](c2-battery/one-step-4-reviewed-v3/report.md) · [성능표](c2-battery/one-step-4-reviewed-v3/summary.csv) · [검증 기록](c2-battery/one-step-4-reviewed-v3/verification.json) · [실험 조건](c2-battery/one-step-4-reviewed-v3/protocol.json)
- **one-step-64** — [폴더](c2-battery/one-step-64/)
- **one-step-64-reviewed-v3** — [보고서](c2-battery/one-step-64-reviewed-v3/report.md) · [성능표](c2-battery/one-step-64-reviewed-v3/summary.csv) · [검증 기록](c2-battery/one-step-64-reviewed-v3/verification.json) · [실험 조건](c2-battery/one-step-64-reviewed-v3/protocol.json)
- **one-step-8-reviewed-v3** — [보고서](c2-battery/one-step-8-reviewed-v3/report.md) · [성능표](c2-battery/one-step-8-reviewed-v3/summary.csv) · [검증 기록](c2-battery/one-step-8-reviewed-v3/verification.json) · [실험 조건](c2-battery/one-step-8-reviewed-v3/protocol.json)

## 간접 RUL · 미래 SOH 궤적

- **간접 RUL 대표 예측 궤적** — [그래프 보기](c2-battery/indirect-rul-representatives-v1/index.html) · [검증 기록](c2-battery/indirect-rul-representatives-v1/verification.json)
- **rul-nasa-context-sweep-cuda-v1** — [보고서](c2-battery/rul-nasa-context-sweep-cuda-v1/report.md) · [성능표](c2-battery/rul-nasa-context-sweep-cuda-v1/summary.csv) · [검증 기록](c2-battery/rul-nasa-context-sweep-cuda-v1/verification.json) · [실험 조건](c2-battery/rul-nasa-context-sweep-cuda-v1/protocol.json)
- **rul-nasa-l64-h256-head-v1** — [보고서](c2-battery/rul-nasa-l64-h256-head-v1/report.md) · [검증 기록](c2-battery/rul-nasa-l64-h256-head-v1/verification.json) · [실험 조건](c2-battery/rul-nasa-l64-h256-head-v1/protocol.json)
- **rul-nasa-l64-h256-zero-v1** — [보고서](c2-battery/rul-nasa-l64-h256-zero-v1/report.md) · [검증 기록](c2-battery/rul-nasa-l64-h256-zero-v1/verification.json) · [실험 조건](c2-battery/rul-nasa-l64-h256-zero-v1/protocol.json)
- **rul-nasa-multi-origin-xpu-v1** — [보고서](c2-battery/rul-nasa-multi-origin-xpu-v1/report.md) · [성능표](c2-battery/rul-nasa-multi-origin-xpu-v1/summary.csv) · [검증 기록](c2-battery/rul-nasa-multi-origin-xpu-v1/verification.json) · [실험 조건](c2-battery/rul-nasa-multi-origin-xpu-v1/protocol.json)
- **rul-nasa-soh80-rescore-v1** — [보고서](c2-battery/rul-nasa-soh80-rescore-v1/report.md) · [성능표](c2-battery/rul-nasa-soh80-rescore-v1/summary.csv) · [검증 기록](c2-battery/rul-nasa-soh80-rescore-v1/verification.json) · [실험 조건](c2-battery/rul-nasa-soh80-rescore-v1/protocol.json)

## 통합 결과 및 갤러리

- **SOH 4→1 및 기존 RUL 전체 셀 갤러리** — [그래프 보기](c2-battery/full-soh4-rul-gallery-v1/index.html) · [검증 기록](c2-battery/full-soh4-rul-gallery-v1/verification.json)
- **v4 SOH · 간접 RUL · 직접 RUL 통합 실험** — [보고서](c2-battery/v4-gpu-suite-v2/report.md) · [검증 기록](c2-battery/v4-gpu-suite-v2/verification.json) · [실험 조건](c2-battery/v4-gpu-suite-v2/protocol.json)

## 데이터 전처리 · 감사

- **data_audit** — [폴더](data_audit/)
- **mit-eol-label-audit-v1** — [보고서](c2-battery/mit-eol-label-audit-v1/report.md) · [성능표](c2-battery/mit-eol-label-audit-v1/summary.csv) · [검증 기록](c2-battery/mit-eol-label-audit-v1/verification.json)
- **preprocessing_batterylife_v4** — [보고서](preprocessing_batterylife_v4/report.md) · [검증 기록](preprocessing_batterylife_v4/verification.json)
- **preprocessing_nasa_review_v3** — [보고서](preprocessing_nasa_review_v3/report.md) · [검증 기록](preprocessing_nasa_review_v3/verification.json)
- **preprocessing_review_all** — [보고서](preprocessing_review_all/report.md) · [검증 기록](preprocessing_review_all/verification.json)
- **preprocessing_reviewed_native_v3** — [보고서](preprocessing_reviewed_native_v3/report.md) · [검증 기록](preprocessing_reviewed_native_v3/verification.json)
- **preprocessing_strict_v2** — [보고서](preprocessing_strict_v2/report.md) · [검증 기록](preprocessing_strict_v2/verification.json)

## 과거 참조 및 진단

- **diagnostics** — [폴더](diagnostics/)
- **strict-v2-zero-head** — [실험 조건](c2-battery/strict-v2-zero-head/protocol.json)

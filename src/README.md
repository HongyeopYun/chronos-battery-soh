# 공통 코드 안내

기존 파일은 저장 실험의 경로·소스 해시를 보존하기 위해 이동하지 않았다. 아래에서 용도별로 찾는다.

[프로젝트 구조](../docs/workspace_layout.md) · [전체 결과](../results/index.html)

## NASA 직접 RUL 개선

| 파일 | 설명 |
|---|---|
| [c2_rul_improvement.py](c2_rul_improvement.py) | Training-only preprocessing for the unchanged 771-64-1 direct RUL head. |

## 직접 RUL · context 및 학습량

| 파일 | 설명 |
|---|---|
| [c2_rul_context_nominal.py](c2_rul_context_nominal.py) | Matched-origin direct RUL contexts, with audited MIT nominal-80% labels. |
| [c2_rul_learning_curve.py](c2_rul_learning_curve.py) | Source-data learning curves; splitting never ranks cells by observed lifetime. |

## 현재 SOH

| 파일 | 설명 |
|---|---|
| [c2_one_step.py](c2_one_step.py) | Matched-context, single-target SOH forecasting and head-only adaptation. |

## 간접 RUL · 미래 SOH 궤적

| 파일 | 설명 |
|---|---|
| [c2_rul.py](c2_rul.py) | Capacity-threshold RUL labels and point-trajectory evaluation. |
| [c2_rul_head.py](c2_rul_head.py) | Head-only adaptation on a few partially observed future SOH trajectories. |

## 통합 결과 및 갤러리

| 파일 | 설명 |
|---|---|
| [c2_v4_experiments.py](c2_v4_experiments.py) | Version-4 experiment data and direct RUL models. No future input to encoders. |

## 데이터 전처리 · 감사

| 파일 | 설명 |
|---|---|
| [battery_preprocessing.py](battery_preprocessing.py) | Essential SOH extraction for the NASA, MIT and CALCE files in data/raw. |
| [preprocessing_v4.py](preprocessing_v4.py) | Versioned cycle-level adaptations of the pinned BatteryML/BatteryLife rules. |
| [reviewed_soh.py](reviewed_soh.py) | Audit-revised native-capacity QC and 64-position inputs with missingness. |
| [strict_preprocessing.py](strict_preprocessing.py) | Raw-trace quality checks for SOH; retain measurements and original cycle IDs. |
| [xjtu_preprocessing.py](xjtu_preprocessing.py) | Extract XJTU Batch 1/2 into the common SOH format. |

## 과거 참조 및 진단

| 파일 | 설명 |
|---|---|
| [battery_fewshot_tuning.py](battery_fewshot_tuning.py) | Fixed-budget supervised adaptation, with battery-disjoint held-out evaluation. |
| [battery_model_zeroshot.py](battery_model_zeroshot.py) | Independent T5/Chronos-2 zero-shot with standard R2 and observed-period thirds. |
| [battery_zeroshot.py](battery_zeroshot.py) | Dataset-independent Chronos zero-shot evaluation of common SOH bundles. |
| [battery_zeroshot_minimal.py](battery_zeroshot_minimal.py) | Compact zero-shot results: score every window, keep tables and figures only. |
| [battery_zeroshot_plots.py](battery_zeroshot_plots.py) | Small reports and representative window plots for common SOH zero-shot runs. |
| [c2_context_comparison.py](c2_context_comparison.py) | Paired persistence comparisons for matched one-step prediction tables. |
| [c2_fewshot_trainer.py](c2_fewshot_trainer.py) | Keep the SOH loss scale equal when adding masked past-only sensor rows. |
| [c2_head_protocol.py](c2_head_protocol.py) | Predefined, physical-cell-disjoint folds for small-budget head adaptation. |
| [c2_partial_tuning.py](c2_partial_tuning.py) | Small, explicit Chronos-2 parameter subsets and validation-only selection. |
| [calce_windows.py](calce_windows.py) | CALCE windows in chronological observation order, across recording files. |
| [soh_checkpoint.py](soh_checkpoint.py) | EOS-masked LoRA: validation-selected learning rate and saved checkpoint. |
| [soh_cleaning.py](soh_cleaning.py) | Conservative capacity-spike exclusion using only earlier measurements. |
| [soh_data.py](soh_data.py) | Common, cycle-level SOH tables; no model or NASA experiment dependencies. |
| [soh_evaluation.py](soh_evaluation.py) | Derived evaluation tables with R²; original training artifacts stay immutable. |
| [soh_experiment.py](soh_experiment.py) | Shared zero-shot / LoRA LOBO implementation, used by both notebooks and CLI. |

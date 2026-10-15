# 실행 진입점 안내

기존 파일은 저장 실험의 경로·소스 해시를 보존하기 위해 이동하지 않았다. 아래에서 용도별로 찾는다.

[프로젝트 구조](../docs/workspace_layout.md) · [전체 결과](../results/index.html)

## NASA 직접 RUL 개선

| 파일 | 설명 |
|---|---|
| [audit_nasa_fewshot_improvement.py](audit_nasa_fewshot_improvement.py) | Independent post-run audit of the locked NASA few-shot experiments. |
| [audit_nasa_fewshot_robustness.py](audit_nasa_fewshot_robustness.py) | Locally reload every selected head in the 12-partition factorial audit. |
| [report_nasa_fewshot_improvement.py](report_nasa_fewshot_improvement.py) | Produce a scientific report from audited, saved NASA predictions only. |
| [run_c2_nasa_fewshot_full.py](run_c2_nasa_fewshot_full.py) | Unchanged-architecture full-weight few-shot follow-up, after head validation. |
| [run_c2_nasa_fewshot_improvement.py](run_c2_nasa_fewshot_improvement.py) | NASA few-shot improvement; unchanged architecture; validation-only selection. |
| [run_c2_nasa_fewshot_robustness.py](run_c2_nasa_fewshot_robustness.py) | NASA paired 2x2 scaling/loss ablation over every disjoint 2/1/1 partition. |

## 직접 RUL · context 및 학습량

| 파일 | 설명 |
|---|---|
| [benchmark_rul_adaptation_xpu.py](benchmark_rul_adaptation_xpu.py) | Measure real XPU Full/LoRA backward passes using source data only. |
| [export_c2_rul_context_nominal.py](export_c2_rul_context_nominal.py) | Transfer finished raw GPU artifacts for independent local audit/reporting. |
| [export_c2_rul_learning_curve.py](export_c2_rul_learning_curve.py) | Bundle verified results, all heads, and exact source snapshots for transfer. |
| [report_c2_rul_adaptation_xpu.py](report_c2_rul_adaptation_xpu.py) | Completion audit and matched Head/Full/LoRA report; no model training. |
| [report_c2_rul_context_nominal.py](report_c2_rul_context_nominal.py) | Audit matched context sweep and render all saved test cells. |
| [report_c2_rul_learning_curve.py](report_c2_rul_learning_curve.py) | Independently audit learning-curve artifacts, then produce comparison plots. |
| [run_c2_rul_adaptation_xpu.py](run_c2_rul_adaptation_xpu.py) | Resumable laptop XPU Full/LoRA follow-up to the frozen nominal80 Head run. |
| [run_c2_rul_context_nominal.py](run_c2_rul_context_nominal.py) | CUDA head-only context sweep. Historical artifacts are never changed. |
| [run_c2_rul_learning_curve.py](run_c2_rul_learning_curve.py) | Frozen-backbone CUDA RUL source-cell and label-budget learning curves. |
| [start_c2_rul_adaptation_xpu.ps1](start_c2_rul_adaptation_xpu.ps1) | 파일의 실행 인자와 연결된 보고서 확인 |

## 현재 SOH

| 파일 | 설명 |
|---|---|
| [render_c2_context32_results.py](render_c2_context32_results.py) | Presentation-only refresh; the completed model protocol remains unchanged. |
| [render_c2_context4_comparison.py](render_c2_context4_comparison.py) | Readable supplementary chart from frozen desktop scores; no re-inference. |
| [render_c2_one_step_predictions.py](render_c2_one_step_predictions.py) | Render saved one-step predictions with only eligible SOH observations. |
| [report_c2_context16.py](report_c2_context16.py) | Report matched 64/32/16 one-step results without changing earlier runs. |
| [report_c2_context2.py](report_c2_context2.py) | Report matched 64/32/16/8/4/2 one-step results without changing earlier runs. |
| [report_c2_context4.py](report_c2_context4.py) | Report matched 64/32/16/8/4 one-step results without changing earlier runs. |
| [report_c2_context8.py](report_c2_context8.py) | Report matched 64/32/16/8 one-step results without changing earlier runs. |
| [run_c2_context16.py](run_c2_context16.py) | 16->1 follow-up with the frozen reviewed 64->1 labels and test timestamps. |
| [run_c2_context2.py](run_c2_context2.py) | 2->1 follow-up with the frozen reviewed 64->1 labels and test timestamps. |
| [run_c2_context32.py](run_c2_context32.py) | 32->1 follow-up with the frozen reviewed 64->1 labels and test timestamps. |
| [run_c2_context4.py](run_c2_context4.py) | 4->1 follow-up with the frozen reviewed 64->1 labels and test timestamps. |
| [run_c2_context8.py](run_c2_context8.py) | 8->1 follow-up with the frozen reviewed 64->1 labels and test timestamps. |
| [run_c2_one_step_context.py](run_c2_one_step_context.py) | Initial C2 64->1 experiment on identical held-out timestamps. |
| [verify_c2_context2_release.py](verify_c2_context2_release.py) | Independent strict release audit of the saved two-position experiment. |

## 간접 RUL · 미래 SOH 궤적

| 파일 | 설명 |
|---|---|
| [render_indirect_rul_representatives.py](render_indirect_rul_representatives.py) | Explain saved indirect-RUL trajectories; no training or new inference. |
| [report_c2_rul_context_sweep.py](report_c2_rul_context_sweep.py) | Matched t64 RUL comparison and horizon-specific failure diagnostics. |
| [report_c2_rul_head.py](report_c2_rul_head.py) | Matched zero-shot/head-only RUL tables and held-out-cell figures. |
| [report_c2_rul_multi_origin.py](report_c2_rul_multi_origin.py) | Report the paired XPU multi-origin intervention without hiding non-hits. |
| [rescore_c2_rul_soh80.py](rescore_c2_rul_soh80.py) | Rescore saved NASA RUL trajectories at SOH <= 0.8; no inference or fitting. |
| [run_c2_rul_context_sweep.py](run_c2_rul_context_sweep.py) | Matched NASA t=64, H256 RUL diagnostic sweep: L16/32/64 on desktop CUDA. |
| [run_c2_rul_head.py](run_c2_rul_head.py) | Four held-out NASA folds: native C2 output-head adaptation, L64 H256. |
| [run_c2_rul_multi_origin.py](run_c2_rul_multi_origin.py) | Matched t64 evaluation: single versus three source origins, fresh XPU heads. |
| [run_c2_rul_pilot.py](run_c2_rul_pilot.py) | NASA L=64 indirect RUL pilot; fixed H=256, native Chronos-2 zero-shot. |
| [verify_c2_rul_context_sweep.py](verify_c2_rul_context_sweep.py) | Independent local re-score and annotated horizon diagnostic figure. |

## 통합 결과 및 갤러리

| 파일 | 설명 |
|---|---|
| [export_c2_v4_suite.py](export_c2_v4_suite.py) | Finalize a verified remote run and export results and selected weights separately. |
| [render_full_soh_rul_gallery.py](render_full_soh_rul_gallery.py) | Whole-test-cell galleries of saved SOH L4 and historical direct RUL runs. |
| [run_c2_v4_suite.py](run_c2_v4_suite.py) | Frozen-protocol CUDA suite: rolling SOH, trajectory RUL, direct RUL adaptation. |
| [verify_report_c2_v4_suite.py](verify_report_c2_v4_suite.py) | Independently re-score saved v4 predictions and render the final report. |

## 데이터 전처리 · 감사

| 파일 | 설명 |
|---|---|
| [audit_battery_preprocessing.py](audit_battery_preprocessing.py) | Independent raw-data and coverage audit. Never changes data or runs models. |
| [audit_mit_eol_labels.py](audit_mit_eol_labels.py) | Read-only audit of author-provided MIT nominal-80% cycle-life labels. |
| [build_preprocessing_v4.py](build_preprocessing_v4.py) | Build audited v4 forecast-safe and retrospective views; never edit v3. |
| [build_reviewed_soh.py](build_reviewed_soh.py) | Build a separate v3 dataset from audited native capacities; never overwrite v2. |
| [clean_soh_outliers.py](clean_soh_outliers.py) | Create conservative cleaned bundles and a small before/after report. |
| [inspect_mit_qc_cases.py](inspect_mit_qc_cases.py) | Inspect every non-collection MIT exclusion, separating waveform failure modes. |
| [preprocess_remaining_batteries.py](preprocess_remaining_batteries.py) | Process all remaining raw datasets; keep reports to essential checks and SOH curves. |
| [preprocess_strict_batteries.py](preprocess_strict_batteries.py) | Audit every raw cycle and build separate strict SOH bundles. No model runs. |
| [preprocess_xjtu.py](preprocess_xjtu.py) | Extract XJTU 2C/3C bundles and write essential summaries and SOH curves. |
| [report_preprocessing_review.py](report_preprocessing_review.py) | Summarize completed independent audits without changing masks or scores. |
| [review_nasa_capacity.py](review_nasa_capacity.py) | Re-audit NASA raw capacity, retaining a verified discharge with an absent charge log. |
| [verify_preprocessing_v4.py](verify_preprocessing_v4.py) | Independent saved-table audit of the v4 release, including raw spot checks. |

## 관리 도구

- [탐색 목록 갱신](maintenance/update_workspace_index.py): 학습 없이 현재 파일 목록에서 결과·코드 안내를 다시 생성한다.

실행 예: `.venv\Scripts\python.exe scripts/maintenance/update_workspace_index.py`

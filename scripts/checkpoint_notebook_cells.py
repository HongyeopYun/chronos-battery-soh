"""Current LoRA notebook cells; historical diagnosis is appended by the builder."""


def checkpoint_lora_cells(setup):
    return [
        ("markdown", """# Chronos LoRA: EOS loss 제외 · best checkpoint · 학습률 3개 비교

**현재 프로토콜:** 평가 배터리를 제외한 세 배터리에서 시간순 train/validation을 나눕니다. 입력 EOS는 유지하고 출력 EOS loss만 제외합니다. 학습률 `1e-5`, `3e-5`, `1e-4` 각각 최대 100 optimizer steps를 학습하며, 10 steps마다 validation CRPS를 계산합니다. Validation을 학습에 합치는 **full-data refit은 하지 않습니다.**

각 후보의 best checkpoint를 저장하고 validation CRPS로 학습률을 선택합니다. Step 0도 후보입니다. 네 fold의 선택을 모두 고정한 후 저장된 adapter를 재로딩해 held-out 배터리를 평가합니다. 세 학습률은 Optuna 없이 모두 비교합니다.

R²는 중앙값·평균 예측에 각각 계산합니다. 각 예측창의 미래 실제 평균으로 `1 - SSE/SST`를 계산한 뒤 창별 점수를 단순평균하고 seed·배터리를 동일 가중 평균합니다. 상수 target 창은 NaN으로 제외하며 음수는 유지합니다. 원시 validation 예측이 없는 과거 이력은 R²를 복원할 수 없어 NaN입니다. Horizon 한 개의 R²도 정의되지 않습니다. 학습률·checkpoint 선택 기준은 CRPS로 유지합니다.

학습 seed는 42, validation seed는 31415/31416/31417(각 32 samples), 최종 추론 seed는 42/43/44(각 100 samples)입니다. 추론 반복과 학습 반복은 다릅니다. 04 Full FT는 여기서 실행하지 않습니다.

마지막 절의 2026-09-29 EOS 진단은 **과거 full-data 학습 결과**이며, 이 새 프로토콜과 구분합니다. [프로토콜](../docs/lora_checkpoint_protocol.md) · [현재 결과 보고서](../results/checkpoint_lora/report.md)"""),
        ("code", setup + '''\nfrom src.soh_checkpoint import (
    load_checkpoint_config, find_checkpoint_run, run_checkpoint_search, verify_checkpoint_run,
)
CONFIG, CHECKPOINT_POLICY = load_checkpoint_config(PROJECT_ROOT)
print("Current LoRA policy:", CHECKPOINT_POLICY)
'''),
        ("markdown", """## 학습·검증·평가 배터리 구분

평가 배터리 1개는 학습률과 step 선택에 사용하지 않습니다. 나머지 세 배터리의 뒤쪽 20% target은 validation으로 남기며, target이 경계에 걸친 window는 제외합니다. Validation의 context는 해당 시점까지 실제 관측된 과거입니다."""),
        ("code", '''soh_data = load_soh(PROJECT_ROOT, CONFIG)
split_rows = []
for battery, series in soh_data.items():
    training, validation, cutoff = temporal_split(series, CONFIG)
    split_rows.append({
        "battery": battery, "train_windows": len(training["target"]),
        "validation_windows": len(validation["target"]),
        "train_target_last_cycle": int(training["forecast_start"][-1] + CONFIG.prediction_length - 1),
        "validation_target_first_cycle": int(validation["forecast_start"][0]),
    })
display(pd.DataFrame(split_rows))
display(pd.DataFrame([{"test": b, "train/validation batteries": ", ".join(x for x in CONFIG.battery_ids if x != b)} for b in CONFIG.battery_ids]))'''),
        ("markdown", """## 12개 후보 학습과 best checkpoint 평가

완료된 동일 실행은 소스·데이터·설정·산출물 해시 및 원시 예측 지표를 검산한 뒤 재사용합니다. 호환되는 실행이 없으면 3개 학습률 × 4개 fold를 실행합니다. `FORCE_RERUN=True`는 새 프로토콜의 전체 탐색을 다시 실행합니다.

각 후보는 같은 초기화와 minibatch 순서로 학습합니다. 검증 난수는 학습 상태를 바꾸지 않습니다. CRPS가 `min_delta=1e-5`보다 크게 개선될 때만 checkpoint를 교체하며, 유사한 점수에서는 앞선 step/학습률을 유지합니다."""),
        ("code", '''CHECKPOINT_RUN = None if FORCE_RERUN else find_checkpoint_run(PROJECT_ROOT, CONFIG, CHECKPOINT_POLICY)
CHECKPOINT_RUN = CHECKPOINT_RUN or run_checkpoint_search(PROJECT_ROOT, CONFIG, CHECKPOINT_POLICY)
manifest = json.loads((CHECKPOINT_RUN / "manifest.json").read_text(encoding="utf-8"))
ZERO_RUN = Path(manifest["zeroshot_run"])
print("Current LoRA run:", CHECKPOINT_RUN)
print("Zero-shot reference:", ZERO_RUN)
display(verify_checkpoint_run(CHECKPOINT_RUN, PROJECT_ROOT, CONFIG, CHECKPOINT_POLICY))
CHECKPOINT_EVAL = export_run(PROJECT_ROOT, CHECKPOINT_RUN)
ZERO_EVAL = evaluation_directory(PROJECT_ROOT, ZERO_RUN)'''),
        ("markdown", """## Validation으로 선택한 학습률과 step

아래 CRPS는 test 점수가 아니라 학습 배터리의 validation 점수입니다. 세 validation seed의 sample을 합치지 않고 각 seed의 점수를 먼저 계산한 뒤 평균합니다. `selected_step`은 실제로 평가할 저장 checkpoint의 시점, `executed_steps`는 후보 학습을 탐색한 총 횟수입니다."""),
        ("code", '''display(pd.DataFrame([
    {"test_battery": fold["test_battery"], "train_seed": fold["train_seed"],
     "learning_rate": fold["learning_rate"], "selected_step": fold["selected_step"],
     "executed_steps": fold["executed_steps"],
     "validation_crps": fold["selected_validation"]["macro"]["crps"]}
    for fold in manifest["folds"]
]))
display(pd.read_csv(CHECKPOINT_EVAL / "validation_candidates.csv").drop(columns="trial_path"))
from scripts.summarize_checkpoint_lora import plot_learning_curves
plot_learning_curves(CHECKPOINT_RUN)
plt.show()'''),
        ("markdown", """## 같은 조건의 zero-shot과 비교

학습률·checkpoint 선택이 끝난 후 생성한 held-out 예측입니다. 표는 seed별 점수를 평균한 뒤 배터리에 같은 가중치를 줍니다. `mean_rmse`는 sample mean을 점예측으로 사용한 오차입니다. `coverage80`과 이상 sample 비율은 0~1 단위입니다."""),
        ("code", '''current_scores = pd.read_csv(CHECKPOINT_EVAL / "distribution_metrics.csv")
score_columns = ["median_rmse", "median_mae", "median_r2", "mean_rmse", "mean_r2", "crps", "coverage80", "width80", "above2_rate"]
current_battery = current_scores.groupby(["model", "battery"])[score_columns].mean()
current_macro = current_battery.groupby(level="model").mean().loc[["ZeroShot", "SelectedLoRA"]]
display(current_macro.round(6))
display(current_battery.round(6))
print("Zero-shot 대비 오차 감소율 (%; 양수=개선)")
display((100 * (1 - current_macro.loc["SelectedLoRA", ["median_rmse", "mean_rmse", "crps"]]
                / current_macro.loc["ZeroShot", ["median_rmse", "mean_rmse", "crps"]])).round(3))
print("R² change (absolute difference; higher is better)")
display((current_macro.loc["SelectedLoRA", ["median_r2", "mean_r2"]] - current_macro.loc["ZeroShot", ["median_r2", "mean_r2"]]).round(6))
display(pd.read_csv(CHECKPOINT_EVAL / "paired_macro.csv"))'''),
        ("markdown", """## 선택 checkpoint의 SOH 곡선

각 배터리의 첫·중간·마지막 window를 선택했습니다. 과거 관측 64개로 미래 16개를 예측하며, 평활화나 clipping 없이 예측 중앙값과 10~90% 구간을 표시합니다. 추론 seed 42의 예시이며 전체 결과는 위의 세 seed 평가 표입니다."""),
        ("code", '''forecast_grid(CHECKPOINT_RUN)
plt.show()
horizon_comparison(ZERO_RUN, CHECKPOINT_RUN)
plt.show()
'''),
        ("markdown", """## 저장·재현 확인과 해석 범위

선택한 adapter를 재로딩했을 때 validation의 원시 sample 배열이 저장 시점과 완전히 일치하는지 검사합니다. 아래 파일에는 그 결과와 모든 test 예측 전 확정한 선택 기록이 있습니다.

학습 seed는 하나입니다. Step 0을 선택하면 미학습 모델이며, 100을 선택하면 이번 탐색 상한에서의 선택입니다. 세 검증 난수와 겹치는 window가 새 배터리 사례를 추가하지는 않습니다. 이 네 배터리를 이미 살펴봤기 때문에 독립적인 새 벤치마크라고 해석하지 않습니다."""),
        ("code", '''replay_checks = pd.read_json(CHECKPOINT_RUN / "checkpoint_replay.json")
display(replay_checks.reindex(columns=["test_battery", "selected_step", "validation_arrays_identical", "validation_arrays_checked", "macro_crps_delta"]))
if "replay_acceptance" in manifest:
    display(manifest["replay_acceptance"])
    print("Sample equality was not mandatory in this evaluation; see the report for replay differences and the recorded pre-test acceptance rule.")
frozen = json.loads((CHECKPOINT_RUN / "selection_frozen.json").read_text(encoding="utf-8"))
print("Choices frozen before test forecasting:", frozen["frozen_at"])
print("No full-data refit:", all(not fold["refit_full_data"] for fold in manifest["folds"]))'''),
    ]

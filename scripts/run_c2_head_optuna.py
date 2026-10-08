"""Fixed-budget Optuna head tuning, then seed/window robustness and held-out tests.

No test scores enter search. Existing 20/10 label windows and crossing guards stay
fixed. All search and confirmation checkpoints are locked before test inference.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
from transformers import set_seed

from src.battery_fewshot_tuning import fixed_training_inputs, adaptation_scores
from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows, sha256, write_json
from src.battery_model_zeroshot import load_model, predict_quantiles, phase_labels, PHASES
from src.c2_head_protocol import make_folds
from src.c2_head_search import alternate_label_windows, indices_from_budget
from src.c2_partial_tuning import configure_subset, model_batch, corrected_loss, validation_scores, eligible_checkpoint, choose_checkpoint
from src.soh_evaluation import window_r2_values, R2_DEFINITION
from scripts.report_r2_distribution import distribution

OUT = ROOT / "results/c2-battery/head-only-optuna20"
WEIGHTS = ROOT / "models/c2-head-optuna20"
GROUPS = ("xjtu_2c", "xjtu_3c")
SEEDS = (42, 43, 44)
TRIALS = 16


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def restore(model, weights):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in weights.items():
            params[name].copy_(value)


def sync(model):
    if model.device.type == "xpu": torch.xpu.synchronize()
    elif model.device.type == "cuda": torch.cuda.synchronize()


def prepare(group):
    old_dir = ROOT / f"results/c2-battery/fewshot-partial-tuning/{group}"
    original = read(ROOT / f"results/c2-battery/full/{group}/c2-battery_run.json")
    bundle = Path(original["bundle"])
    if sha256(bundle / "manifest.json") != original["bundle_fingerprint"]:
        raise ValueError("Dataset manifest changed")
    config = replace(ZeroShotConfig(**original["config"]), battery_ids=None, max_windows_per_battery=None)
    records, coverage, data, _ = prepare_battery_windows(bundle, config)
    fold = make_folds(group, records)[0]
    by_id = {r["battery_id"]: r for r in records}
    old_budget = pd.read_csv(old_dir / "label_budget.csv")
    sets, budget_rows = {}, []
    empty_features = pd.DataFrame({"cycle": []})
    for selection in ("original", "alternate"):
        tasks = {"train": [], "validation": []}
        for split, ids in (("train", fold["train_ids"]), ("validation", fold["validation_ids"])):
            for b in ids:
                w = by_id[b]["windows"]
                rows = old_budget[(old_budget.battery_id == b) & (old_budget.split == split)]
                old_indices = indices_from_budget(w, rows)
                assert len(old_indices) == 10
                # Validation labels remain identical even in the training-window control.
                indices = (alternate_label_windows(w, 10, 1042 + int(b.rsplit("-", 1)[1]))
                           if selection == "alternate" and split == "train" else old_indices)
                if selection == "alternate" and split == "train" and np.array_equal(indices, old_indices):
                    raise ValueError("Alternate training windows did not change")
                tasks[split].extend(fixed_training_inputs(w, empty_features, indices))
                for i in indices:
                    budget_rows.append(dict(selection=selection, split=split, battery_id=b,
                        context_start=int(w["context_cycle"][i, 0]), context_end=int(w["context_cycle"][i, -1]),
                        label_start=int(w["target_cycle"][i, 0]), label_end=int(w["target_cycle"][i, -1])))
        assert len(tasks["train"]) == 20 and len(tasks["validation"]) == 10
        sets[selection] = tasks
    budget = pd.DataFrame(budget_rows)
    pd.testing.assert_frame_equal(old_budget.reset_index(drop=True),
        budget[budget.selection == "original"].drop(columns="selection").reset_index(drop=True))
    directory = OUT / group
    directory.mkdir(parents=True, exist_ok=True)
    budget.to_csv(directory / "label_budget.csv", index=False)
    coverage["role"] = coverage.battery_id.map(lambda b: "train" if b in fold["train_ids"] else
        "validation" if b in fold["validation_ids"] else "test")
    coverage.to_csv(directory / "coverage.csv", index=False)
    source = read(old_dir / "run.json")
    previous = source["selected"]["SOH/head"]
    if sha256(Path(previous["selected_weights"])) != previous["selected_weights_sha256"]:
        raise ValueError("Previous head changed")
    return dict(config=config, records=by_id, data=data, sets=sets, fold=fold, previous=previous,
                bundle_sha256=original["bundle_fingerprint"], label_budget_sha256=sha256(directory / "label_budget.csv"))


def validation_batches(model, tasks):
    return [model_batch(b, model.device) for b in Chronos2Dataset(inputs=tasks, context_length=64,
        prediction_length=16, batch_size=4, output_patch_size=16, min_past=64, mode=DatasetMode.VALIDATION)]


def fit_candidate(model, initial, tasks, validation, baseline, lr, max_steps, seed, directory, weight_path, progress):
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "selected.json"
    if metadata_path.exists():
        saved = read(metadata_path)
        if (saved["seed"], saved["requested_learning_rate"], saved["max_steps"]) != (seed, lr, max_steps):
            raise ValueError("Cannot reuse a different candidate")
        if sha256(Path(saved["weights_path"])) != saved["weights_sha256"]:
            raise ValueError("Saved candidate changed")
        return saved
    restore(model, initial)
    set_seed(seed)
    names = configure_subset(model, "head")
    assert set(names) == set(initial)
    params = dict(model.named_parameters())
    frozen_versions = {n: p._version for n, p in params.items() if n not in names}
    optimizer = torch.optim.AdamW([params[n] for n in names], lr=lr, weight_decay=0., eps=1e-8)
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {id(params[n]) for n in names}
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: max(0., 1 - s / max_steps))
    iterator = iter(Chronos2Dataset(inputs=tasks, context_length=64, prediction_length=16,
        batch_size=4, output_patch_size=16, min_past=64, mode=DatasetMode.TRAIN))
    history, winner = [baseline], baseline
    weights = {n: v.clone() for n, v in initial.items()}
    sync(model)
    started = time.perf_counter()
    for step in range(1, max_steps + 1):
        batch = model_batch(next(iterator), model.device)
        assert batch["context"].shape == (4, 64)
        assert torch.isfinite(batch["future_target"]).sum().item() == 64
        assert torch.isnan(batch["future_covariates"]).all()
        optimizer.zero_grad(set_to_none=True)
        loss = corrected_loss(model, batch)
        if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([params[n] for n in names], 1.)
        optimizer.step()
        scheduler.step()
        if step % 20 == 0:
            row = dict(step=step, learning_rate=lr, **validation_scores(model, validation))
            row["eligible"] = eligible_checkpoint(row, baseline)
            history.append(row)
            chosen = choose_checkpoint(history, baseline)
            if chosen is row:
                winner = row
                weights = {n: params[n].detach().cpu().clone() for n in names}
            pd.DataFrame(history).to_csv(directory / "validation.csv", index=False)
            progress(step=step, max_steps=max_steps, best_validation_loss=winner["validation_loss"])
            configure_subset(model, "head")
    sync(model)
    elapsed = time.perf_counter() - started
    assert all(params[n]._version == v and params[n].grad is None for n, v in frozen_versions.items())
    assert winner == choose_checkpoint(history, baseline)
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, weight_path)
    result = dict(**winner, seed=seed, requested_learning_rate=lr, max_steps=max_steps,
        training_validation_seconds=elapsed, weights_path=str(weight_path), weights_sha256=sha256(weight_path),
        frozen_backbone_verified=True, optimizer_subset_verified=True,
        best_unconstrained_loss=min(r["validation_loss"] for r in history),
        eligible_trained_checkpoints=sum(r["eligible"] for r in history[1:]),
        trainable_parameters=sum(params[n].numel() for n in names), total_parameters=sum(p.numel() for p in params.values()))
    write_json(metadata_path, result)
    restore(model, initial)
    return result


def search(group, prepared, model, initial, status):
    directory = OUT / group
    selection_path = directory / "selection.json"
    if selection_path.exists(): return read(selection_path)
    validation = validation_batches(model, prepared["sets"]["original"]["validation"])
    restore(model, initial)
    baseline = dict(step=0, learning_rate=0., eligible=True, **validation_scores(model, validation))
    # Same source data, normalizer and head as the previous 20-window experiment.
    old = read(ROOT / f"results/c2-battery/fewshot-partial-tuning/{group}/run.json")["selected"]["SOH/head"]
    np.testing.assert_allclose(baseline["validation_loss"], old["baseline"]["validation_loss"], rtol=2e-5)
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=6)
    study = optuna.create_study(study_name=f"{group}-head20", direction="minimize", sampler=sampler,
        storage=f"sqlite:///{(directory / 'study.sqlite3').as_posix()}", load_if_exists=True)
    if not study.trials:
        for lr, steps in ((1e-6, 100), (1e-5, 100), (1e-6, 400), (3e-6, 200)):
            study.enqueue_trial(dict(learning_rate=lr, max_steps=steps))

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-7, 3e-5, log=True)
        steps = trial.suggest_categorical("max_steps", [100, 200, 400])
        def progress(**more):
            status.update(status="searching", group=group, trial=trial.number + 1, trials_per_group=TRIALS,
                          learning_rate=lr, **more)
            write_json(OUT / "status.json", status)
        progress(step=0, max_steps=steps)
        selected = fit_candidate(model, initial, prepared["sets"]["original"]["train"], validation, baseline,
            lr, steps, 42, directory / "trials" / f"trial_{trial.number:02d}",
            WEIGHTS / group / f"trial_{trial.number:02d}.pt", progress)
        trial.set_user_attr("selected", selected)
        print(f"TRIAL {group} {trial.number+1}/{TRIALS}: lr={lr:.3g} max={steps} best_step={selected['step']} "
              f"loss={selected['validation_loss']:.5f} seconds={selected['training_validation_seconds']:.1f}", flush=True)
        return selected["validation_loss"]

    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    study.optimize(objective, n_trials=max(0, TRIALS - completed), n_jobs=1)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(done) != TRIALS: raise ValueError("Search budget changed")
    # Keep the old candidate on tied outcomes rather than preferring a longer run.
    winner = min(done, key=lambda t: (t.value, t.user_attrs["selected"]["step"],
                                      t.params["max_steps"], t.params["learning_rate"], t.number))
    rows = [dict(trial=t.number, **t.params, **{k: v for k, v in t.user_attrs["selected"].items()
                                              if k not in ("max_steps", "learning_rate")}) for t in done]
    pd.DataFrame(rows).to_csv(directory / "trials.csv", index=False)
    result = dict(group=group, selected_trial=winner.number, learning_rate=winner.params["learning_rate"],
        max_steps=winner.params["max_steps"], selected=winner.user_attrs["selected"], baseline=baseline,
        n_trials=len(done), search_seconds=sum(t.user_attrs["selected"]["training_validation_seconds"] for t in done),
        selection_locked_at=now(), sampler="TPESampler(seed=42,n_startup_trials=6)")
    write_json(selection_path, result)
    return result


def confirm(group, prepared, model, initial, selection, status):
    directory = OUT / group
    result_path = directory / "confirmations.json"
    if result_path.exists(): return read(result_path)
    restore(model, initial)
    validation = validation_batches(model, prepared["sets"]["original"]["validation"])
    baseline = dict(step=0, learning_rate=0., eligible=True, **validation_scores(model, validation))
    outcomes = []
    for window_set in ("original", "alternate"):
        for seed in SEEDS:
            name = f"{window_set}_seed{seed}"
            if window_set == "original" and seed == 42:
                selected = selection["selected"]
            else:
                def progress(**more):
                    status.update(status="confirming", group=group, confirmation=name,
                                  learning_rate=selection["learning_rate"], **more)
                    write_json(OUT / "status.json", status)
                selected = fit_candidate(model, initial, prepared["sets"][window_set]["train"], validation,
                    baseline, selection["learning_rate"], selection["max_steps"], seed,
                    directory / "confirmations" / name, WEIGHTS / group / f"{name}.pt", progress)
            outcomes.append(dict(condition=f"optuna_{window_set}", **selected))
            print(f"CONFIRM {group} {name}: selected step {selected['step']}, loss {selected['validation_loss']:.5f}", flush=True)
    result = dict(group=group, locked_at=now(), outcomes=outcomes,
                  rule="LR and maximum training steps locked from search; validation chooses step per seed. No seed or training-window set selected using test results.")
    write_json(result_path, result)
    return result


def summarize_scores(battery, windows, by_seed=False):
    keys = ["dataset", "condition", "cohort", "phase"] + (["seed"] if by_seed else [])
    rows = []
    for key, frame in battery.groupby(keys, sort=False):
        tags = dict(zip(keys, key))
        selected = windows
        for name, value in tags.items():
            if name != "phase" or value != "all": selected = selected[selected[name] == value]
        stats = distribution(selected)
        per_seed = frame.groupby("seed", sort=False)
        scores = {k: float(frame[k].mean(skipna=False)) for k in
                  ("rmse", "wql", "picp80", "width80", "invalid_interval_fraction", "quantile_crossing_fraction")}
        np.testing.assert_allclose(stats["r2_mean"], frame.r2.mean(), rtol=1e-9, atol=1e-9)
        seed_rmse = per_seed.rmse.mean()
        rows.append(dict(**tags, n_batteries=frame.battery_id.nunique(), n_seeds=frame.seed.nunique(),
            n_windows_per_seed=int(frame.n_windows.sum() / frame.seed.nunique()),
            rmse_seed_std=float(seed_rmse.std(ddof=0)), rmse_seed_min=float(seed_rmse.min()),
            rmse_seed_max=float(seed_rmse.max()), **scores, **stats))
    return pd.DataFrame(rows)


def representative_indices(w, cycles):
    phases = phase_labels(w["forecast_start"], cycles)
    result = [0]
    for phase, fraction in (("middle", .5), ("late", 5/6)):
        indices = np.flatnonzero(phases == phase)
        center = cycles.min() + fraction * (cycles.max() - cycles.min() + 1)
        result.append(int(indices[np.argmin(abs(w["forecast_start"][indices] - center))]) if len(indices) else len(phases)-1)
    return result


def evaluate(group, prepared, model, initial, confirmations, status):
    directory = OUT / group
    if (directory / "evaluation.json").exists(): return
    selections = [dict(condition="zero-shot", seed=42, weights_path=None),
                  dict(condition="previous_head20", seed=42, weights_path=prepared["previous"]["selected_weights"]),
                  *confirmations["outcomes"]]
    test_ids = prepared["fold"]["test_ids"]
    records = [prepared["records"][b] for b in test_ids]
    representative_ids = [test_ids[i] for i in np.linspace(0, len(test_ids)-1, min(4, len(test_ids)), dtype=int)]
    with np.load(ROOT / f"results/c2-battery/fewshot/{group}/c2-battery_{group}_fewshot_predictions.npz") as cache:
        slide_starts = {b: cache[f"{b}_forecast_start"].copy() for b in test_ids if f"{b}_forecast_start" in cache}
    battery_rows, window_rows, reps = [], [], {}
    old_cache = ROOT / f"results/c2-battery/head-only-xjtu-all-test/{group}/predictions.npz"
    old_conditions = ["zero-shot", "previous_head20"]
    with np.load(old_cache) as cache:
        for selected in selections:
            condition, seed = selected["condition"], selected["seed"]
            weights = initial if selected["weights_path"] is None else torch.load(selected["weights_path"], map_location="cpu", weights_only=True)
            if set(weights) != set(initial): raise ValueError("Checkpoint is not head-only")
            restore(model, weights)
            model.eval()
            for record in records:
                b, w = record["battery_id"], record["windows"]
                status.update(status="evaluating", group=group, condition=condition, seed=seed, battery=b)
                write_json(OUT / "status.json", status)
                if condition in old_conditions:
                    np.testing.assert_array_equal(cache[f"{b}_target"], w["target"])
                    np.testing.assert_array_equal(cache[f"{b}_forecast_start"], w["forecast_start"])
                    q = cache[f"{b}_condition{old_conditions.index(condition)}_quantiles"].copy()
                    # Verify loaded checkpoint and cached forecasts on a fixed input per record.
                    actual = predict_quantiles(PIPELINE, w["context"][:1], prepared["config"])[1]
                    np.testing.assert_allclose(actual, q[:1], rtol=2e-5, atol=1e-7)
                else:
                    q = predict_quantiles(PIPELINE, w["context"], prepared["config"])[1]
                cycles = prepared["data"].loc[(prepared["data"].battery_id == b) & prepared["data"].is_observed, "cycle"].to_numpy()
                phases = phase_labels(w["forecast_start"], cycles)
                r2 = window_r2_values(w["target"], q[..., 4])
                y, p = w["target"].astype(float), q[..., 4].astype(float)
                sst = ((y-y.mean(axis=1, keepdims=True))**2).sum(axis=1)
                direct = np.full(len(y), np.nan)
                np.divide(((y-p)**2).sum(axis=1), sst, out=direct, where=sst > 0)
                np.testing.assert_allclose(r2, 1-direct, rtol=1e-10, atol=1e-10)
                cohorts = {"all_test": np.ones(len(y), bool)}
                if b in slide_starts:
                    cohorts["slide90"] = np.isin(w["forecast_start"], slide_starts[b])
                    assert cohorts["slide90"].sum() == 30
                for cohort, included in cohorts.items():
                    tags = dict(dataset=group, condition=condition, seed=seed, battery_id=b, cohort=cohort)
                    window_rows.extend(dict(**tags, forecast_start=int(s), phase=ph, r2=float(v))
                        for s, ph, v in zip(w["forecast_start"][included], phases[included], r2[included]))
                    for phase in ("all", *PHASES):
                        mask = included & (True if phase == "all" else phases == phase)
                        if mask.any():
                            battery_rows.append(dict(**tags, phase=phase, n_windows=int(mask.sum()),
                                **adaptation_scores(y[mask], q[mask])))
                if b in representative_ids:
                    idx = representative_indices(w, cycles)
                    for key in ("context", "context_cycle", "target", "target_cycle"):
                        reps[f"{b}_{key}"] = w[key][idx].copy()
                    reps[f"{b}_{condition}_{seed}"] = q[idx].copy()
            print(f"EVAL {group} {condition} seed={seed}: {len(records)} test cells", flush=True)
    restore(model, initial)
    battery, windows = pd.DataFrame(battery_rows), pd.DataFrame(window_rows)
    battery.to_csv(directory / "battery_metrics.csv", index=False)
    windows.to_csv(directory / "window_metrics.csv.gz", index=False)
    summarize_scores(battery, windows).to_csv(directory / "summary.csv", index=False)
    summarize_scores(battery, windows, by_seed=True).to_csv(directory / "seed_summary.csv", index=False)
    np.savez_compressed(directory / "representative_predictions.npz", **reps)
    draw_figures(group, representative_ids, reps, directory)
    write_json(directory / "evaluation.json", dict(status="complete", finished_at=now(),
        test_records=len(records), windows_per_condition=sum(len(r["windows"]["target"]) for r in records),
        slide_windows_per_condition=90, cached_baselines_checked=True, window_r2_directly_verified=True,
        model_choice_before_test=True, representative_ids=representative_ids, full_predictions_retained=False))


def draw_figures(group, ids, reps, directory):
    for window_set in ("original", "alternate"):
        fig, axes = plt.subplots(len(ids), 3, figsize=(15, 3.2*len(ids)), squeeze=False)
        for row, b in enumerate(ids):
            for col, title in enumerate(("First eligible", "Middle", "Late")):
                ax = axes[row, col]
                x = reps[f"{b}_target_cycle"][col]
                ax.plot(reps[f"{b}_context_cycle"][col], reps[f"{b}_context"][col], color=".65", lw=1.1, label="Past SOH")
                ax.plot(x, reps[f"{b}_target"][col], color="black", lw=1.7, label="Actual")
                for condition, color, label in (("zero-shot", "#2878b5", "Zero-shot"), ("previous_head20", "#999999", "Previous head")):
                    ax.plot(x, reps[f"{b}_{condition}_42"][col, :, 4], color=color, lw=1.3, label=label)
                for seed in SEEDS:
                    q = reps[f"{b}_optuna_{window_set}_{seed}"][col]
                    ax.plot(x, q[:, 4], color="#8e44ad", alpha=.85 if seed == 42 else .4, lw=1.2,
                            label="Optuna head (3 seeds)" if seed == 42 else None)
                    if seed == 42:
                        ax.fill_between(x, q[:, 0], q[:, -1], where=q[:, 0] <= q[:, -1], color="#8e44ad", alpha=.1)
                ax.axvline(x[0], color=".6", ls=":", lw=1)
                ax.set(title=f"Cell {b.rsplit('-',1)[1]} | {title} | {x[0]}-{x[-1]}", xlabel="Cycle", ylabel="SOH")
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"Chronos-2 {group} | head-only Optuna | {window_set} training windows\n"
                     "20 train / 10 validation; 64 -> 16; curves for all three seeds; shade: seed 42 raw 10-90% interval")
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(directory / f"forecast_{window_set}.png", dpi=150)
        plt.close(fig)


def render_report():
    tables = [pd.read_csv(OUT / g / "summary.csv") for g in GROUPS]
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(OUT / "summary.csv", index=False)
    lines = ["# Chronos-2 head-only: Optuna 20-window search", "",
        "XJTU 2C·3C에서 기존 학습 20창·검증 10창과 배터리 분할을 고정하고 학습률·최대 학습 길이만 조건별 16회 탐색했다. "
        "테스트를 보기 전에 두 조건의 탐색과 seed/학습창 확인 모델을 모두 고정했다. 기존 결과는 덮어쓰지 않았다.", "",
        "## 핵심 결과", "",
        "Optuna를 적용해도 두 조건이 모두 개선되지는 않았다. 아래 변화율은 전체 미학습 배터리의 모든 창 기준이며, "
        "새 모델은 seed 3개 평균, 기존 head는 seed 42 한 번의 결과다. 따라서 동일 seed 분포끼리의 엄밀한 우위 검증은 아니다.", "",
        "| 데이터 | Optuna: 순정 대비 RMSE 변화 | Optuna: 기존 head 대비 | 학습창을 바꾼 Optuna: 기존 head 대비 |",
        "|---|---:|---:|---:|"]
    for g in GROUPS:
        scores = combined[(combined.dataset == g) & (combined.cohort == "all_test") & (combined.phase == "all")].set_index("condition")
        zero, old, tuned, alternate = [scores.loc[c, "rmse"] for c in ("zero-shot", "previous_head20", "optuna_original", "optuna_alternate")]
        lines.append(f"| {g} | {tuned/zero-1:+.1%} | {tuned/old-1:+.1%} | {alternate/old-1:+.1%} |")
    lines += ["", "2C에서는 검증 손실 개선이 테스트 개선으로 이어지지 않았다. 3C에서는 기존 학습창의 세 seed 모두 기존 head보다 RMSE가 낮았지만, "
        "학습창을 바꾸면 추가 이득이 작아졌다. 현재 결과는 head-only의 보편적 개선이나 최적 설정을 입증하지 않는다. "
        "모든 테스트 예측에서 q10>q90이 없는지와 내부 분위수 역전율은 summary.csv에 별도로 보고한다.", "",
        "## 고정한 실험", "",
        "- 학습 배터리 4·5번 각 10창, 검증 6번 10창. SOH 단독, 64→16, 배치 4. 본체 고정, 출력 head 3.06%만 학습.",
        "- TPE 16 trials, 학습률 1e-7~3e-5 로그 탐색, 최대 100/200/400 steps. AdamW, weight decay 0, 선형 LR 감소, clip 1. "
        "20 steps마다 검증하며 순정도 후보다. 초기 후보 네 개와 sampler 설정은 protocol.json에 기록했다.",
        "- 검증 내부 분위수 역전율과 q10>q90 비율이 순정보다 늘지 않는 체크포인트 중 검증 분위수 손실 최소를 선택했다. "
        "Optuna 목적함수는 이 제약을 만족하는 best checkpoint 손실이다. 테스트 R²·RMSE는 선택에 쓰지 않았다.",
        "- 선택한 LR·최대 steps를 고정하고 seed 42/43/44를 모두 평가했다. seed 42 원래 창 결과는 선택된 trial을 재사용했다. "
        "각 seed의 best step은 같은 검증 규칙으로 고르며 seed를 골라 보고하지 않았다.",
        "- alternate는 학습 배터리의 비중복 label 후보를 수명 구간 10등분 후 위치만으로 하나씩 고른 별도 20창이다. "
        "검증 10창은 동일하고 Optuna 재탐색은 없다. 기존 20창 중 2C는 9창, 3C는 10창이 달라졌다. 같은 배터리에서 창만 바꾸므로 새로운 배터리 분할 검증은 아니다.",
        "- old head는 이전 LR 1e-6/100 steps/seed 42 기준이다. 새 head의 세 seed 평균과 기존 단일 seed를 구분한다.", "",
        "## 검증으로 선택한 설정", "",
        "| 데이터 | LR | 최대 steps | seed 42 선택 step | 검증 loss: 이전 head → Optuna | 탐색 학습·검증 시간 |",
        "|---|---:|---:|---:|---:|---:|"]
    for g in GROUPS:
        s = read(OUT / g / "selection.json")
        old = read(ROOT / f"results/c2-battery/fewshot-partial-tuning/{g}/run.json")["selected"]["SOH/head"]
        lines.append(f"| {g} | {s['learning_rate']:.3g} | {s['max_steps']} | {s['selected']['step']} | "
                     f"{old['validation_loss']:.4f} → {s['selected']['validation_loss']:.4f} | {s['search_seconds']:.1f}초 |")
    for cohort, title in (("all_test", "전체 미학습 배터리·전체 유효 창"), ("slide90", "기존 발표와 동일한 90창")):
        lines += ["", f"## {title}", "",
            "새 Optuna 결과는 seed별 점수 평균이며 RMSE ±는 seed간 표준편차다. 예측을 평균낸 앙상블 점수가 아니다.", "",
            "| 데이터 | 모델/학습창 | 창/seed | RMSE | R² 평균 | R² 중앙값 | R²<0 | WQL | PICP80 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in combined[(combined.cohort == cohort) & (combined.phase == "all")].itertuples():
            picp = f"{r.picp80:.1%}" if np.isfinite(r.picp80) else "N/A"
            rmse = f"{r.rmse:.6f}" + (f" ± {r.rmse_seed_std:.6f}" if r.n_seeds > 1 else "")
            lines.append(f"| {r.dataset} | {r.condition} | {r.n_windows_per_seed} / {r.n_seeds} | {rmse} | "
                         f"{r.r2_mean:.3f} | {r.r2_median:.3f} | {r.r2_negative_fraction:.1%} | {r.wql:.6f} | {picp} |")
    lines += ["", "## 해석 범위", "",
        "R²는 각 16개 target의 평균으로 1−SSE/SST를 계산한다. 배터리 내 창 평균 후 seed·배터리 동등 가중 평균한다. "
        "중앙값과 음수 비율도 같은 분포 가중치다. 상수창만 미정의, 음수는 유지한다. RMSE/WQL/PICP는 배터리별 계산 후 동일 가중 평균한다. "
        "분위수는 정렬·보정하지 않으며, q10>q90이 발생한 배터리의 PICP는 N/A이고 전체 평균에도 전파한다.", "",
        "학습창은 완료된 다른 배터리의 전체 이력에서 선택한다. target 초반 몇 라벨로 적응한 실험이 아니다. "
        "테스트에서는 매 cycle 실제 과거 SOH가 갱신되고 stride 1로 예측한다. 전 수명 재귀 예측이 아니며 겹치는 창은 독립 표본이 아니다. "
        "원래 test 배터리들을 이미 여러 번 확인한 탐색 실험이다. 세 seed와 같은 셀의 다른 학습창은 새 검증 배터리나 완전히 새로운 test의 대체가 아니다.", ""]
    for g in GROUPS:
        lines += [f"## {g} 그래프", "", f"![original]({g}/forecast_original.png)", "", f"![alternate]({g}/forecast_alternate.png)", ""]
    lines += ["세부 seed별 결과: 각 데이터셋의 seed_summary.csv. 모든 창별 R²: window_metrics.csv.gz. "
              "학습창: label_budget.csv. 탐색: trials.csv 및 trial별 validation.csv. 전체 예측 배열은 남기지 않고 대표 창 배열과 그래프를 보관했다."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    global PIPELINE
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prepared = {g: prepare(g) for g in GROUPS}
    protocol = dict(schema_version=1, groups=list(GROUPS), train_windows=20, validation_windows=10,
        context=64, horizon=16, stride=1, input="SOH only", trainable="output_patch_embedding only",
        trials_per_group=TRIALS, lr_range=[1e-7, 3e-5], maximum_steps=[100, 200, 400],
        enqueued_candidates=[[1e-6, 100], [1e-5, 100], [1e-6, 400], [3e-6, 200]],
        search_seed=42, confirmation_seeds=list(SEEDS), batch_size=4, validation_frequency=20,
        optimizer="AdamW weight_decay=0 eps=1e-8; linear LR decay over maximum steps; gradient clip 1",
        selection="Best validation quantile loss under unchanged crossing constraints; step zero included",
        confirmation="Lock LR/max_steps; validation-best step per seed; original and alternate training windows; all reported",
        alternate_rule="Stratified random non-overlapping label positions, seed 1042+cell number; same validation windows",
        pruning=False, sequential_trials=True, validation_refit=False, r2=R2_DEFINITION,
        model_id=prepared[GROUPS[0]]["config"].model_id, revision=prepared[GROUPS[0]]["config"].revision,
        versions=dict(optuna=optuna.__version__, torch=torch.__version__),
        splits={g: dict(**p["fold"], bundle_sha256=p["bundle_sha256"], label_budget_sha256=p["label_budget_sha256"],
                        previous_head_sha256=p["previous"]["selected_weights_sha256"]) for g, p in prepared.items()})
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise ValueError("Frozen protocol changed")
    write_json(protocol_path, protocol)
    if args.prepare_only:
        print("PREPARED: same 20 train / 10 validation windows; alternate train windows; all held-out cells", flush=True)
        return
    state_path = OUT / "status.json"
    if state_path.exists() and read(state_path).get("status") == "complete":
        print("ALREADY_COMPLETE", flush=True)
        return
    status = dict(status="loading", started_at=now(), groups_completed=[], protocol_sha256=sha256(protocol_path))
    write_json(state_path, status)
    try:
        PIPELINE = load_model(prepared[GROUPS[0]]["config"])
        model = PIPELINE.model
        names = configure_subset(model, "head")
        initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n in names}
        status["device"] = str(model.device)
        status["device_name"] = torch.xpu.get_device_name() if model.device.type == "xpu" else str(model.device)
        locked = {}
        for group in GROUPS:
            selection = search(group, prepared[group], model, initial, status)
            locked[group] = confirm(group, prepared[group], model, initial, selection, status)
        status.update(all_selections_locked_at=now(), frozen_backbone_verified=True)
        write_json(state_path, status)
        for group in GROUPS:
            evaluate(group, prepared[group], model, initial, locked[group], status)
            status["groups_completed"].append(group)
            write_json(state_path, status)
        render_report()
        status.update(status="complete", finished_at=now())
        write_json(state_path, status)
        print("OPTUNA_HEAD_COMPLETE", flush=True)
    except Exception as error:
        status.update(status="failed", error=repr(error), failed_at=now())
        write_json(state_path, status)
        raise


if __name__ == "__main__": main()

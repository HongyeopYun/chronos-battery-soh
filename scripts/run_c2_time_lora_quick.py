"""One fixed Chronos-2 time-attention LoRA setting; unchanged 20/10 label budget."""
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
from peft import LoraConfig, get_peft_model
from transformers import set_seed

from src.battery_fewshot_tuning import fixed_training_inputs, adaptation_scores, check_battery_split
from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows, sha256, write_json
from src.battery_model_zeroshot import load_model, predict_quantiles, phase_labels, PHASES
from src.c2_head_protocol import make_folds
from src.c2_head_search import indices_from_budget
from src.c2_partial_tuning import model_batch, corrected_loss, validation_scores, eligible_checkpoint, choose_checkpoint
from src.soh_evaluation import window_r2_values
from scripts.run_c2_head_optuna import summarize_scores, representative_indices, validation_batches, restore, sync

OUT = ROOT / "results/c2-battery/time-attention-lora-quick20"
WEIGHTS = ROOT / "models/c2-time-attention-lora-quick20"
GROUPS = ("xjtu_2c", "xjtu_3c")
LR, STEPS, SEED = 1e-5, 100, 42


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare(group, output_dir=None):
    source = read(ROOT / f"results/c2-battery/full/{group}/c2-battery_run.json")
    bundle = Path(source["bundle"])
    assert sha256(bundle / "manifest.json") == source["bundle_fingerprint"]
    config = replace(ZeroShotConfig(**source["config"]), battery_ids=None, max_windows_per_battery=None)
    records, _, data, _ = prepare_battery_windows(bundle, config)
    fold = make_folds(group, records)[0]
    check_battery_split(fold["train_ids"], fold["validation_ids"], fold["test_ids"])
    by_id = {r["battery_id"]: r for r in records}
    budget = pd.read_csv(ROOT / f"results/c2-battery/fewshot-partial-tuning/{group}/label_budget.csv")
    tasks = {"train": [], "validation": []}
    for split in tasks:
        for b in fold[f"{split}_ids"]:
            w = by_id[b]["windows"]
            rows = budget[(budget.battery_id == b) & (budget.split == split)]
            indices = indices_from_budget(w, rows)
            assert len(indices) == 10
            tasks[split].extend(fixed_training_inputs(w, pd.DataFrame({"cycle": []}), indices))
    assert len(tasks["train"]) == 20 and len(tasks["validation"]) == 10
    directory = (OUT if output_dir is None else output_dir) / group
    directory.mkdir(parents=True, exist_ok=True)
    budget.to_csv(directory / "label_budget.csv", index=False)
    return dict(config=config, records=by_id, data=data, fold=fold, tasks=tasks)


def train(group, prepared, model, initial, status):
    restore(model, initial)
    model.eval()  # Frozen backbone dropout off; LoRA dropout is zero.
    set_seed(SEED)
    params = dict(model.named_parameters())
    names = [n for n, p in params.items() if p.requires_grad]
    assert set(names) == set(initial)
    assert all(".layer.0.self_attention." in n and ".lora_" in n for n in names)
    frozen = {n: p._version for n, p in params.items() if not p.requires_grad}
    batches = validation_batches(model, prepared["tasks"]["validation"])
    baseline = dict(step=0, learning_rate=0., eligible=True, **validation_scores(model, batches))
    with model.disable_adapter():
        base_scores = validation_scores(model, batches)
    np.testing.assert_allclose(baseline["validation_loss"], base_scores["validation_loss"], rtol=1e-6)
    optimizer = torch.optim.AdamW([params[n] for n in names], lr=LR, weight_decay=0., eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: max(0., 1-s/STEPS))
    iterator = iter(Chronos2Dataset(inputs=prepared["tasks"]["train"], context_length=64,
        prediction_length=16, batch_size=4, output_patch_size=16, min_past=64, mode=DatasetMode.TRAIN))
    history, winner, weights = [baseline], baseline, {n: v.clone() for n, v in initial.items()}
    sync(model)
    start = time.perf_counter()
    print(f"TRAIN {group}: rank 8, lr={LR}, steps={STEPS}, 20 train / 10 validation", flush=True)
    for step in range(1, STEPS+1):
        batch = model_batch(next(iterator), model.device)
        assert batch["context"].shape == (4, 64)
        assert torch.isfinite(batch["future_target"]).sum().item() == 64
        assert torch.isnan(batch["future_covariates"]).all()
        assert torch.unique(batch["group_ids"]).numel() == 4
        optimizer.zero_grad(set_to_none=True)
        loss = corrected_loss(model, batch)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([params[n] for n in names], 1.)
        optimizer.step()
        scheduler.step()
        if step % 20 == 0:
            row = dict(step=step, learning_rate=LR, **validation_scores(model, batches))
            row["eligible"] = eligible_checkpoint(row, baseline)
            history.append(row)
            if choose_checkpoint(history, baseline) is row:
                winner, weights = row, {n: params[n].detach().cpu().clone() for n in names}
            status.update(status="training", group=group, step=step)
            write_json(OUT / "status.json", status)
            print(f"  {step}: validation={row['validation_loss']:.6f}, eligible={row['eligible']}", flush=True)
    sync(model)
    elapsed = time.perf_counter()-start
    assert all(params[n]._version == version and params[n].grad is None for n, version in frozen.items())
    assert any(not torch.equal(params[n].detach().cpu(), initial[n]) for n in names)
    path = WEIGHTS / group / "selected.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, path)
    torch.save({n: params[n].detach().cpu() for n in names}, path.with_name("step100.pt"))
    pd.DataFrame(history).to_csv(OUT / group / "validation.csv", index=False)
    selected = dict(**winner, requested_learning_rate=LR, max_steps=STEPS, seed=SEED,
        training_validation_seconds=elapsed, weights_path=str(path), weights_sha256=sha256(path),
        frozen_parameters_verified=True, adapter_updated=True, selection_locked_at=now(), baseline=baseline)
    write_json(OUT / group / "selection.json", selected)
    print(f"SELECTED {group} step={winner['step']}, seconds={elapsed:.1f}", flush=True)
    return selected


def evaluate(group, prepared, pipeline, selected, status):
    model, directory = pipeline.model, OUT / group
    restore(model, torch.load(selected["weights_path"], map_location="cpu", weights_only=True))
    model.eval()
    test_ids = prepared["fold"]["test_ids"]
    representative_ids = [test_ids[i] for i in np.linspace(0, len(test_ids)-1, min(4, len(test_ids)), dtype=int)]
    rows, windows, reps = [], [], {}
    with np.load(ROOT / f"results/c2-battery/fewshot/{group}/c2-battery_{group}_fewshot_predictions.npz") as cache:
        slide_starts = {b: cache[f"{b}_forecast_start"].copy() for b in test_ids if f"{b}_forecast_start" in cache}
    with np.load(ROOT / f"results/c2-battery/head-only-xjtu-all-test/{group}/predictions.npz") as cache:
        for b in test_ids:
            w = prepared["records"][b]["windows"]
            np.testing.assert_array_equal(cache[f"{b}_target"], w["target"])
            np.testing.assert_array_equal(cache[f"{b}_forecast_start"], w["forecast_start"])
            status.update(status="evaluating", group=group, battery=b)
            write_json(OUT / "status.json", status)
            predictions = {"zero-shot": cache[f"{b}_condition0_quantiles"],
                           "previous_head20": cache[f"{b}_condition1_quantiles"],
                           "time_attention_lora": predict_quantiles(pipeline, w["context"], prepared["config"])[1]}
            with model.disable_adapter():
                zero_check = predict_quantiles(pipeline, w["context"][:1], prepared["config"])[1]
            np.testing.assert_allclose(zero_check, predictions["zero-shot"][:1], rtol=2e-5, atol=1e-7)
            cycles = prepared["data"].loc[(prepared["data"].battery_id == b) & prepared["data"].is_observed, "cycle"].to_numpy()
            phases = phase_labels(w["forecast_start"], cycles)
            for condition, q in predictions.items():
                y, pred = w["target"].astype(float), q[..., 4].astype(float)
                r2 = window_r2_values(y, pred)
                sst = ((y-y.mean(axis=1, keepdims=True))**2).sum(axis=1)
                direct = np.full(len(y), np.nan)
                np.divide(((y-pred)**2).sum(axis=1), sst, out=direct, where=sst>0)
                np.testing.assert_allclose(r2, 1-direct, rtol=1e-10, atol=1e-10)
                cohorts = {"all_test": np.ones(len(y), dtype=bool)}
                if b in slide_starts:
                    cohorts["slide90"] = np.isin(w["forecast_start"], slide_starts[b])
                    assert cohorts["slide90"].sum() == 30
                for cohort, included in cohorts.items():
                    tags = dict(dataset=group, condition=condition, seed=SEED, battery_id=b, cohort=cohort)
                    windows.extend(dict(**tags, forecast_start=int(s), phase=ph, r2=float(v))
                        for s, ph, v in zip(w["forecast_start"][included], phases[included], r2[included]))
                    for phase in ("all", *PHASES):
                        mask = included & (True if phase == "all" else phases == phase)
                        if mask.any():
                            rows.append(dict(**tags, phase=phase, n_windows=int(mask.sum()), **adaptation_scores(y[mask], q[mask])))
                if b in representative_ids:
                    idx = representative_indices(w, cycles)
                    reps[f"{b}_{condition}"] = q[idx].copy()
                    for key in ("context", "context_cycle", "target", "target_cycle"):
                        reps[f"{b}_{key}"] = w[key][idx].copy()
            print(f"EVAL {group} {b}: {len(w['target'])} windows", flush=True)
    battery, windows = pd.DataFrame(rows), pd.DataFrame(windows)
    battery.to_csv(directory / "battery_metrics.csv", index=False)
    windows.to_csv(directory / "window_metrics.csv.gz", index=False)
    summary = summarize_scores(battery, windows)
    summary.to_csv(directory / "summary.csv", index=False)
    np.savez_compressed(directory / "representative_predictions.npz", **reps)
    fig, axes = plt.subplots(len(representative_ids), 3, figsize=(15, 3.2*len(representative_ids)), squeeze=False)
    for row, b in enumerate(representative_ids):
        for col, title in enumerate(("First eligible", "Middle", "Late")):
            ax, x = axes[row, col], reps[f"{b}_target_cycle"][col]
            ax.plot(reps[f"{b}_context_cycle"][col], reps[f"{b}_context"][col], color=".65", lw=1, label="Past SOH")
            ax.plot(x, reps[f"{b}_target"][col], color="black", lw=1.8, label="Actual")
            for condition, color, label in (("zero-shot", "#2878b5", "Zero-shot"),
                    ("previous_head20", "#999999", "Head-only (previous)"), ("time_attention_lora", "#c03b76", "Time-attention LoRA")):
                q = reps[f"{b}_{condition}"][col]
                ax.plot(x, q[:, 4], color=color, lw=1.5, label=label)
                if condition == "time_attention_lora":
                    ax.fill_between(x, q[:, 0], q[:, -1], where=q[:, 0]<=q[:, -1], color=color, alpha=.12)
            ax.axvline(x[0], color=".6", ls=":", lw=1)
            ax.set(title=f"Cell {b.rsplit('-',1)[1]} | {title} | {x[0]}-{x[-1]}", xlabel="Cycle", ylabel="SOH")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Chronos-2 {group} | Time-attention LoRA: rank 8, LR 1e-5, max 100 steps\n"
                 f"20 train / 10 validation windows | selected step {selected['step']} | 64 -> 16 | seed 42")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(directory / "forecast.png", dpi=150)
    plt.close(fig)
    return summary


def main():
    if (OUT / "status.json").exists():
        raise ValueError("Existing quick run preserved; choose a new output directory to rerun")
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    status = dict(status="preparing", started_at=now())
    write_json(OUT / "status.json", status)
    try:
        prepared = {g: prepare(g) for g in GROUPS}
        pipeline = load_model(prepared[GROUPS[0]]["config"])
        base = pipeline.model
        targets = [n for n, m in base.named_modules() if re.fullmatch(r"encoder\.block\.\d+\.layer\.0\.self_attention\.[qkvo]", n)]
        assert len(targets) == 48 and all(isinstance(dict(base.named_modules())[n], torch.nn.Linear) for n in targets)
        set_seed(SEED)
        pipeline.model = model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
            target_modules=targets, bias="none"))
        initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        trainable = sum(t.numel() for t in initial.values())
        total = sum(p.numel() for p in model.parameters())
        assert trainable == 589824 and len(initial) == 96
        protocol = dict(groups=list(GROUPS), train_windows=20, validation_windows=10, context=64, horizon=16,
            stride=1, input="SOH only", cross_learning=False, rank=8, alpha=16, lora_dropout=0.,
            learning_rate=LR, maximum_steps=STEPS, seed=SEED, batch_size=4, validation_frequency=20,
            target_modules=targets, trainable_parameters=trainable, total_parameters_with_adapter=total,
            trainable_fraction=trainable/total, model_id=prepared[GROUPS[0]]["config"].model_id,
            revision=prepared[GROUPS[0]]["config"].revision,
            optimizer="AdamW weight_decay=0 eps=1e-8, linear LR decay, gradient clip 1",
            backbone_mode="eval: frozen base dropout disabled; adapter dropout zero",
            selection="Minimum validation quantile loss under existing quantile-crossing guard, including step 0; no refit",
            scope="Single-setting single-seed exploratory offline other-cell adaptation; source histories span lifetime",
            splits={g: p["fold"] for g, p in prepared.items()})
        write_json(OUT / "protocol.json", protocol)
        status.update(status="training", device=str(model.device),
            device_name=torch.xpu.get_device_name() if model.device.type == "xpu" else str(model.device))
        print(f"DEVICE {status['device_name']}; trainable {trainable}/{total} ({trainable/total:.3%})", flush=True)
        selected = {g: train(g, prepared[g], model, initial, status) for g in GROUPS}
        status.update(all_selections_locked_at=now())
        write_json(OUT / "status.json", status)
        tables = [evaluate(g, prepared[g], pipeline, selected[g], status) for g in GROUPS]
        combined = pd.concat(tables, ignore_index=True)
        combined.to_csv(OUT / "summary.csv", index=False)
        lines = ["# C2 시간 Attention LoRA: 단일 설정 빠른 실험", "",
            "시간 Attention Q/K/V/O에만 rank 8, alpha 16 LoRA. 학습률 1e-5, 최대 100 step, seed 42. "
            "학습 20창·검증 10창과 분할은 이전 실험과 동일. 출력 head·그룹 Attention·FFN 등 기존 가중치는 고정했다.", "",
            "학습: 4·5번 각 10창, 검증: 6번 10창. 두 조건의 체크포인트를 검증으로 확정한 뒤 모든 미학습 배터리를 평가했다. "
            "순정 step 0도 선택 후보이며, 분위수 역전율이 순정보다 늘어나는 체크포인트는 기존 규칙대로 제외했다.", ""]
        for g in GROUPS:
            s = selected[g]
            lines.append(f"- {g}: 선택 step {s['step']}, 검증 loss {s['baseline']['validation_loss']:.4f} → {s['validation_loss']:.4f}; 학습·검증 {s['training_validation_seconds']:.1f}초.")
        for cohort in ("all_test", "slide90"):
            lines += ["", f"## {cohort}", "", "| 조건 | 모델 | RMSE | R² 평균 | R² 중앙값 | 음수 창 | WQL | PICP80 |",
                "|---|---|---:|---:|---:|---:|---:|---:|"]
            for r in combined[(combined.cohort==cohort) & (combined.phase=="all")].itertuples():
                lines.append(f"| {r.dataset} | {r.condition} | {r.rmse:.6f} | {r.r2_mean:.3f} | {r.r2_median:.3f} | {r.r2_negative_fraction:.1%} | {r.wql:.6f} | {r.picp80:.1%} |")
        lines += ["", "R²는 각 16-cycle 예측창 안에서 표준 1−SSE/SST로 계산 후 배터리 내 창 평균·배터리 간 동등 가중 평균. "
            "중앙값·음수 비율도 배터리 동등 가중 분포이다. RMSE/WQL/PICP도 배터리 동등 가중이며 상수 창 R²는 미정의. "
            "q10>q90 발생 시 PICP는 N/A이며 분위수는 보정하지 않았다.", "",
            "단일 설정·단일 seed의 탐색 결과로 LoRA의 최적 성능이나 일반적 우위를 입증하지 않는다. "
            "20창은 미래 라벨 320개와 과거 context를 사용한다. 다른 배터리의 완료된 이력을 이용한 적응이며 대상 배터리 초반만 쓰는 설정이 아니다. "
            "테스트는 실제 과거 SOH를 갱신하는 rolling 64→16, stride 1; 겹치는 창은 독립 표본이 아니다."]
        for g in GROUPS:
            lines += ["", f"![{g}]({g}/forecast.png)"]
        (OUT / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        status.update(status="complete", finished_at=now(), total_seconds=time.perf_counter()-start,
            frozen_parameters_verified=True, cached_zero_shot_verified=True, direct_window_r2_verified=True,
            test_windows_per_condition=int(combined[(combined.cohort=="all_test") & (combined.phase=="all") &
                (combined.condition=="time_attention_lora")].n_windows_per_seed.sum()))
        write_json(OUT / "status.json", status)
        print(combined[(combined.cohort=="all_test") & (combined.phase=="all")][
            ["dataset", "condition", "rmse", "r2_mean", "r2_median", "r2_negative_fraction"]].to_string(index=False), flush=True)
        print("TIME_ATTENTION_LORA_COMPLETE", flush=True)
    except Exception as error:
        status.update(status="failed", error=repr(error), failed_at=now())
        write_json(OUT / "status.json", status)
        raise


if __name__ == "__main__":
    main()

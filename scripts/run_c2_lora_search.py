"""Bounded validation-only PEFT search, seed confirmation, then locked test evaluation."""
import argparse
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
import pandas as pd
import torch
from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
from transformers import set_seed

from scripts.run_c2_time_lora_quick import prepare, now, read
from scripts.run_c2_head_optuna import validation_batches, sync, summarize_scores, representative_indices
from src.c2_lora_tuning import (MODES, attach_lora, reset_lora, configure_lora, adapter_state,
    load_adapter_state, frozen_state_fingerprint)
from src.c2_partial_tuning import model_batch, corrected_loss, validation_scores, eligible_checkpoint, choose_checkpoint
from src.battery_zeroshot import write_json, sha256
from src.battery_model_zeroshot import load_model, predict_quantiles, phase_labels, PHASES
from src.battery_fewshot_tuning import adaptation_scores
from src.soh_evaluation import window_r2_values, R2_DEFINITION

OUT = ROOT / "results/c2-battery/lora-search20"
WEIGHTS = ROOT / "models/c2-lora-search20"
GROUPS = ("xjtu_2c", "xjtu_3c")
LRS = (1e-5, 3e-5, 1e-4)
STEPS = 400
SEEDS = (42, 43, 44)


@torch.no_grad()
def validate(model, batches):
    scores = validation_scores(model, batches)
    targets, medians = [], []
    qi = int(torch.argmin(abs(model.quantiles-.5)))
    for batch in batches:
        targets.append(batch["future_target"].cpu().numpy())
        medians.append(model(**batch).quantile_preds[:, qi, :16].cpu().numpy())
    y, p = np.concatenate(targets).astype(float), np.concatenate(medians).astype(float)
    scores.update(validation_rmse=float(np.sqrt(np.mean((y-p)**2))),
        validation_r2_mean=float(np.nanmean(window_r2_values(y, p))))
    return scores


def train_candidate(group, prepared, model, mode, lr, seed, candidate_id, status):
    directory = OUT / group / candidate_id
    selected_path = directory / "selected.json"
    if selected_path.exists():
        selected = read(selected_path)
        assert (selected["mode"], selected["requested_learning_rate"], selected["seed"], selected["max_steps"]) == (mode, lr, seed, STEPS)
        assert sha256(Path(selected["weights_path"])) == selected["weights_sha256"]
        return selected
    directory.mkdir(parents=True, exist_ok=True)
    reset_lora(model, seed)
    names = configure_lora(model, mode)
    params = dict(model.named_parameters())
    frozen_versions = {n: p._version for n, p in params.items() if n not in names}
    initial = adapter_state(model)
    batches = validation_batches(model, prepared["tasks"]["validation"])
    baseline = dict(step=0, learning_rate=0., eligible=True, **validate(model, batches))
    quick = read(ROOT / f"results/c2-battery/time-attention-lora-quick20/{group}/selection.json")["baseline"]
    np.testing.assert_allclose(baseline["validation_loss"], quick["validation_loss"], rtol=2e-6)
    set_seed(seed)
    optimizer = torch.optim.AdamW([params[n] for n in names], lr=lr, weight_decay=0., eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: max(0., 1-s/STEPS))
    iterator = iter(Chronos2Dataset(inputs=prepared["tasks"]["train"], context_length=64,
        prediction_length=16, batch_size=4, output_patch_size=16, min_past=64, mode=DatasetMode.TRAIN))
    history, winner, weights = [baseline], baseline, initial
    sync(model)
    started = time.perf_counter()
    print(f"TRAIN {group} {candidate_id}: {mode}, lr={lr}, seed={seed}", flush=True)
    for step in range(1, STEPS+1):
        batch = model_batch(next(iterator), model.device)
        assert batch["context"].shape == (4,64) and torch.isfinite(batch["future_target"]).sum().item() == 64
        assert torch.isnan(batch["future_covariates"]).all() and torch.unique(batch["group_ids"]).numel() == 4
        optimizer.zero_grad(set_to_none=True)
        loss = corrected_loss(model, batch)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([params[n] for n in names], 1.)
        optimizer.step()
        scheduler.step()
        if step % 20 == 0:
            row = dict(step=step, learning_rate=lr, train_loss_last=float(loss.detach()), **validate(model, batches))
            row["eligible"] = eligible_checkpoint(row, baseline)
            history.append(row)
            if choose_checkpoint(history, baseline) is row:
                winner, weights = row, adapter_state(model)
            pd.DataFrame(history).to_csv(directory / "validation.csv", index=False)
            status.update(status="training", group=group, candidate=candidate_id, step=step,
                best_validation_loss=winner["validation_loss"])
            write_json(OUT / "status.json", status)
            if step % 100 == 0:
                print(f"  {step}/{STEPS} loss={row['validation_loss']:.5f}, best={winner['validation_loss']:.5f}, eligible={row['eligible']}", flush=True)
    sync(model)
    elapsed = time.perf_counter()-started
    assert all(params[n]._version == version and params[n].grad is None for n, version in frozen_versions.items())
    assert any(not torch.equal(params[n].detach().cpu(), initial[n]) for n in names)
    weight_path = WEIGHTS / group / f"{candidate_id}.pt"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, weight_path)
    result = dict(**winner, mode=mode, candidate_id=candidate_id, requested_learning_rate=lr, max_steps=STEPS,
        seed=seed, training_validation_seconds=elapsed, weights_path=str(weight_path), weights_sha256=sha256(weight_path),
        trainable_parameters=sum(params[n].numel() for n in names),
        total_parameters=sum(p.numel() for p in params.values()), frozen_parameters_verified=True,
        best_unconstrained_loss=min(r["validation_loss"] for r in history),
        eligible_checkpoints=sum(r["eligible"] for r in history[1:]), locked_at=now(), baseline=baseline)
    write_json(selected_path, result)
    print(f"SELECT {group} {candidate_id}: step {winner['step']}, loss {winner['validation_loss']:.5f}, {elapsed:.1f}s", flush=True)
    return result


def search_and_confirm(prepared, model, status):
    outcomes = {}
    for group in GROUPS:
        candidates = [train_candidate(group, prepared[group], model, mode, lr, 42,
            f"{mode}_lr{lr:.0e}_seed42", status) for mode in MODES for lr in LRS]
        # Select by validation only, with a smaller model and earlier step as tie breakers.
        winner = min(candidates, key=lambda s: (s["validation_loss"], s["trainable_parameters"], s["step"], s["requested_learning_rate"]))
        pd.DataFrame([{k: v for k, v in s.items() if k != "baseline"} for s in candidates]).to_csv(OUT / group / "trials.csv", index=False)
        write_json(OUT / group / "selection.json", winner)
        confirmed = [winner]
        for seed in SEEDS[1:]:
            confirmed.append(train_candidate(group, prepared[group], model, winner["mode"],
                winner["requested_learning_rate"], seed, f"confirmation_seed{seed}", status))
        outcomes[group] = confirmed
        write_json(OUT / group / "confirmations.json", confirmed)
    return outcomes


def score_forecast(group, b, w, q, phases, slide_starts, condition, seed, battery_rows, window_rows):
    y, p = w["target"].astype(float), q[...,4].astype(float)
    r2 = window_r2_values(y, p)
    sst = ((y-y.mean(axis=1, keepdims=True))**2).sum(axis=1)
    direct = np.full(len(y), np.nan)
    np.divide(((y-p)**2).sum(axis=1), sst, out=direct, where=sst>0)
    np.testing.assert_allclose(r2, 1-direct, rtol=1e-10, atol=1e-10)
    cohorts = {"all_test": np.ones(len(y), dtype=bool)}
    if b in slide_starts:
        cohorts["slide90"] = np.isin(w["forecast_start"], slide_starts[b])
        assert cohorts["slide90"].sum() == 30
    for cohort, included in cohorts.items():
        tags = dict(dataset=group, condition=condition, seed=seed, battery_id=b, cohort=cohort)
        window_rows.extend(dict(**tags, forecast_start=int(s), phase=ph, r2=float(v))
            for s, ph, v in zip(w["forecast_start"][included], phases[included], r2[included]))
        for phase in ("all", *PHASES):
            mask = included & (True if phase=="all" else phases==phase)
            if mask.any():
                battery_rows.append(dict(**tags, phase=phase, n_windows=int(mask.sum()), **adaptation_scores(y[mask],q[mask])))


def evaluate(group, prepared, pipeline, outcomes, status):
    directory, model = OUT / group, pipeline.model
    if (directory / "evaluation.json").exists():
        return
    battery, windows, reps = [], [], {}
    test_ids = prepared["fold"]["test_ids"]
    rep_ids = [test_ids[i] for i in np.linspace(0,len(test_ids)-1,min(4,len(test_ids)),dtype=int)]
    with np.load(ROOT / f"results/c2-battery/fewshot/{group}/c2-battery_{group}_fewshot_predictions.npz") as cache:
        slide_starts = {b: cache[f"{b}_forecast_start"].copy() for b in test_ids if f"{b}_forecast_start" in cache}
    conditions = [("zero-shot",42,None), ("previous_head20",42,None),
                  *[("selected_lora",s["seed"],s) for s in outcomes]]
    start = time.perf_counter()
    with np.load(ROOT / f"results/c2-battery/head-only-xjtu-all-test/{group}/predictions.npz") as cache:
        for condition, seed, selected in conditions:
            if selected:
                assert sha256(Path(selected["weights_path"])) == selected["weights_sha256"]
                load_adapter_state(model, torch.load(selected["weights_path"],map_location="cpu",weights_only=True))
            model.eval()
            for b in test_ids:
                w = prepared["records"][b]["windows"]
                np.testing.assert_array_equal(w["target"],cache[f"{b}_target"])
                np.testing.assert_array_equal(w["forecast_start"],cache[f"{b}_forecast_start"])
                status.update(status="evaluating", group=group, condition=condition, seed=seed, battery=b)
                write_json(OUT / "status.json",status)
                if selected:
                    q = predict_quantiles(pipeline,w["context"],prepared["config"])[1]
                else:
                    q = cache[f"{b}_condition{0 if condition=='zero-shot' else 1}_quantiles"].copy()
                    if condition=="zero-shot":
                        with model.disable_adapter():
                            actual = predict_quantiles(pipeline,w["context"][:1],prepared["config"])[1]
                        np.testing.assert_allclose(q[:1],actual,rtol=2e-5,atol=1e-7)
                cycles = prepared["data"].loc[(prepared["data"].battery_id==b)&prepared["data"].is_observed,"cycle"].to_numpy()
                phases = phase_labels(w["forecast_start"],cycles)
                score_forecast(group,b,w,q,phases,slide_starts,condition,seed,battery,windows)
                if b in rep_ids:
                    idx=representative_indices(w,cycles)
                    reps[f"{b}_{condition}_{seed}"]=q[idx].copy()
                    for key in ("context","context_cycle","target","target_cycle"):
                        reps[f"{b}_{key}"]=w[key][idx].copy()
            print(f"EVAL {group} {condition} seed={seed} complete",flush=True)
    # Add the already locked Optuna head on exactly the same cells and forecast origins.
    olddir=ROOT / f"results/c2-battery/head-only-optuna20/{group}"
    oldb=pd.read_csv(olddir/"battery_metrics.csv")
    oldw=pd.read_csv(olddir/"window_metrics.csv.gz")
    oldb=oldb[oldb.condition=="optuna_original"].copy()
    oldw=oldw[oldw.condition=="optuna_original"].copy()
    oldb["condition"]=oldw["condition"]="optuna_head"
    neww=pd.DataFrame(windows)
    for seed in SEEDS:
        a=neww[(neww.condition=="selected_lora")&(neww.seed==seed)][["cohort","battery_id","forecast_start"]].sort_values(["cohort","battery_id","forecast_start"]).reset_index(drop=True)
        b=oldw[oldw.seed==seed][["cohort","battery_id","forecast_start"]].sort_values(["cohort","battery_id","forecast_start"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b)
    battery=pd.concat([pd.DataFrame(battery),oldb],ignore_index=True)
    windows=pd.concat([neww,oldw],ignore_index=True)
    battery.to_csv(directory/"battery_metrics.csv",index=False)
    windows.to_csv(directory/"window_metrics.csv.gz",index=False)
    summarize_scores(battery,windows).to_csv(directory/"summary.csv",index=False)
    summarize_scores(battery,windows,by_seed=True).to_csv(directory/"seed_summary.csv",index=False)
    np.savez_compressed(directory/"representative_predictions.npz",**reps)
    with np.load(olddir/"representative_predictions.npz") as oldreps:
        for b in rep_ids:
            np.testing.assert_array_equal(reps[f"{b}_target_cycle"],oldreps[f"{b}_target_cycle"])
            reps[f"{b}_optuna_head_42"]=oldreps[f"{b}_optuna_original_42"].copy()
    draw_figures(group,rep_ids,reps,outcomes)
    write_json(directory/"evaluation.json",dict(status="complete",finished_at=now(),
        inference_scoring_plot_seconds=time.perf_counter()-start,test_batteries=len(test_ids),
        windows_per_condition_per_seed=sum(len(prepared["records"][b]["windows"]["target"]) for b in test_ids),
        baseline_verified=True,matched_optuna_windows=True,direct_window_r2_verified=True))


def draw_figures(group,ids,reps,outcomes):
    fig,axes=plt.subplots(len(ids),3,figsize=(15,3.2*len(ids)),squeeze=False)
    for row,b in enumerate(ids):
        for col,title in enumerate(("First eligible","Middle","Late")):
            ax,x=axes[row,col],reps[f"{b}_target_cycle"][col]
            ax.plot(reps[f"{b}_context_cycle"][col],reps[f"{b}_context"][col],color=".65",lw=1,label="Past SOH")
            ax.plot(x,reps[f"{b}_target"][col],color="black",lw=1.8,label="Actual")
            for condition,color,label in (("zero-shot","#2878b5","Zero-shot"),("optuna_head","#d38b20","Optuna head (seed 42)")):
                ax.plot(x,reps[f"{b}_{condition}_42"][col,:,4],color=color,lw=1.4,label=label)
            for seed in SEEDS:
                q=reps[f"{b}_selected_lora_{seed}"][col]
                ax.plot(x,q[:,4],color="#a43b79",alpha=.95 if seed==42 else .4,lw=1.4,label="Selected LoRA (3 seeds)" if seed==42 else None)
                if seed==42:
                    ax.fill_between(x,q[:,0],q[:,-1],where=q[:,0]<=q[:,-1],color="#a43b79",alpha=.1)
            ax.axvline(x[0],color=".6",ls=":",lw=1)
            ax.set(title=f"Cell {b.rsplit('-',1)[1]} | {title} | {x[0]}-{x[-1]}",xlabel="Cycle",ylabel="SOH")
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    choice=outcomes[0]
    fig.suptitle(f"Chronos-2 {group} | {choice['mode']} LoRA, rank 8, LR {choice['requested_learning_rate']:.0e}\n"
        "20 train / 10 validation | 64 -> 16 | three seeds, no test-based selection")
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(OUT/group/"forecast.png",dpi=150)
    plt.close(fig)


def report():
    summary=pd.concat([pd.read_csv(OUT/g/"summary.csv") for g in GROUPS],ignore_index=True)
    summary.to_csv(OUT/"summary.csv",index=False)
    lines=["# C2 LoRA 개선 실험: 고정 20창", "",
        "시간 Attention Q/K/V/O만 조정하는 LoRA와, 출력 head의 세 선형층에도 LoRA를 추가하는 방식을 비교했다. "
        "rank 8, alpha 16, LR 1e-5/3e-5/1e-4, 최대 400 step. 배터리별 조건 6개, 검증 분위수 손실로만 선택 후 seed 42/43/44 확인.", "",
        "학습 4·5번 각 10창, 검증 6번 10창. 본체 원래 가중치는 전부 고정했고 해시로 검증했다. "
        "모든 설정·체크포인트는 테스트 전에 확정했다. 선택 기준은 기존과 같은 정규화 분위수 손실이며 "
        "검증 분위수 역전율이 순정보다 늘어나는 체크포인트는 제외, step 0 포함. Raw SOH 검증 RMSE는 진단용으로만 기록.", "",
        "| 조건 | 선택 구조 | LR | seed별 선택 step | 학습 파라미터 |", "|---|---|---:|---|---:|"]
    for g in GROUPS:
        outcomes=read(OUT/g/"confirmations.json")
        s=outcomes[0]
        lines.append(f"| {g} | {s['mode']} | {s['requested_learning_rate']:.0e} | {', '.join(str(r['step']) for r in outcomes)} | {s['trainable_parameters']:,} |")
    for cohort in ("all_test","slide90"):
        lines += ["",f"## {cohort}","","| 조건 | 모델 | RMSE | R² 평균 | R² 중앙값 | R²<0 | WQL | PICP80 |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
        for r in summary[(summary.cohort==cohort)&(summary.phase=="all")].itertuples():
            lines.append(f"| {r.dataset} | {r.condition} ({r.n_seeds} seeds) | {r.rmse:.6f} ± {r.rmse_seed_std:.6f} | {r.r2_mean:.3f} | {r.r2_median:.3f} | {r.r2_negative_fraction:.1%} | {r.wql:.6f} | {r.picp80:.1%} |")
    lines += ["", "Optuna head와 selected LoRA는 각각 동일 seed 3개의 평균이다. 예측 앙상블이 아니며 seed 선택은 없다. "
        "R²는 창별 표준 1−SSE/SST 후 배터리·seed 동등 가중으로 집계하며 중앙값·음수 비율도 같은 분포 가중치다. "
        "RMSE/WQL/PICP도 배터리·seed 동등 가중. 음수 R²는 유지하며 q10>q90이면 PICP는 N/A로 전파한다.", "",
        "이번에도 완료된 다른 배터리 이력에서 라벨 20창을 택했다. 대상 배터리 초반만 사용하는 설정이 아니다. "
        "세 seed는 새로운 배터리 분할 검증을 대신하지 않는다. 기존 테스트 결과를 이미 반복해서 확인한 탐색 연구이므로 "
        "새 데이터셋에 대한 확정적 일반화 주장으로 해석할 수 없다. 학습 라벨 수·관측 길이·평가 창·지표 공식은 변경하지 않았다."]
    for g in GROUPS:
        lines += ["",f"![{g}]({g}/forecast.png)"]
    (OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(summary[(summary.cohort=="all_test")&(summary.phase=="all")][["dataset","condition","rmse","r2_mean","r2_median"]].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--search-only",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"status.json").exists() and read(OUT/"status.json")["status"]=="complete":
        print("ALREADY_COMPLETE")
        return
    status=dict(status="preparing",started_at=now())
    write_json(OUT/"status.json",status)
    try:
        prepared={g:prepare(g,OUT) for g in GROUPS}
        protocol=dict(groups=list(GROUPS),modes=list(MODES),rank=8,alpha=16,lora_dropout=0.,learning_rates=list(LRS),
            max_steps=STEPS,validation_frequency=20,train_windows=20,validation_windows=10,context=64,horizon=16,
            stride=1,batch_size=4,search_seed=42,confirmation_seeds=list(SEEDS),cross_learning=False,
            optimizer="AdamW weight_decay=0 eps=1e-8, linear decay, clip 1, frozen backbone eval",
            selection="Minimum validation normalized quantile loss under unchanged crossing guard; step0 allowed",
            raw_validation_rmse="Diagnostic only; never selects checkpoints",validation_refit=False,
            r2=R2_DEFINITION,model_id=prepared[GROUPS[0]]["config"].model_id,revision=prepared[GROUPS[0]]["config"].revision,
            splits={g:dict(**p["fold"],label_budget_sha256=sha256(OUT/g/"label_budget.csv")) for g,p in prepared.items()})
        if (OUT/"protocol.json").exists():
            assert read(OUT/"protocol.json")==protocol
        write_json(OUT/"protocol.json",protocol)
        pipeline=load_model(prepared[GROUPS[0]]["config"])
        pipeline.model,targets=attach_lora(pipeline.model)
        model=pipeline.model
        fingerprint=frozen_state_fingerprint(model)
        write_json(OUT/"foundation.json",dict(sha256=fingerprint,target_modules=targets))
        status.update(device=str(model.device),device_name=torch.xpu.get_device_name() if model.device.type=="xpu" else str(model.device))
        print(f"DEVICE {status['device_name']}: 12 candidates + 4 seed confirmations",flush=True)
        selected=search_and_confirm(prepared,model,status)
        assert fingerprint==frozen_state_fingerprint(model)
        status.update(status="selections_locked",all_selections_locked_at=now(),foundation_hash_verified=True)
        write_json(OUT/"status.json",status)
        if args.search_only:
            print("SEARCH_COMPLETE_NO_TEST_EVALUATED",flush=True)
            return
        for g in GROUPS:
            evaluate(g,prepared[g],pipeline,selected[g],status)
        assert fingerprint==frozen_state_fingerprint(model)
        report()
        status.update(status="complete",finished_at=now(),foundation_hash_verified=True,direct_window_r2_verified=True)
        write_json(OUT/"status.json",status)
        print("LORA_SEARCH_COMPLETE",flush=True)
    except Exception as error:
        status.update(status="failed",error=repr(error),failed_at=now())
        write_json(OUT/"status.json",status)
        raise


if __name__=="__main__":
    main()

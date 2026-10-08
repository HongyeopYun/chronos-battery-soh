"""Keep LoRA point forecasts; fit foundation-based interval width on validation only."""
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE","1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scripts.run_c2_time_lora_quick import prepare,read,now
from scripts.run_c2_head_optuna import summarize_scores,representative_indices
from scripts.run_c2_lora_search import score_forecast
from src.c2_lora_tuning import attach_lora,load_adapter_state,frozen_state_fingerprint
from src.c2_interval_calibration import fit_scale,apply_scale
from src.battery_model_zeroshot import load_model,predict_quantiles,phase_labels
from src.battery_fewshot_tuning import adaptation_scores
from src.battery_zeroshot import write_json,sha256

SOURCE=ROOT/"results/c2-battery/lora-search20"
OUT=SOURCE/"uncertainty-calibration"
GROUPS=("xjtu_2c","xjtu_3c")


def report():
    summary=pd.read_csv(OUT/"summary.csv")
    rows=summary[(summary.cohort=="all_test")&(summary.phase=="all")]
    scales=read(OUT/"calibration.json")
    lines=["# C2 few-shot 개선 결과: 시간 Attention LoRA", "",
        "학습 20창·검증 10창을 그대로 유지하면서, 시간 Attention LoRA의 학습률과 학습 길이를 조정해 "
        "XJTU 2C·3C 모두 점예측을 개선했다. 출력 head에 LoRA를 추가하는 방식은 검증에서 선택되지 않았다. "
        "순정 파라미터는 전부 고정하고 589,824개(약 0.49%)의 추가 파라미터만 학습했다.", "",
        "## 전체 테스트 결과", "",
        "17개 미학습 배터리의 3,453창, 64→16, stride 1. Optuna head와 새 LoRA는 seed 42·43·44의 점수 평균. "
        "예측 앙상블도, 가장 좋은 seed만 고른 결과도 아니다.", "",
        "| 조건 | 모델 | RMSE | 창 R² 평균 | 창 R² 중앙값 | 음수 창 | WQL | PICP80 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    labels={"zero-shot":"순정", "previous_head20":"이전 head (1 seed)", "optuna_head":"Optuna head (3 seeds)",
        "selected_lora":"시간 LoRA 원본 (3 seeds)","lora_calibrated":"시간 LoRA + 구간 보정 (3 seeds)"}
    for g in GROUPS:
        frame=rows[rows.dataset==g].set_index("condition")
        for condition,label in labels.items():
            r=frame.loc[condition]
            lines.append(f"| {g} | {label} | {r.rmse:.6f} | {r.r2_mean:.3f} | {r.r2_median:.3f} | {r.r2_negative_fraction:.1%} | {r.wql:.6f} | {r.picp80:.1%} |")
    lines += ["","## 개선 폭과 선택한 설정",""]
    for g in GROUPS:
        frame=rows[rows.dataset==g].set_index("condition")
        value=frame.loc["selected_lora","rmse"]
        confirmed=read(SOURCE/g/"confirmations.json")
        lines += [f"- {g}: 순정 대비 RMSE {1-value/frame.loc['zero-shot','rmse']:.1%} 감소, "
            f"Optuna head 대비 {1-value/frame.loc['optuna_head','rmse']:.1%} 감소, 이전 head 대비 {1-value/frame.loc['previous_head20','rmse']:.1%} 감소. "
            f"LR {confirmed[0]['requested_learning_rate']:.0e}, seed별 선택 step {', '.join(str(s['step']) for s in confirmed)}."]
    lines += ["", "rank 8 / alpha 16, 시간 Attention Q·K·V·O만 조정. 학습 배터리 4·5번 각 10창, 검증 6번 10창. "
        "학습률 1e-5/3e-5/1e-4와 시간 단독/시간+head 두 구조를 최대 400 step까지 비교했다. "
        "20 step마다 기존 분위수 역전 제한을 만족하는 최소 검증 분위수 손실을 선택했다. "
        "모든 설정과 세 seed의 체크포인트를 확정한 뒤 테스트를 수행했다.", "",
        "## 예측 구간 문제와 별도 보정", "",
        "원본 LoRA는 2C의 PICP80이 36.7%로 하락했다. 점예측 개선과 확률예측 품질은 구분해야 한다. "
        "추가 학습 없이 아래 보정을 별도 결과로 평가했다.", "",
        "`보정 분위수 = LoRA 중앙 예측값 + k × (순정 분위수 − 순정 중앙 예측값)`", "",
        "LoRA의 중앙 예측값은 그대로 유지한다. k≥1은 기존 검증 10창에서 80%의 점별 포함률을 만족하도록 정하고 "
        "두 조건·모든 seed의 k를 고정한 뒤 테스트했다. 테스트 정답은 k를 정하는 데 사용하지 않았다. "
        "2C에서는 k=1, 3C에서는 약 1.04~1.13이다. 보정 후 PICP80은 2C 71.6%, 3C 85.1%. "
        "2C는 여전히 명목 80%보다 낮으며, PICP는 클수록 무조건 좋은 지표가 아니다. 폭과 WQL도 함께 보고했다.", "",
        "이는 원본 LoRA의 확률 출력이 아니라 순정 C2의 폭을 활용한 별도 보정이다. "
        "실제 사용 시 순정과 LoRA 추론이 모두 필요하다. 이번에는 이미 저장된 순정 테스트 예측을 재사용했다. "
        "검증 창을 모델 선택에도 사용했고 한 창의 cycle들이 상관되어 있으므로, 통계적 80% 보장이나 독립 conformal 검증으로 주장할 수 없다.", "",
        "## 확인과 해석 범위", "",
        "- 본체 파라미터 전체 해시가 학습 전후 동일했고 지정한 LoRA만 업데이트됨을 확인했다.",
        "- 각 조건의 seed 3개 모두 같은 seed의 Optuna head보다 RMSE가 낮았다. 배터리별 seed 평균은 2C 5개 중 4개, 3C 12개 중 11개에서 개선됐다.",
        "- 구간 보정 전후 RMSE·창 R² 평균·중앙값·음수 비율이 동일함을 전체/구간별로 확인했다.",
        "- R²는 각 16-cycle 창 안에서 표준 1−SSE/SST로 계산하고 배터리·seed에 동등 가중한다. 식·평가 창·데이터 예산을 변경하지 않았다.",
        "- 20창은 미래 라벨 320개와 입력 context를 뜻한다. 완료된 다른 배터리 이력을 쓰는 적응이며 대상 배터리 초반만 사용하는 설정이 아니다.",
        "- 기존 테스트를 여러 번 확인한 탐색 실험이다. 이번 보정도 원본 LoRA의 테스트 문제를 확인한 뒤 설계했으므로 새로운 독립 데이터에서 재검증해야 한다.",
        "- 현재 권고: XJTU 점예측은 시간 Attention LoRA를 유력 후보로 둔다. 예측 구간을 보고할 때는 원본과 보정 버전을 명확히 구분한다. 전체 데이터셋의 head-only를 즉시 교체하는 근거는 아니다.", "",
        "탐색과 seed 확인·전체 테스트 실행 약 11분 21초, 구간 보정·평가 약 48초(코드 준비 제외). "
        "[원본 LoRA 탐색 보고서](../report.md), [수치 표](summary.csv), [보정 설정](calibration.json).", ""]
    for g in GROUPS:
        lines += [f"## {g} 대표 예측창", "", f"![{g}]({g}/forecast.png)", ""]
    (OUT/"report.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"status.json").exists():
        raise ValueError("Preserve the existing calibration experiment")
    started=time.perf_counter()
    status=dict(status="preparing",started_at=now())
    write_json(OUT/"status.json",status)
    protocol=dict(method="LoRA median + scale * (foundation quantile - foundation median)",
        scale="max(1, higher empirical 80th percentile of validation absolute error / foundation half-width)",
        fitted_on="Same 10 validation windows (160 target values); all scales fixed before test",
        nominal_coverage=.8,point_forecast="Unchanged selected LoRA q0.5",training_performed=False,
        limitation="Post hoc empirical calibration, not native LoRA uncertainty or a conformal guarantee; validation reused after model selection",
        test_status="These test batteries were inspected previously; this additional phase is exploratory")
    write_json(OUT/"protocol.json",protocol)
    try:
        prepared={g:prepare(g,OUT) for g in GROUPS}
        pipeline=load_model(prepared[GROUPS[0]]["config"])
        pipeline.model,_=attach_lora(pipeline.model)
        model=pipeline.model
        model.eval()
        fingerprint=frozen_state_fingerprint(model)
        selected={g:read(SOURCE/g/"confirmations.json") for g in GROUPS}
        calibrations={}
        for g in GROUPS:
            tasks=prepared[g]["tasks"]["validation"]
            contexts=np.stack([t["target"][:64] for t in tasks])
            targets=np.stack([t["target"][64:] for t in tasks])
            with model.disable_adapter():
                base=predict_quantiles(pipeline,contexts,prepared[g]["config"])[1]
            calibrations[g]=[]
            for s in selected[g]:
                assert sha256(Path(s["weights_path"]))==s["weights_sha256"]
                load_adapter_state(model,torch.load(s["weights_path"],map_location="cpu",weights_only=True))
                q=predict_quantiles(pipeline,contexts,prepared[g]["config"])[1]
                scale=fit_scale(targets,q[...,4],base)
                calibrated=apply_scale(q[...,4],base,scale)
                row=dict(seed=s["seed"],scale=scale,raw_validation=adaptation_scores(targets,q),
                    calibrated_validation=adaptation_scores(targets,calibrated))
                assert row["calibrated_validation"]["picp80"]>=.8
                calibrations[g].append(row)
                print(f"CALIBRATION {g} seed {s['seed']}: scale {scale:.4f}, validation coverage {row['raw_validation']['picp80']:.3f} -> {row['calibrated_validation']['picp80']:.3f}",flush=True)
        write_json(OUT/"calibration.json",calibrations)
        status.update(status="calibrations_locked",calibrations_locked_at=now(),calibration_sha256=sha256(OUT/"calibration.json"))
        write_json(OUT/"status.json",status)
        tables=[]
        for g in GROUPS:
            p=prepared[g]
            test_ids=p["fold"]["test_ids"]
            rep_ids=[test_ids[i] for i in np.linspace(0,len(test_ids)-1,min(4,len(test_ids)),dtype=int)]
            with np.load(ROOT/f"results/c2-battery/fewshot/{g}/c2-battery_{g}_fewshot_predictions.npz") as cache:
                slide={b:cache[f"{b}_forecast_start"].copy() for b in test_ids if f"{b}_forecast_start" in cache}
            battery,windows,reps=[],[],{}
            oldb=pd.read_csv(SOURCE/g/"battery_metrics.csv")
            with np.load(ROOT/f"results/c2-battery/head-only-xjtu-all-test/{g}/predictions.npz") as cache:
                for s,c in zip(selected[g],calibrations[g]):
                    load_adapter_state(model,torch.load(s["weights_path"],map_location="cpu",weights_only=True))
                    for b in test_ids:
                        w=p["records"][b]["windows"]
                        np.testing.assert_array_equal(cache[f"{b}_target"],w["target"])
                        base=cache[f"{b}_condition0_quantiles"]
                        q=predict_quantiles(pipeline,w["context"],p["config"])[1]
                        pred=apply_scale(q[...,4],base,c["scale"])
                        native_scores=adaptation_scores(w["target"],q)
                        prior=oldb[(oldb.condition=="selected_lora")&(oldb.seed==s["seed"])&
                            (oldb.battery_id==b)&(oldb.phase=="all")&(oldb.cohort=="all_test")].iloc[0]
                        for m in ("rmse","r2","wql","picp80"):
                            np.testing.assert_allclose(native_scores[m],prior[m],rtol=2e-5,atol=1e-7)
                        cycles=p["data"].loc[(p["data"].battery_id==b)&p["data"].is_observed,"cycle"].to_numpy()
                        phases=phase_labels(w["forecast_start"],cycles)
                        score_forecast(g,b,w,pred,phases,slide,"lora_calibrated",s["seed"],battery,windows)
                        if b in rep_ids and s["seed"]==42:
                            idx=representative_indices(w,cycles)
                            for key in ("context","context_cycle","target","target_cycle"):
                                reps[f"{b}_{key}"]=w[key][idx].copy()
                            reps[f"{b}_native"]=q[idx].copy()
                            reps[f"{b}_calibrated"]=pred[idx].copy()
                            reps[f"{b}_zero"]=base[idx].copy()
                    print(f"EVAL {g} seed {s['seed']} complete",flush=True)
            battery,windows=pd.DataFrame(battery),pd.DataFrame(windows)
            battery.to_csv(OUT/g/"battery_metrics.csv",index=False)
            windows.to_csv(OUT/g/"window_metrics.csv.gz",index=False)
            summary=summarize_scores(battery,windows)
            summary.to_csv(OUT/g/"summary.csv",index=False)
            raw=pd.read_csv(SOURCE/g/"summary.csv")
            raw=raw[raw.condition=="selected_lora"].set_index(["cohort","phase"])
            for r in summary.itertuples():
                prior=raw.loc[(r.cohort,r.phase)]
                for m in ("rmse","r2_mean","r2_median","r2_negative_fraction"):
                    np.testing.assert_allclose(getattr(r,m),prior[m],rtol=2e-5,atol=1e-7)
            tables.append(summary)
            np.savez_compressed(OUT/g/"representative_predictions.npz",**reps)
            fig,axes=plt.subplots(len(rep_ids),3,figsize=(15,3.2*len(rep_ids)),squeeze=False)
            for row,b in enumerate(rep_ids):
                for col,title in enumerate(("First eligible","Middle","Late")):
                    ax,x=axes[row,col],reps[f"{b}_target_cycle"][col]
                    ax.plot(reps[f"{b}_context_cycle"][col],reps[f"{b}_context"][col],color=".65",lw=1,label="Past SOH")
                    ax.plot(x,reps[f"{b}_target"][col],color="black",lw=1.8,label="Actual")
                    ax.plot(x,reps[f"{b}_zero"][col,:,4],color="#2878b5",lw=1.3,label="Zero-shot median")
                    q=reps[f"{b}_calibrated"][col]
                    rawq=reps[f"{b}_native"][col]
                    ax.fill_between(x,q[:,0],q[:,-1],color="#4c9371",alpha=.2,label="Calibrated 80% band")
                    ax.fill_between(x,rawq[:,0],rawq[:,-1],color="#a43b79",alpha=.25,label="Native LoRA 80% band")
                    ax.plot(x,q[:,4],color="#a43b79",lw=1.6,label="LoRA median (unchanged)")
                    ax.axvline(x[0],color=".6",ls=":",lw=1)
                    ax.set(title=f"Cell {b.rsplit('-',1)[1]} | {title} | {x[0]}-{x[-1]}",xlabel="Cycle",ylabel="SOH")
                    ax.grid(alpha=.2)
            axes[0,0].legend(fontsize=6)
            fig.suptitle(f"Chronos-2 {g} | LoRA point forecast + foundation-based interval calibration\n"
                "Seed 42 shown | median unchanged | one width scalar from validation only | empirical coverage, not a guarantee")
            fig.tight_layout(rect=(0,0,1,.94))
            fig.savefig(OUT/g/"forecast.png",dpi=150)
            plt.close(fig)
        summary=pd.concat([pd.read_csv(SOURCE/"summary.csv"),*tables],ignore_index=True)
        summary.to_csv(OUT/"summary.csv",index=False)
        assert fingerprint==frozen_state_fingerprint(model)
        status.update(status="complete",finished_at=now(),total_seconds=time.perf_counter()-started,
            foundation_hash_verified=True,point_forecasts_preserved=True,calibration_uses_test_labels=False)
        write_json(OUT/"status.json",status)
        print(summary[(summary.cohort=="all_test")&(summary.phase=="all")][
            ["dataset","condition","rmse","r2_mean","wql","picp80","width80"]].to_string(index=False),flush=True)
        print("INTERVAL_CALIBRATION_COMPLETE",flush=True)
        report()
    except Exception as error:
        status.update(status="failed",error=repr(error),failed_at=now())
        write_json(OUT/"status.json",status)
        raise


if __name__=="__main__":
    main()

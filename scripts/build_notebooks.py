"""Build experiment notebooks, preserving outputs of unchanged cells."""
import hashlib
from pathlib import Path
import shutil
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARCHIVE = ROOT / "notebooks" / "archive" / "20260923_before_improvements"

SETUP = '''from pathlib import Path
import sys

# Works with the kernel started from the project root or notebooks/.
PROJECT_ROOT = next(
    p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
    if (p / "data" / "raw").is_dir() and (p / "src" / "soh_experiment.py").is_file()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from src.soh_experiment import (
    load_config, load_soh, make_windows, temporal_split,
    find_zeroshot, find_lobo, run_zeroshot, run_lobo, verify_saved_run,
)
from src.soh_plots import forecast_grid, horizon_comparison, validation_curves, window_comparison
from src.soh_evaluation import export_run, export_diagnosis, evaluation_directory

CONFIG = load_config(PROJECT_ROOT)
FORCE_RERUN = False  # True: new run directory; False: reuse only verified compatible artifacts.
print("Project:", PROJECT_ROOT)
print(CONFIG)
'''

ZERO = [
    ("markdown", """# Chronos zero-shot: 재현 가능한 기준 실험

**02 갈무리:** zero-shot 기준선은 확보했습니다. 고정한 XPU 실행의 macro 중앙값 RMSE는 **0.016307**, 평균 예측 RMSE는 **0.018540**입니다. 마지막 절에서 저장된 진단 CSV로 기준 지표를 확인합니다. 04에서는 같은 실행 환경과 평가 조건으로 이 기준선을 다시 맞춥니다.

MAT의 방전 용량으로 `SOH = Qk / Q1`을 구성합니다. 실제 최근 64개 SOH를 사용해 다음 16개를 예측하는 **rolling-origin** 실험입니다. 매 시점의 실제 관측을 새로 제공하므로 첫 64개만 이용하는 수명 전체 예측과는 다릅니다.

모델 revision, 데이터 해시, 라이브러리 버전, seed와 예측 배열을 실행별로 저장합니다. 기본 설정은 100 samples, 추론 seed 42/43/44이며 점 예측은 보간 중앙값과 sample mean을 함께 보고합니다. `configs/experiment.json`에서 설정을 변경하세요.

R²는 배터리·seed별 전체 window/horizon을 펼쳐 `1 - SSE/SST`로 계산한 뒤 배터리를 동일 가중치로 평균합니다. 1은 완벽한 예측, 0은 평가 target 평균을 항상 예측한 수준이며 음수도 가능합니다. Horizon/개별 window 점수는 해당 범위의 평균을 기준으로 합니다. 상수 target은 NaN입니다. 원본 실행 기록은 보존하고 R²를 포함한 파생 평가를 `results/evaluation_r2/`에 저장합니다.

이전 노트북과 실행 출력은 `archive/20260923_before_improvements/`에 보관했습니다."""),
    ("code", SETUP),
    ("markdown", "## 원본 데이터 및 평가 윈도우\n정규화에는 첫 관측만 사용합니다. 배터리별 오차를 동일 가중치로 평균합니다."),
    ("code", '''soh_data = load_soh(PROJECT_ROOT, CONFIG)
display(pd.DataFrame([
    {"battery": b, "cycles": len(s), "first_soh": s[0], "last_soh": s[-1],
     "evaluation_windows": len(make_windows(s, CONFIG.context_length, CONFIG.prediction_length)["target"])}
    for b, s in soh_data.items()
]))'''),
    ("markdown", "## 추론 및 저장\n동일한 코드·데이터·모델·환경·추론 설정의 완료된 실행만 재사용합니다. 중단되거나 손상된 결과는 재사용하지 않습니다."),
    ("code", '''ZERO_RUN = None if FORCE_RERUN else find_zeroshot(PROJECT_ROOT, CONFIG)
ZERO_RUN = ZERO_RUN or run_zeroshot(PROJECT_ROOT, CONFIG)
print("Saved run:", ZERO_RUN)
display(verify_saved_run(ZERO_RUN, PROJECT_ROOT, CONFIG))
ZERO_EVAL = export_run(PROJECT_ROOT, ZERO_RUN)'''),
    ("markdown", "## 자동 계산된 성능\n표준편차는 추론 seed별 점수 변동입니다. 중첩 윈도우를 독립 표본으로 간주한 신뢰구간이 아닙니다. 단일 기준 모델의 표준편차는 정의되지 않아 NaN으로 표시합니다."),
    ("code", '''summary = pd.read_csv(ZERO_EVAL / "summary.csv")
metrics = pd.read_csv(ZERO_EVAL / "metrics.csv")
display(summary)
display(metrics.groupby(["model", "estimator", "battery"])[["rmse", "mae", "r2"]].mean())
display(metrics[metrics.model == "ZeroShot"].groupby("battery")[["coverage80", "width80"]].mean())'''),
    ("markdown", "## 예측 구간과 horizon별 오차\n아래 예시는 첫 번째 추론 seed의 첫·중간·마지막 윈도우입니다. 10~90% 예측 구간은 실제 coverage를 함께 확인해야 합니다."),
    ("code", '''forecast_grid(ZERO_RUN)
plt.show()
horizon_comparison(ZERO_RUN)
plt.show()
horizon_comparison(ZERO_EVAL, metric="r2")
plt.show()'''),
    ("markdown", """## 해석 시 주의

- 한 실제 cycle이 최대 16개 평가 윈도우에 포함됩니다. 배터리 수는 4개입니다.
- `sqrt(mean(error²))`와 window별 RMSE의 평균은 서로 다릅니다.
- Chronos T5의 토큰 양자화 때문에 샘플 예측이 계단 형태일 수 있습니다. 보간 중앙값은 두 bin 사이 값을 반환할 수도 있습니다.
- 03 노트북은 이 실행과 호환되는 zero-shot 예측을 찾아 같은 target, horizon, 집계 규칙으로 LoRA와 비교합니다."""),
]

from scripts.checkpoint_notebook_cells import checkpoint_lora_cells

LORA = checkpoint_lora_cells(SETUP)


# These tables refer to the frozen diagnosis, independently of any new run above.
CLOSEOUT_DATA = '''DIAGNOSIS = PROJECT_ROOT / "results" / "diagnostics" / "20260929"
DIAGNOSIS_EVAL = export_diagnosis(PROJECT_ROOT, DIAGNOSIS)
diagnostic_metrics = pd.read_csv(DIAGNOSIS_EVAL / "seed_confirmation" / "metrics.csv")
keys = ["model", "battery", "inference_seed"]
assert len(diagnostic_metrics) == 36 and not diagnostic_metrics.duplicated(keys).any()
assert set(diagnostic_metrics["inference_seed"]) == {42, 43, 44}
score_columns = ["median_rmse", "median_r2", "mean_rmse", "mean_r2", "crps", "above2_rate", "coverage80"]
# First average inference seeds, then give the four batteries equal weight.
diagnostic_battery = diagnostic_metrics.groupby(["model", "battery"])[score_columns].mean()
diagnostic_macro = diagnostic_battery.groupby(level="model").mean()
print("Fixed snapshot: 20260929; original XPU runs + EOS-loss ablation; training seed 42")
'''

ZERO.extend([
    ("markdown", """## 02 갈무리: 04에 넘길 zero-shot 기준선

아래 표는 `20260923T063348Z_zeroshot_b57856`의 고정 결과입니다. 위 실행 셀에서 다른 설정으로 새 실험을 만들더라도 이 표는 바뀌지 않습니다. 추론 seed 42/43/44의 점수를 평균하고 네 배터리에 같은 가중치를 줍니다. `mean_rmse`는 100개 sample을 평균한 예측의 오차이며, 배터리 평균이라는 뜻이 아닙니다. CRPS는 예측 분포 점수로 낮을수록 좋습니다.

결론은 **기준선 확보**입니다. zero-shot에도 드문 극단 sample이 있고, 80% 구간의 실제 포함률은 전체 평균 약 80.32%입니다. 이 데이터에서 평가한 미래 16-cycle 예측과 장기 수명/RUL 예측을 구분합니다."""),
    ("code", CLOSEOUT_DATA + '''display(diagnostic_macro.loc[["zero"]].round(6))
display(diagnostic_battery.xs("zero", level="model").round(6))'''),
    ("markdown", """02에서는 추가 튜닝 없이 이 결과를 기준선으로 보관합니다. 03은 LoRA의 학습 효과와 분포 악화를 다룹니다. 04 Full fine-tuning은 별도 실험이며 아직 실행하지 않았습니다.

[02·03 정리와 04 비교 조건](../docs/02_03_closeout.md) · [상세 진단 보고서](../results/diagnostics/20260929/diagnosis.md)"""),
])

LORA.extend([
    ("markdown", """## 03 갈무리: 기존 LoRA와 EOS loss 제외 비교

아래 결과는 앞의 기존 LoRA 실행과 구분되는 **2026-09-29 고정 진단**입니다. 원본 LoRA는 `20260923T063628Z_lora_2df6cb`입니다. EOS 제외 모델은 같은 데이터·초기화·학습률·선택 epoch·업데이트 수를 사용하되 마지막 출력 EOS label의 loss만 제외했습니다. 입력 EOS는 유지하고, 예측 16개 토큰의 loss 가중치도 유지했습니다. 출력 clipping은 하지 않았습니다.

네 fold, 학습 seed 42 하나, 추론 seed 42/43/44를 사용했습니다. Test 결과로 checkpoint를 선택하지는 않았지만, 이미 관찰한 배터리에 대한 사후 원인 진단이므로 새로운 독립 성능 검증은 아닙니다. 비율 열 `above2_rate`, `coverage80`은 0~1 단위입니다."""),
    ("code", CLOSEOUT_DATA + '''model_order = ["zero", "original_lora", "without_eos"]
display(diagnostic_macro.loc[model_order].round(6))
median_by_battery = diagnostic_battery["median_rmse"].unstack("model")[model_order]
median_by_battery["without_eos_vs_zero_error_change_pct"] = (
    median_by_battery["without_eos"] / median_by_battery["zero"] - 1
) * 100
display(median_by_battery.round(6))
display(diagnostic_battery["median_r2"].unstack("model")[model_order].round(6))
display(diagnostic_battery["mean_r2"].unstack("model")[model_order].round(6))
print("Error change: negative is better. Inference repetitions are not training repetitions.")'''),
    ("markdown", """### 분리 검증으로 확인한 원인

- **EOS 학습 항:** 이 항만 제외했을 때 네 배터리 모두 평균 예측 RMSE와 CRPS가 회복됐습니다. macro 중앙값 RMSE는 zero-shot보다 8.66% 감소했습니다.
- **재학습 업데이트 증가:** B0005에서 같은 full-data 학습의 81회와 135회를 비교하면 평균 예측 RMSE가 `0.041309 → 0.551199`로 악화됐습니다. epoch를 그대로 옮기면 데이터가 늘어난 만큼 업데이트도 늘어납니다.
- **학습률:** B0005에서 학습률만 `1e-4 → 3e-5`로 낮추어도 폭주가 줄었습니다. 나머지 배터리의 학습률 효과까지 분리한 것은 아닙니다.
- **추론 구현:** B0005 앞 32개 window에서 PEFT generate wrapper를 우회해도 결과가 같고, 같은 checkpoint의 FP32 추론에서도 폭주했습니다. FP32 재학습이나 cache 영향까지 배제한 것은 아닙니다.

다음 표는 **B0005 전체 window, 추론 seed 42 하나**의 비교입니다. 위 세 추론 seed 평균과 구분합니다. `selection_prefix`는 데이터와 업데이트 수가 함께 바뀐 보조 실험이므로 한 요인 비교 표에서 제외했습니다."""),
    ("code", '''training_controls = pd.read_csv(DIAGNOSIS_EVAL / "training" / "training_controls.csv")
display(training_controls.loc[
    training_controls["variant"] != "selection_prefix",
    ["variant", "checkpoint", "steps", "median_rmse", "median_r2", "mean_rmse", "mean_r2", "crps", "above2_rate"],
].round(6))
from IPython.display import Image
display(Image(filename=str(DIAGNOSIS / "diagnosis_comparison.png"), width=1000))'''),
    ("markdown", """### 03의 결론과 남은 문제

현재 LoRA의 문제를 단순히 데이터 부족이나 rank 부족으로 결론내릴 수 없습니다. **EOS 목적함수와 학습량의 영향**을 확인했고, EOS 제외 LoRA를 04의 비교 후보로 남깁니다.

- B0006 중앙값 RMSE는 zero-shot보다 약 0.77% 나쁩니다. B0007 중앙값은 기존 LoRA보다 나빠졌습니다.
- EOS 제외의 80% 구간 포함률은 macro 약 74.46%, B0007 약 65.29%로 낮아졌습니다. 정확도 개선과 불확실성 보정은 별도 문제입니다.
- 드문 극단 sample이 여전히 남고, 학습 난수 반복은 아직 하지 않았습니다.
- `cuda-fullft`의 KL 정규화·검증 정책 실험은 다른 프로토콜입니다. 이번 EOS 제외 결과와 개선율을 직접 비교하지 않습니다.

04에서는 같은 base model·장치·분할·EOS 정책으로 zero-shot / LoRA / Full FT를 비교하고, 학습률과 checkpoint는 훈련 배터리 내부 검증으로 고릅니다. 검증한 checkpoint를 그대로 평가하며 별도 refit은 기본 비교에 넣지 않습니다. 현재 진단 점수는 참고값이고, 그 프로토콜에 맞춘 LoRA 대조군을 다시 평가해야 합니다. **04 구현·학습은 이번 갈무리에 포함하지 않았습니다.**

[02·03 정리와 04 비교 조건](../docs/02_03_closeout.md) · [상세 진단 보고서와 원시 결과](../results/diagnostics/20260929/diagnosis.md)"""),
])


def write_notebook(name, cells):
    destination = ROOT / "notebooks" / name
    backup = ARCHIVE / name
    if destination.exists() and not backup.exists():
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination, backup)
        assert hashlib.sha256(destination.read_bytes()).digest() == hashlib.sha256(backup.read_bytes()).digest()
    previous = nbf.read(destination, as_version=4) if destination.exists() else None
    notebook = nbf.v4.new_notebook()
    if previous is not None:
        notebook.metadata = previous.metadata
    notebook.metadata["kernelspec"] = {"display_name": ".venv (Python 3.12)", "language": "python", "name": "python3"}
    old_cells = {(cell.cell_type, cell.source): cell for cell in previous.cells} if previous else {}
    notebook.cells = []
    for index, (kind, source) in enumerate(cells):
        cell = old_cells.get((kind, source))
        if cell is None:
            cell = nbf.v4.new_markdown_cell(source) if kind == "markdown" else nbf.v4.new_code_cell(source)
            cell.id = hashlib.sha256(f"{name}:{index}:{source}".encode()).hexdigest()[:12]
        notebook.cells.append(cell)
    nbf.validate(notebook)
    nbf.write(notebook, destination)
    print(destination)


if __name__ == "__main__":
    write_notebook("02_chronos_zeroshot.ipynb", ZERO)
    write_notebook("03_chronos_lora_lobo.ipynb", LORA)

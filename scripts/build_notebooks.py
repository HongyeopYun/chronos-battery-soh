"""Build the two small experiment notebooks; archive previous contents once."""
import hashlib
from pathlib import Path
import shutil

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
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

CONFIG = load_config(PROJECT_ROOT)
FORCE_RERUN = False  # True: new run directory; False: reuse only verified compatible artifacts.
print("Project:", PROJECT_ROOT)
print(CONFIG)
'''

ZERO = [
    ("markdown", """# Chronos zero-shot: 재현 가능한 기준 실험

MAT의 방전 용량으로 `SOH = Qk / Q1`을 구성합니다. 실제 최근 64개 SOH를 사용해 다음 16개를 예측하는 **rolling-origin** 실험입니다. 매 시점의 실제 관측을 새로 제공하므로 첫 64개만 이용하는 수명 전체 예측과는 다릅니다.

모델 revision, 데이터 해시, 라이브러리 버전, seed와 예측 배열을 실행별로 저장합니다. 기본 설정은 100 samples, 추론 seed 42/43/44이며 점 예측은 보간 중앙값과 sample mean을 함께 보고합니다. `configs/experiment.json`에서 설정을 변경하세요.

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
display(verify_saved_run(ZERO_RUN, PROJECT_ROOT, CONFIG))'''),
    ("markdown", "## 자동 계산된 성능\n표준편차는 추론 seed별 점수 변동입니다. 중첩 윈도우를 독립 표본으로 간주한 신뢰구간이 아닙니다. 단일 기준 모델의 표준편차는 정의되지 않아 NaN으로 표시합니다."),
    ("code", '''summary = pd.read_csv(ZERO_RUN / "summary.csv")
metrics = pd.read_csv(ZERO_RUN / "metrics.csv")
display(summary)
display(metrics.groupby(["model", "estimator", "battery"])[["rmse", "mae"]].mean())
display(metrics[metrics.model == "ZeroShot"].groupby("battery")[["coverage80", "width80"]].mean())'''),
    ("markdown", "## 예측 구간과 horizon별 오차\n아래 예시는 첫 번째 추론 seed의 첫·중간·마지막 윈도우입니다. 10~90% 예측 구간은 실제 coverage를 함께 확인해야 합니다."),
    ("code", '''forecast_grid(ZERO_RUN)
plt.show()
horizon_comparison(ZERO_RUN)
plt.show()'''),
    ("markdown", """## 해석 시 주의

- 한 실제 cycle이 최대 16개 평가 윈도우에 포함됩니다. 배터리 수는 4개입니다.
- `sqrt(mean(error²))`와 window별 RMSE의 평균은 서로 다릅니다.
- Chronos T5의 토큰 양자화 때문에 샘플 예측이 계단 형태일 수 있습니다. 보간 중앙값은 두 bin 사이 값을 반환할 수도 있습니다.
- 03 노트북은 이 실행과 호환되는 zero-shot 예측을 찾아 같은 target, horizon, 집계 규칙으로 LoRA와 비교합니다."""),
]

LORA = [
    ("markdown", """# Chronos LoRA: LOBO + 시간순 검증 + 전체 학습 데이터 재학습

**Outer LOBO:** 평가 배터리 1개는 학습과 epoch 선택에서 제외합니다.

**Epoch 선택:** 나머지 3개 배터리 각각의 뒤쪽 20%를 validation target으로 사용합니다. Train window의 마지막 target은 경계 이전, validation window의 첫 target은 경계 이후입니다. 경계에 걸친 target window는 제외하며 validation context에는 그 시점까지 관측된 과거가 들어갑니다.

**Refit:** 배터리별 validation RMSE의 평균으로 epoch를 고른 후, 새로운 모델을 생성하여 3개 배터리의 전체 윈도우로 선택된 epoch만큼 학습합니다. 검증에서 개선이 없으면 epoch 0을 선택할 수 있습니다. 평가 배터리 결과는 이 결정에 쓰지 않습니다.

기본값은 학습 seed 42 한 개와 추론 seed 42/43/44입니다. 추론 seed 반복은 학습 초기화 반복을 대신하지 않습니다. 추가 학습 반복은 `train_seeds`에 지정하세요.

이전 노트북과 실행 출력은 `archive/20260923_before_improvements/`에 보관했습니다."""),
    ("code", SETUP),
    ("markdown", "## Fold와 시간 경계 확인\nValidation target이 train context 또는 train target에 들어가지 않도록 target 시간으로 분할합니다. 이 검증은 같은 학습 배터리의 미래 구간이며, 새로운 배터리로의 일반화는 outer LOBO에서 평가합니다."),
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
display(pd.DataFrame([{"test": b, "train": ", ".join(x for x in CONFIG.battery_ids if x != b)} for b in CONFIG.battery_ids]))'''),
    ("markdown", "## 공통 함수로 4개 fold 실행\nSeed를 모델·LoRA 생성 전에 고정합니다. Validation 추론은 다음 epoch의 학습 난수에 영향을 주지 않습니다. 매 fold의 adapter, loss, 선택 과정, 원시 예측 samples를 저장합니다. 호환되는 baseline이 없으면 자동 생성합니다."),
    ("code", '''LORA_RUN = None if FORCE_RERUN else find_lobo(PROJECT_ROOT, CONFIG)
LORA_RUN = LORA_RUN or run_lobo(PROJECT_ROOT, CONFIG)
manifest = json.loads((LORA_RUN / "manifest.json").read_text(encoding="utf-8"))
ZERO_RUN = Path(manifest["zeroshot_run"])
print("LoRA run:", LORA_RUN)
print("Zero-shot reference:", ZERO_RUN)
display(verify_saved_run(ZERO_RUN, PROJECT_ROOT, CONFIG))
display(verify_saved_run(LORA_RUN, PROJECT_ROOT, CONFIG))'''),
    ("markdown", "## Validation 선택 결과\nEpoch 0은 adapter가 아직 학습되지 않은 상태입니다. 선택된 epoch와 refit loss를 모두 확인합니다."),
    ("code", '''display(pd.DataFrame([
    {"test": fold["test_battery"], "train_seed": fold["train_seed"],
     "selected_epoch": fold["selected_epoch"], "refit_losses": fold["refit_losses"]}
    for fold in manifest["folds"]
]))
validation_curves(LORA_RUN)
plt.show()'''),
    ("markdown", "## 동일 조건 비교\n숫자를 직접 입력하지 않고 저장된 예측에서 계산한 표입니다. RMSE와 MAE의 improvement가 양수이면 LoRA가 더 좋습니다. 추론 seed별 macro 점수와 배터리별 점수를 함께 봅니다."),
    ("code", '''display(pd.read_csv(LORA_RUN / "summary.csv"))
comparison = pd.read_csv(LORA_RUN / "paired_comparison.csv")
display(comparison.groupby(["estimator", "battery"])[["rmse_zero", "rmse_lora", "rmse_improvement", "mae_zero", "mae_lora"]].mean())
display(pd.read_csv(LORA_RUN / "paired_macro.csv"))
coverage = pd.read_csv(LORA_RUN / "metrics.csv")
display(coverage.groupby("battery")[["coverage80", "width80"]].mean())'''),
    ("markdown", "## Horizon 및 예측 시작점별 비교\nWindow별 RMSE 차이의 평균은 전체 RMSE 차이와 다릅니다. 거의 0인 차이는 허용오차에 따라 tied로 분류하여 improved/worsened와 중복되지 않게 합니다."),
    ("code", '''horizon_comparison(ZERO_RUN, LORA_RUN)
plt.show()
window_comparison(LORA_RUN)
plt.show()
paired_windows = pd.read_csv(LORA_RUN / "paired_windows.csv")
display(pd.crosstab(
    paired_windows.loc[paired_windows.estimator == "median", "battery"],
    paired_windows.loc[paired_windows.estimator == "median", "outcome"],
    normalize="index",
))'''),
    ("code", '''forecast_grid(LORA_RUN)
plt.show()'''),
    ("markdown", """## 저장된 모델 재사용과 해석

`manifest['folds']`에 adapter 경로가 기록되어 있습니다. `src.soh_experiment.reload_adapter(CONFIG, adapter_path)`로 pinned base model과 adapter를 다시 불러올 수 있습니다. 예측 NPZ에는 context, target, forecast_start, samples, median, mean, q10, q90이 저장됩니다.

- 데이터·모델 revision·환경·seed·예측 설정이 다른 실행을 섞지 않습니다.
- 3개 배터리에서 나온 중첩 윈도우는 수백 개 독립 배터리 사례가 아닙니다.
- Validation 선택 후 전체 데이터로 다시 학습하므로 validation 개선이 평가 배터리 개선을 보장하지 않습니다.
- 학습 seed 1개일 때 보고되는 추론 변동만으로 학습의 안정성이나 통계적 유의성을 주장하지 않습니다.
- 정격 용량이 아닌 첫 관측 용량 기준 SOH입니다. 장기 RUL 및 전압/전류 기반 SOH 추정은 별도 실험입니다."""),
]


def write_notebook(name, cells):
    destination = ROOT / "notebooks" / name
    backup = ARCHIVE / name
    if destination.exists() and not backup.exists():
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination, backup)
        assert hashlib.sha256(destination.read_bytes()).digest() == hashlib.sha256(backup.read_bytes()).digest()
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"display_name": ".venv (Python 3.12)", "language": "python", "name": "python3"}
    notebook.cells = [nbf.v4.new_markdown_cell(source) if kind == "markdown" else nbf.v4.new_code_cell(source)
                      for kind, source in cells]
    nbf.validate(notebook)
    nbf.write(notebook, destination)
    print(destination)


if __name__ == "__main__":
    write_notebook("02_chronos_zeroshot.ipynb", ZERO)
    write_notebook("03_chronos_lora_lobo.ipynb", LORA)

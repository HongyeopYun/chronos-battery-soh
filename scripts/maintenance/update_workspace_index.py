"""Rebuild read-only navigation for stable experiment and code locations."""
from __future__ import annotations

import ast
import html
import json
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = [
    "NASA 직접 RUL 개선", "직접 RUL · context 및 학습량", "현재 SOH",
    "간접 RUL · 미래 SOH 궤적", "통합 결과 및 갤러리", "데이터 전처리 · 감사", "과거 참조 및 진단",
]


def category(name: str) -> str:
    if "nasa_fewshot" in name or "nasa-fewshot" in name or name == "c2_rul_improvement":
        return CATEGORIES[0]
    if any(x in name for x in ("learning_curve", "learning-curve", "context_nominal", "context-nominal", "adaptation_xpu", "adaptation-xpu")):
        return CATEGORIES[1]
    if any(x in name for x in ("full_soh_rul", "full-soh4", "v4_suite", "v4-gpu-suite", "v4_experiments")):
        return CATEGORIES[4]
    if any(x in name for x in ("preprocess", "reviewed", "review_nasa", "soh_outliers", "data_audit", "eol_labels", "eol-label", "mit_qc")) and "one-step" not in name:
        return CATEGORIES[5]
    if "rul" in name:
        return CATEGORIES[3]
    if any(x in name for x in ("one_step", "one-step", "context2", "context4", "context8", "context16", "context32")):
        return CATEGORIES[2]
    return CATEGORIES[6]


def title(name: str) -> str:
    known = {
        "rul-nasa-fewshot-report-v1": "NASA 직접 RUL 개선 · 요약 및 전후 그래프",
        "rul-nasa-fewshot-improvement-v1": "NASA Head 개선 · 탐색 및 저장 예측",
        "rul-nasa-fewshot-robustness-v1": "NASA 개선 원인 · 셀 분할 반복 검증",
        "rul-nasa-fewshot-full-v1": "NASA 개선 Head 이후 전체 가중치 학습",
        "rul-context-nominal80-v1": "직접 RUL 입력 길이 · MIT 정격 80%",
        "rul-learning-curve-v1": "직접 RUL 학습 셀 수 · 정답 수",
        "full-soh4-rul-gallery-v1": "SOH 4→1 및 기존 RUL 전체 셀 갤러리",
        "indirect-rul-representatives-v1": "간접 RUL 대표 예측 궤적",
        "v4-gpu-suite-v2": "v4 SOH · 간접 RUL · 직접 RUL 통합 실험",
    }
    return known.get(name, name)


def result_catalog() -> list[dict]:
    results = ROOT / "results"
    dirs = sorted(p for p in (results / "c2-battery").iterdir() if p.is_dir() and not p.name.startswith("."))
    dirs += sorted(p for p in results.iterdir() if p.is_dir() and p.name != "c2-battery" and not p.name.startswith("."))
    records = []
    for folder in dirs:
        links = []
        for filename, label in [("index.html", "그래프 보기"), ("report.md", "보고서"), ("summary.csv", "성능표"), ("verification.json", "검증 기록"), ("audit.json", "감사 기록"), ("protocol.json", "실험 조건")]:
            if (folder / filename).is_file():
                links.append({"label": label, "path": (folder / filename).relative_to(results).as_posix()})
        if not links:
            links = [{"label": "폴더", "path": folder.relative_to(results).as_posix() + "/"}]
        records.append({"name": folder.name, "title": title(folder.name), "category": category(folder.name), "path": folder.relative_to(ROOT).as_posix(), "links": links})
    order = {name: i for i, name in enumerate(CATEGORIES)}
    records.sort(key=lambda x: (order[x["category"]], 0 if x["name"] == "rul-nasa-fewshot-report-v1" else 1, x["name"]))
    return records


def write_results(records: list[dict]) -> None:
    e = html.escape
    sections = []
    markdown = ["# 실험 결과 탐색", "", "[브라우저 탐색 페이지](index.html) · [프로젝트 구조](../docs/workspace_layout.md)", "", "아래는 저장된 결과의 위치 안내다. 파일 존재와 실험 검증 완료는 구분하며 각 실행의 조건·검증 기록을 확인한다.", ""]
    for i, group in enumerate(CATEGORIES):
        items = [r for r in records if r["category"] == group]
        if not items:
            continue
        cards = []
        markdown += [f"## {group}", ""]
        for r in items:
            links = " ".join(f'<a href="{quote(x["path"], safe="/")}">{e(x["label"])}</a>' for x in r["links"])
            cards.append(f'<article><h3>{e(r["title"])}</h3><p class="path">{e(r["path"])}</p><nav>{links}</nav></article>')
            mdlinks = " · ".join(f'[{x["label"]}]({quote(x["path"], safe="/")})' for x in r["links"])
            markdown.append(f'- **{r["title"]}** — {mdlinks}')
        sections.append(f'<section id="group-{i}"><h2>{e(group)}</h2><div class="grid">{"".join(cards)}</div></section>')
        markdown.append("")
    nav = " ".join(f'<a href="#group-{i}">{e(group)}</a>' for i, group in enumerate(CATEGORIES))
    document = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Chronos 배터리 연구 · 결과 탐색</title>
<style>body{{margin:0;background:#f4f6fa;color:#1b2840;font:16px/1.6 'Malgun Gothic',sans-serif}}main{{max-width:1150px;margin:auto;padding:38px 24px 70px}}h1{{font-size:32px;margin:8px 0}}h2{{margin:32px 0 14px}}h3{{font-size:17px;margin:0 0 6px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:14px}}article{{background:white;border:1px solid #dae1ec;border-radius:12px;padding:20px}}a{{color:#2453a4}}nav{{display:flex;flex-wrap:wrap;gap:10px 16px}}.path{{font-size:12px;color:#657189;overflow-wrap:anywhere}}.lead{{color:#526078}}header nav{{margin:20px 0}}.notice{{padding:14px 18px;background:#e9effa;border-radius:10px}}</style>
<main><header><p class="lead">Chronos-2 · SOH / RUL</p><h1>저장된 실험 결과</h1><p class="lead">원래 실험 경로를 유지하면서 주제별로 모은 탐색 페이지</p><nav>{nav}</nav><nav><a href="../README.md">프로젝트 안내</a><a href="../docs/workspace_layout.md">폴더 구조</a><a href="../presentations/README.md">발표 자료</a><a href="../scripts/README.md">실행 파일</a></nav></header><p class="notice">현재 SOH, 미래 SOH 궤적을 통한 간접 RUL, 새 회귀 Head를 사용하는 직접 RUL은 서로 다른 과제입니다. 데이터 버전·EOL 기준·학습 정답 수·평가 시점을 확인하세요.</p>{''.join(sections)}</main></html>'''
    (ROOT / "results/index.html").write_text(document, encoding="utf-8")
    (ROOT / "results/README.md").write_text("\n".join(markdown), encoding="utf-8")
    (ROOT / "docs/experiment_catalog.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_code_index(directory: str) -> None:
    folder = ROOT / directory
    lines = [f"# {'실행 진입점' if directory == 'scripts' else '공통 코드'} 안내", "", "기존 파일은 저장 실험의 경로·소스 해시를 보존하기 위해 이동하지 않았다. 아래에서 용도별로 찾는다.", "", "[프로젝트 구조](../docs/workspace_layout.md) · [전체 결과](../results/index.html)", ""]
    grouped: dict[str, list] = {c: [] for c in CATEGORIES}
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix not in (".py", ".ps1") or p.name == "__init__.py":
            continue
        description = ""
        if p.suffix == ".py":
            doc = ast.get_docstring(ast.parse(p.read_text(encoding="utf-8-sig")))
            if doc:
                description = doc.splitlines()[0].strip().replace("|", "/")
        grouped[category(p.stem)].append((p.name, description))
    for group, items in grouped.items():
        if not items:
            continue
        lines += [f"## {group}", "", "| 파일 | 설명 |", "|---|---|"]
        lines += [f"| [{name}]({name}) | {desc or '파일의 실행 인자와 연결된 보고서 확인'} |" for name, desc in items]
        lines.append("")
    if directory == "scripts":
        lines += ["## 관리 도구", "", "- [탐색 목록 갱신](maintenance/update_workspace_index.py): 학습 없이 현재 파일 목록에서 결과·코드 안내를 다시 생성한다.", "", "실행 예: `.venv\\Scripts\\python.exe scripts/maintenance/update_workspace_index.py`", ""]
    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    records = result_catalog()
    write_results(records)
    write_code_index("scripts")
    write_code_index("src")
    print(json.dumps({"result_directories": len(records), "navigation": str(ROOT / "results/index.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

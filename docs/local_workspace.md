# 로컬 폴더 이동 뒤 기존 실험 재사용

2026-10-01에 프로젝트를 `chronos-nasa-soh`에서 `chronos-battery-soh`로 옮긴 로컬 환경을 정리했다. 현재 02·03 노트북은 R²가 포함된 zero-shot과 EOS loss 제외·validation 유지·best-checkpoint LoRA를 사용한다. 노트북 생성 원본은 `scripts/build_notebooks.py`와 `scripts/checkpoint_notebook_cells.py`다.

## 경로 연결

기존 실험의 manifest와 trial 기록에는 프로젝트의 옛 절대경로가 남아 있다. 원본 기록과 해시를 유지하기 위해, 기존 `chronos-nasa-soh` 경로를 현재 `chronos-battery-soh` 폴더로 연결하는 Windows junction을 만들었다. 실제 프로젝트는 현재 폴더 하나다.

NASA 원본은 프로젝트 옆 `Dataset/Nasa/` 폴더에 있다. 기존 실행 코드가 찾는 `data/raw/B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat`은 해당 원본을 가리키는 hard link다. 연결 전에 네 파일의 SHA-256이 기존 zero-shot manifest의 데이터 해시와 모두 일치하는지 확인했다. 다른 데이터셋용 `data/raw/Calce`, `MIT`, `Nasa`, `XJTU` 폴더는 그대로 유지했다.

Junction과 hard link는 이 PC의 파일시스템 설정이며 Git에 포함되지 않는다. 다른 PC에서는 README의 `data/raw/` 위치에 원본 파일을 준비해야 한다. 기존 manifest의 절대경로도 함께 접근 가능해야 한다. Hard link는 원본과 같은 파일이므로 어느 쪽에서든 내용을 수정하면 함께 바뀐다.

## 노트북 정리

이전 버전으로 저장됐던 02·03은 `notebooks/.ipynb_checkpoints/before_cleanup_20261001_141826/`에 백업하고 최신 커밋 기준으로 복구했다. 이 백업도 Git에서 제외한다.

VS Code에 이전 노트북 탭이나 커널이 열려 있다면 탭을 닫고 현재 `chronos-battery-soh/notebooks/`의 파일을 다시 연 뒤 `.venv` 커널을 재시작한다. 생성 코드를 실행할 때는 다음 명령을 사용한다. 동일한 셀의 출력은 유지하고 내용이 바뀐 셀의 출력은 비운다.

```powershell
.\.venv\Scripts\python.exe scripts\build_notebooks.py
```

기존 zero-shot, 과거 LoRA, 현재 LoRA의 데이터·예측·adapter 해시를 다시 검사한다. 완료된 호환 실행이 검증되면 저장된 결과를 사용한다. 현재 결과 확인에는 `FORCE_RERUN = False`를 유지한다.

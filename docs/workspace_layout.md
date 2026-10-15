# 프로젝트 폴더 안내

정리일: 2026-10-15. 원본 실험 수치, 모델 가중치, 데이터 전처리 규칙을 바꾸지 않은 파일 정리다.

```text
chronos-battery-soh/
  README.md                    시작 페이지
  AGENTS.md                    새 산출물 저장 위치와 보존 규칙
  src/                         공통 학습·평가·전처리 코드
  scripts/                     기존 실행 진입점
    maintenance/               탐색 목록 생성 등 관리 도구
  configs/                     실행 설정
  tests/                       기존 검증 코드
  data/raw/, data/processed/   원본·가공 데이터
  models/                      모델 캐시·가중치
  results/
    index.html                 SOH / 직접 RUL / 간접 RUL / 정제 결과 탐색
    c2-battery/                기존 경로를 유지한 개별 실험
    preprocessing_*/          데이터 정제·복구 근거
  presentations/published/    발표 완성본
  artifacts/
    presentation-builds/      발표 생성 코드·중간 PPTX·렌더·초안
    remote-transfer/          데스크톱 전송 및 수신 묶음
    exports/                  결과 배포 ZIP
    maintenance/              정리 전후 해시·이동 기록·복구 자료
  archive/cleanup-20261013/   과거 실험 보관본
```

## 자주 찾는 항목

| 작업 | 시작 위치 |
|---|---|
| NASA 직접 RUL 개선 전후 보기 | [NASA 보고서](../results/c2-battery/rul-nasa-fewshot-report-v1/index.html) |
| 전체 SOH 4→1 및 RUL 그래프 | [전체 셀 갤러리](../results/c2-battery/full-soh4-rul-gallery-v1/index.html) |
| 직접 RUL 입력 길이·MIT 정격 80% 비교 | [context 갤러리](../results/c2-battery/rul-context-nominal80-v1/index.html) |
| 간접 RUL 대표 예측 경로 | [결과 탐색](../results/index.html)에서 간접 RUL 선택 |
| 학습·검증·보고서 실행 파일 찾기 | [scripts 안내](../scripts/README.md) |
| 모델과 공통 코드 찾기 | [src 안내](../src/README.md) |
| 발표 완성본·이전 초안 | [발표 안내](../presentations/README.md) |
| 이전 T5·LoRA 등 복구 | [archive 안내](../archive/README.md) |

## 기존 실험 경로를 유지한 이유

완료된 실험의 프로토콜과 검증 코드는 실행 소스의 SHA-256 및 기존 파일 경로를 사용한다. `src/`, 기존 `scripts/`, `results/` 내부의 개별 실험 폴더를 임의로 재배치하면 저장 가중치 재로딩, 과거 보고서 링크와 검증이 깨질 수 있다. 이 경로들은 그대로 두고 주제별 탐색 페이지를 추가했다.

결과 탐색은 폴더 이름과 현재 존재하는 보고서를 연결하는 목록이다. 단순히 파일이 존재한다는 이유로 실험 성공이나 재현을 보장하지 않는다. 각 실행의 `verification.json`, `audit.json`, `protocol.json`을 함께 확인한다.

이번에 실제로 이동한 것은 Lab 최상위와 ChatGPT 작업 폴더의 빌드·차트 캐시·발표 산출물, 데스크톱 전송 묶음, 결과 ZIP, 과거 실험 복구본이다. 이동 목록과 파일별 SHA-256은 [정리 기록](../artifacts/maintenance/20261015-workspace-cleanup/)에 있다.

공용 `Lab/Dataset`, 다른 연구인 `Lab/Models`, 논문·개인 문서는 기존 위치에 둔다. ChatGPT 프로젝트의 동기화된 `sources/`는 읽기 전용 참조이며 이동하거나 수정하지 않는다.

## 새 작업을 저장할 때

- 실험은 새 이름의 `results/c2-battery/<run-name>/`에 저장한다. 완료된 실행을 덮어쓰지 않는다.
- 발표 작업은 `artifacts/presentation-builds/<work-name>/`에서 만들고 완성본만 `presentations/published/`에 둔다.
- 전송·다운로드 ZIP은 `artifacts/remote-transfer/`, 최종 결과 ZIP은 `artifacts/exports/`에 둔다.
- 임시 점검 파일은 `artifacts/maintenance/<date>-<task>/`에 둔다.
- 탐색 목록 갱신: 프로젝트 루트에서 `.venv\Scripts\python.exe scripts/maintenance/update_workspace_index.py`.

`artifacts/`, 과거 보관본, 로컬 발표 완성본은 Git에서 제외한다. Git 제외는 삭제나 원격 백업 완료를 의미하지 않는다.

## 중복 저장 공간 정리

2026-10-15에 중복 전송 ZIP을 삭제하고, 과거 모델 8개와 원본 데이터 101개를 하드링크로 통합해 실제 디스크 공간 약 18.9 GiB를 확보했다. 기존 실행·복구 경로와 파일 내용은 유지했다. 탐색기의 폴더 크기는 공유 파일을 중복 계산할 수 있다. 공유 파일을 수정해야 할 때는 직접 덮어쓰지 말고 새 파일을 만들어 교체한다. [완료 내역과 검증](../artifacts/maintenance/20261015-storage-cleanup/README.md).

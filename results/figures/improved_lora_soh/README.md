# 개선 LoRA의 SOH 예측 곡선

2026-09-29 EOS loss 제외 실험의 저장 예측을 시각화했다. 학습 seed 42, 추론 seed 42, window당 100 samples를 사용했다. 재학습·새 추론은 하지 않았다.

- `improved_lora_4x3_windows`: 배터리 4행 × 초기·중간·후기 3열. 각 패널에 실제 관측 context 64개와 미래 16개 예측을 표시한다. 기존 `../zeroshot_0.png`와 같은 배치·색상·축·범례 스타일이며, 각 패널의 축 범위는 자동으로 설정한다.
- `improved_lora_4battery_overview`: 전체 실제 SOH 곡선에 같은 세 예측 구간을 겹친 2×2 요약 그림. 전체 수명을 한 번에 예측한 결과가 아니라, 각 시점의 실제 과거 64개를 입력한 별도 예측들이다.
- 두 그림 모두 PNG(240 dpi), PDF, SVG로 제공한다.

파란색 선은 실제 관측 context, 주황색 선은 실제 미래값, 초록색 선은 예측 sample의 보간 중앙값이다. 파란색 음영은 10~90% 분위수이고, 회색 점선은 마지막 context cycle이다. x축은 방전 cycle 번호, SOH는 `Qk/Q1`이다. 평활화나 clipping을 적용하지 않았다.

## 윈도우 선택

오차를 기준으로 고르지 않고 전체 평가 window의 첫 번째·중간·마지막(`0, n//2, n-1`)을 선택했다.

| 배터리 | 초기 예측 구간 | 중간 예측 구간 | 후기 예측 구간 |
|---|---|---|---|
| B0005 | 65–80 | 109–124 | 153–168 |
| B0006 | 65–80 | 109–124 | 153–168 |
| B0007 | 65–80 | 109–124 | 153–168 |
| B0018 | 65–80 | 91–106 | 117–132 |

`selected_windows.csv`에 context/target 구간과 각 window RMSE, `selected_forecasts.csv`에 실제값·중앙값·분위수가 있다. `figure_metadata.json`에는 원본 예측 파일과 SHA-256을 기록했다. 전체 SOH 곡선은 저장된 context/target에서 복원하고 모든 window와 일치하는지 확인했다.

세 윈도우는 전체 평가 결과의 일부다. 전체 성능·학습 조건·남은 한계는 [진단 보고서](../../diagnostics/20260929/diagnosis.md)를 참조한다.

프로젝트 루트에서 재생성:

```powershell
.\.venv\Scripts\python.exe scripts\plot_improved_lora.py
```

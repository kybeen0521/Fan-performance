import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# ------------------------------
# 파일 선택
# ------------------------------
Tk().withdraw()
file_path = filedialog.askopenfilename(title="CSV 파일을 선택하세요", filetypes=[("CSV files", "*.csv")])
if not file_path:
    raise SystemExit("❌ 파일이 선택되지 않았습니다.")

# ------------------------------
# CSV 파일 읽기 (Result 부분만)
# ------------------------------
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# "Result"가 시작되는 줄 찾기
start_idx = None
for i, line in enumerate(lines):
    if "Result" in line:
        start_idx = i + 1  # "Result" 다음 줄이 실제 헤더
        break

# Result 구간만 다시 DataFrame으로 읽기
df_result = pd.read_csv(file_path, skiprows=start_idx, encoding='utf-8')

# ------------------------------
# 데이터 확인
# ------------------------------
print("📄 Result 부분 컬럼:")
print(df_result.columns.tolist())
print(df_result.head())

# ------------------------------
# 필요한 열 추출
# ------------------------------
col_Q = [c for c in df_result.columns if "Air Volume" in c][0]
col_Ptotal = [c for c in df_result.columns if "Total Pressure" in c][0]
col_Pshaft = [c for c in df_result.columns if "Shaft Power" in c][0]

Q_m3_min = df_result[col_Q].values
P_total_mmAq = df_result[col_Ptotal].values
P_shaft_kW = df_result[col_Pshaft].values

# ------------------------------
# 단위 변환 및 효율 계산
# ------------------------------
Q_m3_s = Q_m3_min / 60
P_total_Pa = P_total_mmAq * 9.80665
P_shaft_W = P_shaft_kW * 1000

eta = (Q_m3_s * P_total_Pa / P_shaft_W) * 100

# ------------------------------
# 결과 DataFrame 및 출력
# ------------------------------
df_eff = pd.DataFrame({
    "Air Volume (m³/min)": Q_m3_min,
    "Total Pressure (mmAq)": P_total_mmAq,
    "Shaft Power (kW)": P_shaft_kW,
    "Computed Efficiency (%)": np.round(eta, 2)
})

print("\n✅ 계산 결과:")
print(df_eff)

# ------------------------------
# 효율 곡선 시각화
# ------------------------------
plt.figure(figsize=(7,4))
plt.plot(Q_m3_min, eta, 'o-', color='darkorange', lw=2, label='Computed Efficiency')
plt.xlabel("Air Volume (m³/min)")
plt.ylabel("Efficiency (%)")
plt.title("Fan Total Efficiency Curve (from CSV)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

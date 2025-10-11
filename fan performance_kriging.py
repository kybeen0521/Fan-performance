#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fan Efficiency from CSV + Cubic Spline + Gaussian Ordinary Kriging (Semivariogram)
Author: Final version (Random sampling Kriging + RMSE/R²)
Date: 2025-10-11
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# -----------------------------
# Constants
# -----------------------------
KRIGING_NOISE = 1e-12

# -----------------------------
# Utility / Kriging functions
# -----------------------------
def cubic_spline_interpolation(x, y, x_new):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n < 2:
        return np.full_like(x_new, y[0] if n == 1 else np.nan)
    h = np.diff(x)
    alpha = np.zeros(n)
    for i in range(1, n - 1):
        alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1])
    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    c = np.zeros(n)
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)
    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = ((y[j + 1] - y[j]) / h[j]) - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])
    y_new = np.zeros_like(x_new, dtype=float)
    for idx, xi in enumerate(x_new):
        j = np.searchsorted(x, xi) - 1
        j = np.clip(j, 0, n - 2)
        dx = xi - x[j]
        y_new[idx] = y[j] + b[j] * dx + c[j] * dx**2 + d[j] * dx**3
    return y_new

def gaussian_variogram(h, nugget, sill, vrange):
    return nugget + sill * (1.0 - np.exp(-(h / vrange) ** 2))

def ordinary_kriging_1d(x_data, y_data, x_pred, nugget, sill, vrange, variogram_func):
    x_data = np.asarray(x_data)
    n = len(x_data)
    cov_matrix = np.array([[sill - variogram_func(abs(xi - xj), 0.0, sill, vrange)
                            for xj in x_data] for xi in x_data])
    cov_matrix += KRIGING_NOISE * np.eye(n)
    cov_aug = np.zeros((n + 1, n + 1))
    cov_aug[:n, :n] = cov_matrix
    cov_aug[-1, :-1] = 1.0
    cov_aug[:-1, -1] = 1.0
    cov_aug[-1, -1] = 0.0
    y_pred = np.zeros(len(x_pred))
    for i, xp in enumerate(x_pred):
        k = np.array([sill - variogram_func(abs(xp - xi), 0.0, sill, vrange) for xi in x_data])
        sol = np.linalg.solve(cov_aug, np.append(k, 1.0))
        weights = sol[:-1]
        y_pred[i] = np.dot(weights, y_data)
    return y_pred

def semivariogram(x, y, n_lags=10):
    h_list, gamma_list = [], []
    n = len(x)
    for i in range(n - 1):
        for j in range(i + 1, n):
            h_list.append(abs(x[j] - x[i]))
            gamma_list.append(0.5 * (y[j] - y[i]) ** 2)
    h_arr = np.array(h_list)
    gamma_arr = np.array(gamma_list)
    if h_arr.size == 0:
        return np.array([]), np.array([])
    lag_edges = np.linspace(0.0, h_arr.max(), n_lags + 1)
    gamma_avg, lag_center = [], []
    for k in range(n_lags):
        mask = (h_arr >= lag_edges[k]) & (h_arr < lag_edges[k + 1])
        if np.any(mask):
            gamma_avg.append(np.mean(gamma_arr[mask]))
            lag_center.append((lag_edges[k] + lag_edges[k + 1]) / 2.0)
    return np.array(lag_center), np.array(gamma_avg)

# -----------------------------
# Main
# -----------------------------
def main():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(title="CSV 파일을 선택하세요", filetypes=[("CSV files", "*.csv")])
    if not file_path:
        raise SystemExit("❌ 파일이 선택되지 않았습니다.")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    start_idx = None
    for i, line in enumerate(lines):
        if "Result" in line:
            start_idx = i + 1
            break
    if start_idx is None:
        raise SystemExit("❌ 'Result' 섹션을 찾을 수 없습니다. CSV 형식을 확인하세요.")

    df_result = pd.read_csv(file_path, skiprows=start_idx, encoding='utf-8')
    logger.info("📄 Result 부분 컬럼:")
    logger.info(df_result.columns.tolist())

    def find_col(key_list):
        for key in key_list:
            for c in df_result.columns:
                if key.lower() in c.lower():
                    return c
        raise KeyError(f"컬럼 찾기 실패: {key_list}")

    col_Q = find_col(["Air Volume", "Air_Volume", "유량", "Flow"])
    col_Ptotal = find_col(["Total Pressure", "Total_Pressure", "전체 압력", "압력"])
    col_Pshaft = None
    for key in ["Shaft Power", "Input Power", "Power", "샤프트"]:
        try:
            col_Pshaft = find_col([key])
            break
        except KeyError:
            col_Pshaft = None
    if col_Pshaft is None:
        raise KeyError("Shaft/Input Power 관련 컬럼을 찾을 수 없습니다.")

    Q_m3_min = df_result[col_Q].astype(float).values
    P_total_mmAq = df_result[col_Ptotal].astype(float).values
    P_shaft_kW = df_result[col_Pshaft].astype(float).values

    Q_m3_s = Q_m3_min / 60.0
    P_total_Pa = P_total_mmAq * 9.80665
    P_shaft_W = P_shaft_kW * 1000.0

    with np.errstate(divide='ignore', invalid='ignore'):
        eta = (Q_m3_s * P_total_Pa) / P_shaft_W * 100.0
    eta = np.nan_to_num(eta, nan=0.0, posinf=0.0, neginf=0.0)
    eta = np.clip(eta, 0.0, 100.0)

    sort_idx = np.argsort(Q_m3_s)
    Q_sorted = Q_m3_s[sort_idx]
    eta_sorted = eta[sort_idx]

    Q_unique, idx_first = np.unique(Q_sorted, return_index=True)
    eta_unique = []
    for q in Q_unique:
        mask = np.isclose(Q_sorted, q)
        eta_unique.append(np.mean(eta_sorted[mask]))
    Q_unique = np.array(Q_unique)
    eta_unique = np.array(eta_unique)

    q_dense = np.linspace(Q_unique.min(), Q_unique.max(), 200)
    eta_spline_dense = cubic_spline_interpolation(Q_unique, eta_unique, q_dense)

    lag, gamma_exp = semivariogram(Q_unique, eta_unique, n_lags=12)
    if lag.size == 0:
        nugget_fit = 0.0
        sill_fit = np.var(eta_unique)
        range_fit = (Q_unique.max() - Q_unique.min()) / 3
    else:
        nugget_fit = float(gamma_exp[0]) if gamma_exp.size > 0 else 0.0
        sill_fit = float(gamma_exp.max()) if gamma_exp.size > 0 else np.var(eta_unique)
        idx95 = np.argmax(gamma_exp >= 0.95 * sill_fit) if gamma_exp.size > 0 else -1
        range_fit = float(lag[idx95]) if 0 <= idx95 < len(lag) else float(lag[-1])

    logger.info("\n=== Semivariogram Estimated Parameters ===")
    logger.info(f"nugget={nugget_fit:.6g}, sill={sill_fit:.6g}, range={range_fit:.6g}")

    # -----------------------------------------------------
    # Random sampling + Kriging + RMSE/R² evaluation
    # -----------------------------------------------------
    points_list = [10, 15, 20, 25]
    colors = ["purple", "blue", "green", "orange"]
    results = []

    plt.figure(figsize=(10, 6))
    plt.plot(Q_m3_min, eta, 'o', label="Original computed points", alpha=0.6)

    for n_pts, color in zip(points_list, colors):
        random_idx = np.sort(np.random.choice(len(q_dense), n_pts, replace=False))
        q_spline = q_dense[random_idx]
        eta_ctrl = eta_spline_dense[random_idx]

        eta_pred = ordinary_kriging_1d(q_spline, eta_ctrl, q_dense,
                                       nugget_fit, sill_fit, range_fit,
                                       gaussian_variogram)

        rmse = np.sqrt(np.mean((eta_spline_dense - eta_pred) ** 2))
        ss_res = np.sum((eta_spline_dense - eta_pred) ** 2)
        ss_tot = np.sum((eta_spline_dense - np.mean(eta_spline_dense)) ** 2)
        r2 = 1 - ss_res / ss_tot

        results.append({"Samples": n_pts, "RMSE": rmse, "R²": r2})
        plt.plot(q_dense * 60.0, eta_pred, color=color, lw=2, label=f"Kriging (Random {n_pts} pts)")

    plt.xlabel("Air Volume (m³/min)")
    plt.ylabel("Efficiency (%)")
    plt.title("Fan Efficiency: Random Sampled Kriging (Semivariogram)")
    plt.ylim(0, max(110, eta.max() * 1.2))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    logger.info("\n📊 Kriging 성능 비교 (랜덤 샘플 기반)")
    for r in results:
        logger.info(f"{r['Samples']:>5} pts → RMSE = {r['RMSE']:.4f}, R² = {r['R²']:.4f}")

if __name__ == "__main__":
    main()

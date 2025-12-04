# -*- coding: utf-8 -*-
"""
S-形線性資料 + 線性迴歸（含極值影響比較）GUI
需求：
1) 產生略帶 S 型的線性 2D 數據與散佈圖，並建立 Linear Regression 迴歸線
2) GUI 可設定：線性相依程度(0~1)、資料數量 N，按鈕產生數據 / 建立迴歸線
3) 加入少量極值，並同時顯示「含極值」與「移除極值」兩條迴歸線（不同顏色）

相依：numpy、matplotlib、tkinter
(不使用 sklearn，迴歸以 numpy.polyfit 完成)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


def generate_s_shape_data(n=200, linearity=0.8, outlier_count=0, outlier_mag=8.0, seed=42):
    """
    產生略帶 S 型（在主要線性趨勢上加上一點 tanh 非線性），再加入高斯雜訊。
    linearity ∈ [0,1]：代表線性相依程度；越大 => 雜訊越小（R^2 越高）。
    outlier_count：要製造的極值數量
    outlier_mag：極值幅度（相對整體 y 範圍的倍數）
    """
    rng = np.random.default_rng(seed)

    # x 均勻分佈在 [-1, 1]
    x = rng.uniform(-1.0, 1.0, size=n)

    # 主線性趨勢
    slope_true = 1.2
    intercept_true = 0.3
    y_lin = slope_true * x + intercept_true

    # 略微 S 型的非線性（tanh）
    s_amp = 0.35  # S 型強度（固定不大，確保仍以線性為主）
    y_nonlin = s_amp * np.tanh(3 * x)

    # 雜訊幅度依 linearity 調整：linearity 越大，noise 越小
    # 將 noise_std 調成 0.02~0.35 之間
    noise_std = 0.35 * (1.0 - float(linearity)) + 0.02
    noise = rng.normal(0.0, noise_std, size=n)

    y = y_lin + y_nonlin + noise

    # 人工加極值（在 y 軸上大幅偏移部份點）
    if outlier_count > 0:
        outlier_count = int(min(outlier_count, n))
        idx = rng.choice(n, size=outlier_count, replace=False)
        y_span = (np.max(y) - np.min(y) + 1e-6)
        kicks = (rng.choice([-1.0, 1.0], size=outlier_count) *
                 outlier_mag * y_span)
        y[idx] += kicks

    return x, y


def linear_regression_np(x, y):
    """用 numpy.polyfit 做 y = a*x + b 的最小平方線性迴歸，並回傳 (a, b, r2, yhat)。"""
    a, b = np.polyfit(x, y, deg=1)
    yhat = a * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return float(a), float(b), float(r2), yhat


def mad_zscore_mask(y, thresh=3.5):
    """
    用 robust 的 MAD z-score 抓極值。
    回傳：布林遮罩 is_outlier（True 代表離群）
    """
    med = np.median(y)
    mad = np.median(np.abs(y - med)) + 1e-12
    z = 0.6745 * (y - med) / mad
    return np.abs(z) > thresh


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("S-shaped Linear Data & Linear Regression (with Outliers Effect)")
        self.geometry("1000x660")

        # 狀態
        self.x = None
        self.y = None
        self.seed = 42

        self._build_ui()

    def _build_ui(self):
        # ---- 控制區 ----
        box = ttk.LabelFrame(self, text="參數設定", padding=8)
        box.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        self.var_n = tk.IntVar(value=200)
        self.var_lin = tk.DoubleVar(value=0.8)     # 線性相依程度
        self.var_oln = tk.IntVar(value=6)          # 極值個數
        self.var_olm = tk.DoubleVar(value=8.0)     # 極值幅度(倍)

        ttk.Label(box, text="資料數量 N").grid(row=0, column=0, sticky="w")
        ttk.Entry(box, width=8, textvariable=self.var_n).grid(row=0, column=1, padx=4)

        ttk.Label(box, text="線性相依程度 (0~1)").grid(row=0, column=2, sticky="w")
        ttk.Entry(box, width=6, textvariable=self.var_lin).grid(row=0, column=3, padx=4)

        ttk.Label(box, text="極值個數").grid(row=0, column=4, sticky="w")
        ttk.Entry(box, width=6, textvariable=self.var_oln).grid(row=0, column=5, padx=4)

        ttk.Label(box, text="極值幅度 (倍)").grid(row=0, column=6, sticky="w")
        ttk.Entry(box, width=6, textvariable=self.var_olm).grid(row=0, column=7, padx=4)

        ttk.Button(box, text="產生數據", command=self.on_generate).grid(row=0, column=8, padx=10)
        ttk.Button(box, text="建立迴歸線", command=self.on_fit).grid(row=0, column=9, padx=4)

        # ---- 圖表區 ----
        self.fig, self.ax = plt.subplots(figsize=(7.0, 4.8), dpi=100)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("Scatter (S-shaped), With & Without Outliers Regression")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # ---- 報表區 ----
        panel = ttk.Frame(self)
        panel.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.info = tk.StringVar(value="請先按『產生數據』。")
        ttk.Label(panel, textvariable=self.info, foreground="#333").pack(side=tk.LEFT)

        self.param = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.param, foreground="#0a5").pack(side=tk.RIGHT)

    # ---------------- 事件 ----------------
    def on_generate(self):
        try:
            n = max(10, int(self.var_n.get()))
            lin = float(self.var_lin.get())
            oln = max(0, int(self.var_oln.get()))
            olm = float(self.var_olm.get())
            if not (0 <= lin <= 1):
                raise ValueError
        except Exception:
            messagebox.showerror("輸入錯誤", "請確認：N 為整數、線性相依程度在 [0,1]、極值個數為整數、極值幅度為數字")
            return

        self.x, self.y = generate_s_shape_data(
            n=n, linearity=lin, outlier_count=oln, outlier_mag=olm, seed=self.seed
        )
        self._draw_scatter()
        self.info.set("資料已產生：散佈圖更新完成。")
        self.param.set(f"N={n} | 線性相依={lin:.2f} | 極值數={oln} | 幅度x{olm:g}")

    def on_fit(self):
        if self.x is None or self.y is None:
            messagebox.showwarning("尚未有資料", "請先按『產生數據』。")
            return

        # 1) 含極值的迴歸線
        a_all, b_all, r2_all, _ = linear_regression_np(self.x, self.y)

        # 2) 偵測離群值（MAD z-score），移除後再做一次迴歸
        mask_out = mad_zscore_mask(self.y, thresh=3.5)
        keep = ~mask_out
        kept = np.count_nonzero(keep)
        if kept < 3:
            messagebox.showwarning("有效點太少", "移除極值後點數不足以回歸。請降低極值個數或幅度。")
            return

        a_clean, b_clean, r2_clean, _ = linear_regression_np(self.x[keep], self.y[keep])

        # 畫線
        self._draw_scatter()
        xs = np.linspace(np.min(self.x), np.max(self.x), 200)

        # 含極值
        ys_all = a_all * xs + b_all
        self.ax.plot(xs, ys_all, color="crimson", lw=2.0, label=f"含極值回歸 y={a_all:.3f}x+{b_all:.3f} (R²={r2_all:.3f})")

        # 去除極值
        ys_clean = a_clean * xs + b_clean
        self.ax.plot(xs, ys_clean, color="seagreen", lw=2.0, label=f"去極值回歸 y={a_clean:.3f}x+{b_clean:.3f} (R²={r2_clean:.3f})")

        self.ax.legend(loc="best")
        self.canvas.draw()

        self.info.set(
            f"含極值：斜率={a_all:.4f}, 截距={b_all:.4f}, R²={r2_all:.4f} ｜ "
            f"去極值：斜率={a_clean:.4f}, 截距={b_clean:.4f}, R²={r2_clean:.4f}（移除 {np.count_nonzero(mask_out)} 點）"
        )

    # --------------- 繪圖輔助 ---------------
    def _draw_scatter(self):
        self.ax.clear()
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("Scatter (S-shaped), With & Without Outliers Regression")
        self.ax.grid(True, alpha=0.25)

        if self.x is not None and self.y is not None:
            self.ax.scatter(self.x, self.y, s=18, alpha=0.75, edgecolor="k", linewidths=0.2, label="data")
            self.ax.legend(loc="best")

        self.canvas.draw()


if __name__ == "__main__":
    App().mainloop()

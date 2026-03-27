import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df_train = pd.read_excel("Data4Regression.xlsx", sheet_name=0)
df_test  = pd.read_excel("Data4Regression.xlsx", sheet_name=1)

x_train = df_train.iloc[:, 0].values
y_train = df_train.iloc[:, 1].values
x_test  = df_test.iloc[:, 0].values
y_test  = df_test.iloc[:, 1].values

#三频三角函数
def func(x, a, b, c1, c2, c3, d1, d2, d3, w):
    return (a * x + b 
            + c1 * np.sin(w * x) + d1 * np.cos(w * x)
            + c2 * np.sin(2 * w * x) + d2 * np.cos(2 * w * x)
            + c3 * np.sin(3 * w * x) + d3 * np.cos(3 * w * x))

p0 = [0.05, -0.6, 1.0, 0.5, 0.3, 1.0, 0.5, 0.3, 0.75]
popt, _ = curve_fit(
    func, x_train, y_train,
    p0=p0,
    maxfev=200000,
    ftol=1e-12,
    xtol=1e-12,
    gtol=1e-12
)
a, b, c1, c2, c3, d1, d2, d3, w = popt
y_pred_train = func(x_train, a, b, c1, c2, c3, d1, d2, d3, w)
y_pred_test  = func(x_test,  a, b, c1, c2, c3, d1, d2, d3, w)

x_plot = np.linspace(x_train.min(), x_train.max(), 600)
y_plot = func(x_plot, a, b, c1, c2, c3, d1, d2, d3, w)

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, c='#3498db', s=50, alpha=0.7, label='训练集')
plt.scatter(x_test,  y_test,  c='#e74c3c', s=50, alpha=0.7, label='测试集')
plt.plot(x_plot, y_plot, c='#e74c3c', linewidth=3)
plt.title("三频三角函数高精度拟合", fontsize=15, pad=15)

plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.savefig("三频三角函数拟合.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"函数：y = ax + b + c1·sin(wx)+d1·cos(wx) + c2·sin(2wx)+d2·cos(2wx) + c3·sin(3wx)+d3·cos(3wx)\n")

print(f"a  = {a:.6f}")
print(f"b  = {b:.6f}")
print(f"c1 = {c1:.6f}   d1 = {d1:.6f}")
print(f"c2 = {c2:.6f}   d2 = {d2:.6f}")
print(f"c3 = {c3:.6f}   d3 = {d3:.6f}")
print(f"w  = {w:.6f}\n")

print("完整拟合函数：")
print(f"y = {a:.6f}x + {b:.6f}")
print(f" + {c1:.6f}sin({w:.6f}x) + {d1:.6f}cos({w:.6f}x)")
print(f" + {c2:.6f}sin({2*w:.6f}x) + {d2:.6f}cos({2*w:.6f}x)")
print(f" + {c3:.6f}sin({3*w:.6f}x) + {d3:.6f}cos({3*w:.6f}x)")

print("\n" + "=" * 70)
print(f"训练集 MSE = {mean_squared_error(y_train, y_pred_train):.6f}")
print(f"测试集 MSE = {mean_squared_error(y_test, y_pred_test):.6f}")
print("=" * 70)
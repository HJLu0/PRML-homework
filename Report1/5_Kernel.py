import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df_train = pd.read_excel("Data4Regression.xlsx", sheet_name=0)
df_test  = pd.read_excel("Data4Regression.xlsx", sheet_name=1)

x_train = df_train.iloc[:, [0]].values    # 必须是二维
y_train = df_train.iloc[:, 1].values
x_test  = df_test.iloc[:, [0]].values
y_test  = df_test.iloc[:, 1].values

# 高斯核
kernel_model = KernelRidge(kernel='rbf', gamma=0.3, alpha=0.01)
kernel_model.fit(x_train, y_train)

y_pred_train = kernel_model.predict(x_train)
y_pred_test  = kernel_model.predict(x_test)

x_plot = np.linspace(x_train.min(), x_train.max(), 300).reshape(-1, 1)
y_plot = kernel_model.predict(x_plot)

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, c='#3498db', s=50, alpha=0.7, label='训练集')
plt.scatter(x_test,  y_test,  c='#e74c3c', s=50, alpha=0.7, label='测试集')
plt.plot(x_plot, y_plot, c='#e67e22', linewidth=3, label='核拟合曲线')

plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.title("核方法（RBF高斯核）拟合结果", fontsize=14)
plt.savefig("核拟合.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"训练集 MSE = {mean_squared_error(y_train, y_pred_train):.6f}")
print(f"测试集 MSE = {mean_squared_error(y_test, y_pred_test):.6f}")
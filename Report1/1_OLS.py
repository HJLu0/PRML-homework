import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df_train = pd.read_excel("Data4Regression.xlsx", sheet_name=0)
df_test  = pd.read_excel("Data4Regression.xlsx", sheet_name=1)

x_train = df_train.iloc[:,0].values
y_train = df_train.iloc[:,1].values
x_test  = df_test.iloc[:,0].values
y_test  = df_test.iloc[:,1].values

# 最小二乘法
X = np.c_[np.ones(len(x_train)), x_train]
theta = np.linalg.inv(X.T @ X) @ X.T @ y_train

y_hat_train = X @ theta
X_test_mat = np.c_[np.ones(len(x_test)), x_test]
y_hat_test = X_test_mat @ theta

plt.figure(figsize=(10,6))
plt.scatter(x_train, y_train, color='#3498db', label='训练集', s=45, alpha=0.7)
plt.scatter(x_test, y_test, color='#e74c3c', label='测试集', s=45, alpha=0.7)

x_line = np.linspace(x_train.min(), x_train.max(), 200)
y_line = theta[0] + theta[1] * x_line
plt.plot(x_line, y_line, color='#2ecc71', linewidth=3, label='拟合曲线')

plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.title("最小二乘法", fontsize=14)
plt.savefig("最小二乘法.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"拟合参数: w0={theta[0]:.6f}, w1={theta[1]:.6f}")
print(f"训练MSE: {mean_squared_error(y_train, y_hat_train):.6f}")
print(f"测试MSE: {mean_squared_error(y_test, y_hat_test):.6f}")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def make_moons_3d(n_samples=500, noise=0.1):
    # Generate the original 2D make_moons data
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  

    # Concatenating the positive and negative moons with an offset and noise
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y_labels = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # Adding Gaussian noise
    X += np.random.normal(scale=noise, size=X.shape)

    return X, y_labels

# 生成训练集 (1000个数据)
X_train, y_train = make_moons_3d(n_samples=500, noise=0.2)
train_data = pd.DataFrame(X_train, columns=['X', 'Y', 'Z'])
train_data['Category'] = ['C0' if label == 0 else 'C1' for label in y_train]
train_data.to_csv('train_data.csv', index=False)

# 生成测试集 (500个数据)
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)
test_data = pd.DataFrame(X_test, columns=['X', 'Y', 'Z'])
test_data['Category'] = ['C0' if label == 0 else 'C1' for label in y_test]
test_data.to_csv('test_data.csv', index=False)

print("成功生成并保存文件：train_data.csv 和 test_data.csv")

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
colors = train_data['Category'].map({'C0': 0, 'C1': 1})
scatter = ax.scatter(train_data['X'], train_data['Y'], train_data['Z'], c=colors, cmap='viridis')
ax.legend(*scatter.legend_elements(), title="Classes")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Make Moons - Training Set')
plt.show()
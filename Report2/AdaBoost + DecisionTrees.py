import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from skimage import measure


try:
    train_df = pd.read_csv("train_data.csv")
    test_df = pd.read_csv("test_data.csv")

    X_train = train_df[['X', 'Y', 'Z']]
    y_train = train_df['Category'].map({'C0': 0, 'C1': 1})
    
    X_test = test_df[['X', 'Y', 'Z']]
    y_test = test_df['Category'].map({'C0': 0, 'C1': 1})
except FileNotFoundError:
    print("错误：未找到 CSV 文件。")
    exit()

# 训练 AdaBoost 模型
base_clf = DecisionTreeClassifier(max_depth=5)
ada_model = AdaBoostClassifier(
    base_estimator=base_clf, 
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)
ada_model.fit(X_train, y_train)


y_train_pred = ada_model.predict(X_train)
y_test_pred = ada_model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"训练集准确率: {train_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")

report_dict = classification_report(y_test, y_test_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
final_report = report_df.drop(['macro avg', 'weighted avg'])

print(final_report)

def plot_3d_ada_surface(model, X_data, y_data, resolution=50):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 确定坐标轴范围
    x_min, x_max = X_data['X'].min() - 0.5, X_data['X'].max() + 0.5
    y_min, y_max = X_data['Y'].min() - 0.5, X_data['Y'].max() + 0.5
    z_min, z_max = X_data['Z'].min() - 0.5, X_data['Z'].max() + 0.5

    # 创建 3D 网格点
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    z_grid = np.linspace(z_min, z_max, resolution)
    xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    
    grid_pts = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    grid_df = pd.DataFrame(grid_pts, columns=['X', 'Y', 'Z'])
    probs = model.predict_proba(grid_df)[:, 1]
    vol = probs.reshape(xx.shape)

    try:
        verts, faces, normals, values = measure.marching_cubes(vol, level=0.5)

        # 映射回原始坐标系
        verts[:, 0] = verts[:, 0] * (x_max - x_min) / (resolution - 1) + x_min
        verts[:, 1] = verts[:, 1] * (y_max - y_min) / (resolution - 1) + y_min
        verts[:, 2] = verts[:, 2] * (z_max - z_min) / (resolution - 1) + z_min

        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                        color='purple', alpha=0.4, lw=0.5, antialiased=True)
    except Exception as e:
        print(f"曲面生成失败: {e}")

    ax.scatter(X_data['X'], X_data['Y'], X_data['Z'], c=y_data, cmap='coolwarm', s=10, alpha=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D AdaBoost Decision Surface (Cyan Style)")
    
    ax.view_init(elev=20, azim=45)
    plt.show()
plot_3d_ada_surface(ada_model, X_train, y_train)
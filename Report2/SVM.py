import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
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

# 可视化
def plot_svm_3d(model, kernel_name, X_data, y_data, resolution=40):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 确定坐标范围
    x_min, x_max = X_data['X'].min() - 0.5, X_data['X'].max() + 0.5
    y_min, y_max = X_data['Y'].min() - 0.5, X_data['Y'].max() + 0.5
    z_min, z_max = X_data['Z'].min() - 0.5, X_data['Z'].max() + 0.5

    # 生成网格
    xi = np.linspace(x_min, x_max, resolution)
    yi = np.linspace(y_min, y_max, resolution)
    zi = np.linspace(z_min, z_max, resolution)
    xx, yy, zz = np.meshgrid(xi, yi, zi, indexing='ij')
    
    # 预测概率
    grid_pts = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    grid_df = pd.DataFrame(grid_pts, columns=['X', 'Y', 'Z'])
    
    # 获取决策值 (距离超平面的距离)
    decision_values = model.decision_function(grid_df)
    vol = decision_values.reshape(xx.shape)

    # 提取决策边界 (决策值为 0 的地方)
    try:
        verts, faces, _, _ = measure.marching_cubes(vol, level=0.0)
        verts[:, 0] = verts[:, 0] * (x_max - x_min) / (resolution - 1) + x_min
        verts[:, 1] = verts[:, 1] * (y_max - y_min) / (resolution - 1) + y_min
        verts[:, 2] = verts[:, 2] * (z_max - z_min) / (resolution - 1) + z_min

        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                        color='purple', alpha=0.35, lw=0.2, antialiased=True)
    except:
        pass

    ax.scatter(X_data['X'], X_data['Y'], X_data['Z'], c=y_data, cmap='coolwarm', s=10, alpha=0.3)
    ax.set_title(f"SVM with {kernel_name} Kernel")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=45)
    plt.show()

# 三种核函数
kernels = [
    ('Linear', SVC(kernel='linear')),
    ('RBF', SVC(kernel='rbf', gamma='auto')),
    ('Polynomial', SVC(kernel='poly', degree=3))
]

for name, clf in kernels:
    print(f"\n" + "="*30)
    print(f"核函数: {name}")
    
    # 训练模型
    clf.fit(X_train, y_train)
    
    # 预测并评估
    y_tr_pred = clf.predict(X_train)
    y_te_pred = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_tr_pred)
    test_acc = accuracy_score(y_test, y_te_pred)
    
    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    
    report = classification_report(y_test, y_te_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().drop(['macro avg', 'weighted avg'])
    print(report_df)
    plot_svm_3d(clf, name, X_train, y_train)
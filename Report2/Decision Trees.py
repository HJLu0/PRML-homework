import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

try:
    train_df = pd.read_csv("train_data.csv")
    test_df = pd.read_csv("test_data.csv")
    
    X_train = train_df[['X', 'Y', 'Z']]
    y_train = train_df['Category'].map({'C0': 0, 'C1': 1})
    
    X_test = test_df[['X', 'Y', 'Z']]
    y_test = test_df['Category'].map({'C0': 0, 'C1': 1})
except FileNotFoundError:
    print("错误：未找到 CSV 数据文件。")
    exit()

# 训练决策树模型
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=42)
dt_model.fit(X_train, y_train)

# 训练集准确率
y_train_pred = dt_model.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)

# 测试集准确率
y_test_pred = dt_model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Decision Trees 训练集准确率: {train_acc:.4f}")
print(f"Decision Trees 测试集准确率: {test_acc:.4f}")

report = classification_report(y_test, y_test_pred, output_dict=True)
for label in ['0', '1']:
    metrics = report[label]
    print(f"类别 {label}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1-score={metrics['f1-score']:.2f}")

# 3D 决策盒子可视化函数
def plot_3d_decision_boxes(model, X_data, y_data):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 获取边界
    x_min, x_max = X_data['X'].min(), X_data['X'].max()
    y_min, y_max = X_data['Y'].min(), X_data['Y'].max()
    z_min, z_max = X_data['Z'].min(), X_data['Z'].max()

    tree = model.tree_
    node_bounds = np.zeros((tree.node_count, 6))
    node_bounds[0] = [x_min, x_max, y_min, y_max, z_min, z_max]

    def get_bounds(node_id):
        if tree.children_left[node_id] != -1: 
            feat = tree.feature[node_id]
            thresh = tree.threshold[node_id]
            
            node_bounds[tree.children_left[node_id]] = node_bounds[node_id].copy()
            node_bounds[tree.children_right[node_id]] = node_bounds[node_id].copy()
            
            node_bounds[tree.children_left[node_id], feat * 2 + 1] = thresh 
            node_bounds[tree.children_right[node_id], feat * 2] = thresh    
            
            get_bounds(tree.children_left[node_id])
            get_bounds(tree.children_right[node_id])

    get_bounds(0)

    leaf_nodes = np.where(tree.children_left == -1)[0]
    
    for node_id in leaf_nodes:
        box_class = np.argmax(tree.value[node_id])
        color = 'purple' if box_class == 0 else 'gold'
        b = node_bounds[node_id]
        
        vertices = np.array([
            [b[0], b[2], b[4]], [b[1], b[2], b[4]], [b[1], b[3], b[4]], [b[0], b[3], b[4]],
            [b[0], b[2], b[5]], [b[1], b[2], b[5]], [b[1], b[3], b[5]], [b[0], b[3], b[5]]
        ])
        
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]], 
            [vertices[4], vertices[5], vertices[6], vertices[7]], 
            [vertices[0], vertices[1], vertices[5], vertices[4]], 
            [vertices[2], vertices[3], vertices[7], vertices[6]], 
            [vertices[0], vertices[3], vertices[7], vertices[4]], 
            [vertices[1], vertices[2], vertices[6], vertices[5]]  
        ]
        
        poly3d = Poly3DCollection(faces, facecolors=color, linewidths=0.2, edgecolors='k', alpha=0.08)
        ax.add_collection3d(poly3d)

    ax.scatter(X_data['X'], X_data['Y'], X_data['Z'], c=y_data, cmap='viridis', s=10, alpha=0.4)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("3D Decision Boundary: Precise Partitioning Surfaces")
    ax.view_init(elev=20, azim=45)
    plt.show()
plot_3d_decision_boxes(dt_model, X_train, y_train)
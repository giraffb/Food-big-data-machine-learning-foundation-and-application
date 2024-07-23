from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 定义训练和评估决策树模型的函数
def train_evaluate_decision_tree(criterion, X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(criterion=criterion)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return clf, accuracy

# 加载葡萄酒数据集
data = load_wine()
X = data.data
y = data.target

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义要使用的算法
algorithms = ["gini", "entropy"]  # gini 对应 CART， entropy 对应 ID3/C4.5

# 存储结果
results = {}

# 训练和评估每种算法的决策树模型
for criterion in algorithms:
    model, accuracy = train_evaluate_decision_tree(criterion, X_train, X_test, y_train, y_test)
    results[criterion] = (model, accuracy)

# 打印性能比较
print("Performance comparison:")
for criterion, (model, accuracy) in results.items():
    print(f"{criterion} algorithm accuracy: {accuracy:.4f}")

# 可视化决策树
for criterion, (model, accuracy) in results.items():
    plt.figure(figsize=(20, 12))
    plot_tree(model, feature_names=data.feature_names, class_names=data.target_names, filled=True, rounded=True)
    plt.title(f"{criterion} algorithm\nAccuracy: {accuracy:.4f}")
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 加载数据集
file_path = 'archive (7)/zoo.csv'
zoo_data = pd.read_csv(file_path, header=0)

# 提取特征和目标变量
feature_names = zoo_data.columns.tolist()
X = zoo_data[feature_names[:-1]]
y = zoo_data[feature_names[-1]]

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 将目标变量转换为PLS回归的二进制矩阵
lb = LabelBinarizer()
y_train_binarized = lb.fit_transform(y_train)
y_test_binarized = lb.transform(y_test)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用n_components=16拟合PLS回归模型
pls = PLSRegression(n_components=16)
pls.fit(X_train_scaled, y_train_binarized)

# 进行预测
y_pred_binarized = pls.predict(X_test_scaled)
y_pred = lb.inverse_transform(y_pred_binarized)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 假设y_test和y_pred是你的测试集标签和预测标签
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
# 生成分类报告
class_report = classification_report(y_test, y_pred)

# 打印结果
print(f'偏最小二乘法判别分析- 准确率: {accuracy:.2%}')
print('\n混淆矩阵:')
print(conf_matrix)
print('\n分类报告:')
print(class_report)

# 绘制混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=lb.classes_, yticklabels=lb.classes_)
plt.xlabel('预测值')
plt.ylabel('实际值')
plt.title('混淆矩阵')
plt.show()

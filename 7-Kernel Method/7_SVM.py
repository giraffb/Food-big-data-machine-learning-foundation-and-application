import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 加载数据，提取特征和标签
df = pd.read_excel('Date_Fruit_Datasets.xlsx')  # 确保文件路径正确
X = df.iloc[:, :-1]  # 特征是除了最后一列的所有列
y = df.iloc[:, -1]   # 标签是最后一列
print(df)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建SVM分类器实例，这里使用RBF核
svm_classifier = SVC(kernel='rbf', C=1, gamma='scale')

# 训练模型
svm_classifier.fit(X_train, y_train)

# 预测测试集
y_pred = svm_classifier.predict(X_test)

# 评估模型
print("准确率:", accuracy_score(y_test, y_pred))
print("分类报告:\n", classification_report(y_test, y_pred))

# 使用混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 使用seaborn的heatmap函数绘制混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.show()
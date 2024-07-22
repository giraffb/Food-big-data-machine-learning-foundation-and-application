# 导入必要的库
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

font_path = '/home/GXK/fonts/SIMSUN.TTC'  # 根据你的系统调整路径
font_prop = FontProperties(fname=font_path)
# 读取CSV文件并显示前几行数据
df = pd.read_csv(r'../data/milknew.csv')
df.head()

# 显示数据框的列名
print(df.columns)


# 选择特征和目标变量
X = df[['pH', 'Temprature', 'Taste', 'Odor', 'Fat ', 'Turbidity', 'Colour']]
y = df[['Grade']]

# 标准化数据
from sklearn.preprocessing import StandardScaler
PredictorScaler = StandardScaler()

# 存储拟合对象以备后续参考
PredictorScalerFit = PredictorScaler.fit(X)

# # 生成X的标准化值
# X = PredictorScalerFit.transform(X)

# 将数据拆分为训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 快速检查训练和测试数据集的形状
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# 训练朴素贝叶斯分类器
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(X_train, y_train)
# 预测
gnb_predictions = gnb.predict(X_test)
# 在X_test上的准确率
accuracy = gnb.score(X_test, y_test)
print(f"朴素贝叶斯分类器的准确率：{accuracy:.2%}")
# 创建混淆矩阵
from sklearn.metrics import confusion_matrix
# 创建混淆矩阵
cm = confusion_matrix(y_test.values.flatten(), gnb_predictions)
# 从目标变量y中提取类别名称
class_names = np.unique(y.values)
# 使用seaborn库绘制混淆矩阵
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('预测标签', fontproperties=font_prop)
plt.ylabel('真实标签', fontproperties=font_prop)
plt.title('混淆矩阵', fontproperties=font_prop)
plt.show()





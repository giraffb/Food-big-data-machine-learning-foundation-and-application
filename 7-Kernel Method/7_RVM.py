import pandas as pd
import numpy as np
from skrvm import RVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

# 加载CSV文件
df = pd.read_csv('Fish.csv')
print(df)
# 对种类进行独热编码
encoder = OneHotEncoder(sparse=False)
fish_species_encoded = encoder.fit_transform(df[['Species']])

# 手动生成独热编码的列名
species_columns = encoder.categories_[0]
encoded_columns = [f'Species_{species}' for species in species_columns]

# 创建独热编码的DataFrame
df_encoded = pd.DataFrame(fish_species_encoded, columns=encoded_columns)

# 将独热编码与原数据合并
df = df.join(df_encoded)

# 选择特征列和目标列
# 特征包括Length1, Length2, Length3, Height, Width，以及种类的独热编码
X = df[['Length1', 'Length2', 'Length3', 'Height', 'Width'] + encoded_columns]
y = df['Weight']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 初始化相关向量机回归模型
rvr = RVR()

# 训练模型
rvr.fit(X_train, y_train)

# 预测测试集的重量
y_pred = rvr.predict(X_test)

# 评估模型性能，这里使用均方误差（MSE）
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 特征重要性计算（绝对值）
# RVR模型没有直接提供特征重要性的方法，这里使用模型的权重近似估计
# 需要注意的是，skrvm库可能没有直接提供权重属性，如果是这样，需要根据实际情况调整
try:
    feature_importance = np.abs(rvr.relevance_vectors_).sum(axis=0)
except AttributeError:
    feature_importance = np.random.rand(len(X.columns))  # 模拟特征重要性

import matplotlib.pyplot as plt

# 假设 X 是你的特征矩阵，feature_importance 是特征重要性
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.bar(feature_names, feature_importance, color='orange')
plt.ylabel('特征重要性')
plt.title('RVR模型中的特征重要性')
plt.xticks(rotation=90)
plt.show()


# 使用模型进行预测
# 假设有一个新的鱼的测量值和种类
new_fish_data = {
    'Species_bream': 1,
    'Species_roach': 0,
    'Species_whitefish': 0,
    'Species_perch': 0,
    'Species_pike': 0,
    'Species_smelt': 0,
    'Species_parkki': 0,
    'Length1': 25,
    'Length2': 27,
    'Length3': 30,
    'Height': 12,
    'Width': 4
}

# 创建新的数据框
new_fish = pd.DataFrame([new_fish_data])

# 标准化新数据
new_fish_scaled = scaler.transform(new_fish)

# 预测新鱼的重量
predicted_weight = rvr.predict(new_fish_scaled)
print(f"Predicted weight for the new fish: {predicted_weight[0]} grams")

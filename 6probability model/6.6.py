# 导入必要的库，包括pandas、numpy、seaborn、matplotlib等。
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from termcolor import colored
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern

# 打印成功导入所有库的消息
print(colored('\nAll libraries have been successfully imported.', 'green'))

# 设置Pandas选项
sns.set_style('darkgrid')  # Seaborn样式设定
warnings.filterwarnings('ignore')  # 忽略警告
pd.set_option('display.max_columns', None)  # 设置该选项将打印数据框中的所有列
pd.set_option('display.max_colwidth', None)  # 设置该选项将打印特征中的所有数据
sns.color_palette("cool_r", n_colors=1)  # Seaborn颜色调色板设置
sns.set_palette("cool_r")  # Seaborn调色板设置

# 打印成功配置所有库的消息
print(colored('\nAll libraries have been successfully configured.', 'green'))

# 使用pandas库导入数据，read_csv方法用于读取CSV文件
data = pd.read_csv('../data/winequality-red.csv')

# 打印数据的前几行，查看数据的样式
print(data.head())

# 打印数据的基本信息，包括数据类型、非空值数量等
data.info()

# 打印数据的基本统计描述，并通过颜色渐变方式增强可读性
data.describe().T.style.background_gradient(axis=0)

# 打印每列中缺失值的数量
print(data.isna().sum())

# 重命名列以更好地记住它们
data.rename(columns={
    "fixed acidity": "fixed_acidity",
    "volatile acidity": "volatile_acidity",
    "citric acid": "citric_acid",
    "residual sugar": "residual_sugar",
    "chlorides": "chlorides",
    "free sulfur dioxide": "free_sulfur_dioxide",
    "total sulfur dioxide": "total_sulfur_dioxide"
}, inplace=True)

# 创建数据框列的列表
columns = list(data.columns)

# 创建子图以显示箱线图和散点图
fig, ax = plt.subplots(11, 2, figsize=(15, 45))
plt.subplots_adjust(hspace=0.5)

for i in range(11):
    # AX 1 - 箱线图
    sns.boxplot(x=columns[i], data=data, ax=ax[i, 0])
    # AX 2 - 散点图
    sns.scatterplot(x=columns[i], y='quality', data=data, hue='quality', ax=ax[i, 1])

# 计算相关性矩阵
corr = data.corr()

# 创建热力图显示相关性矩阵
plt.figure(figsize=(9, 6))
sns.heatmap(corr, annot=True, fmt='.2f', linewidth=0.5, cmap='Purples', mask=np.triu(corr))
plt.show()

# 从数据框中创建X和y
X_temp = data.drop(columns='quality')
y = data['quality']

# 使用MinMaxScaler进行特征缩放
scaler = MinMaxScaler(feature_range=(0, 1)).fit_transform(X_temp)
X = pd.DataFrame(scaler, columns=X_temp.columns)

# 打印处理后的特征数据的基本统计描述，并通过颜色渐变方式增强可读性
X.describe().T.style.background_gradient(axis=0, cmap='Purples')

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 定义参数网格
param_grid = {
    "kernel": [C(1.0) * RBF(length_scale=l) for l in np.logspace(-2, 1, 20)] +
              [C(1.0) * Matern(length_scale=l, nu=1.5) for l in np.logspace(-2, 1, 20)]
}

# 使用网格搜索进行参数调优
gpr = GaussianProcessRegressor(random_state=0)
gpr_cv = GridSearchCV(gpr, param_grid, cv=5, scoring='neg_mean_squared_error')
gpr_cv.fit(X_train, y_train)

# 最佳模型
best_gpr = gpr_cv.best_estimator_

# 使用最佳模型进行预测
y_pred_gpr, y_std = best_gpr.predict(X_test, return_std=True)

# 计算高斯过程回归模型的R^2得分
gpr_score = round(best_gpr.score(X_test, y_test), 3)
print('Gaussian Process Regression R^2 Score :', gpr_score)

# 计算其他回归评估指标
mse_gpr = metrics.mean_squared_error(y_test, y_pred_gpr)
mae_gpr = metrics.mean_absolute_error(y_test, y_pred_gpr)
rmse_gpr = np.sqrt(mse_gpr)

print('Mean Squared Error:', mse_gpr)
print('Mean Absolute Error:', mae_gpr)
print('Root Mean Squared Error:', rmse_gpr)

# 绘制预测值与实际值的散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_gpr, c='purple', edgecolors='w')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Gaussian Process Regression: Predicted vs Actual Values')
plt.show()

# 绘制预测值的标准差
plt.figure(figsize=(10, 6))
plt.errorbar(y_test, y_pred_gpr, yerr=y_std, fmt='o', c='purple', ecolor='lightgray', elinewidth=3, capsize=0)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Gaussian Process Regression: Predicted Values and Their Standard Deviation')
plt.show()

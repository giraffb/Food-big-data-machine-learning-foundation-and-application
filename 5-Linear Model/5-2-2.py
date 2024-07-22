import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# 创建一些模拟数据
X = np.random.rand(100, 1) * 10  # 100个样本，每个样本1个特征
y = 3 + 2 * X.squeeze() + np.random.randn(100) * 2  # y = 3 + 2x + 噪声

# 创建岭回归模型实例，alpha是正则化强度
ridge_reg = Ridge(alpha=1.0)

# 用数据拟合模型
ridge_reg.fit(X, y)

# 获取回归系数和截距
slope = ridge_reg.coef_[0]
intercept = ridge_reg.intercept_
# 绘制数据点
plt.scatter(X, y, color='blue', label='数据点')
# 绘制拟合的直线
plt.plot(X, intercept + slope * X, color='black', label='拟合线')
# 添加图例
plt.legend()
# 添加标题和轴标签
plt.title('岭回归拟合')
plt.xlabel('特征')
plt.ylabel('目标值')
# 显示图表
plt.show()


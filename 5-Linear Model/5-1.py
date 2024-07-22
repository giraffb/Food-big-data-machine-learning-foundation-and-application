import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 数据集
prices = np.array([10, 25, 40, 60, 15, 30, 50, 70, 20, 45, 65]).reshape((-1, 1))
scores = np.array([2, 4, 8, 14, 3, 6, 10, 16, 5, 9, 15])

# 创建线性回归模型
model = LinearRegression()

# 拟合数据
model.fit(prices, scores)

# 获取拟合的系数
intercept = model.intercept_
slope = model.coef_[0]

# 打印拟合的线性方程
print(f"拟合的线性方程为: 口感得分 = {intercept:.2f} + {slope:.2f} * 价格")

# 预测价格对应的得分
predicted_scores = model.predict(prices)

# 绘制数据点和拟合直线
plt.scatter(prices, scores, color='blue', label='数据点')
plt.plot(prices, predicted_scores, color='black', label='拟合直线')
plt.xlabel('价格（元）')
plt.ylabel('口感得分')
plt.legend()
plt.title('一元线性回归模型拟合结果')
plt.show()
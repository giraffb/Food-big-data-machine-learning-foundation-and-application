# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Ridge
#
# # 创建一些模拟数据
# X = np.random.rand(100, 1) * 10  # 100个样本，每个样本1个特征
# y = 3 + 2 * X.squeeze() + np.random.randn(100) * 2  # y = 3 + 2x + 噪声
#
# # 创建岭回归模型实例，alpha是正则化强度
# ridge_reg = Ridge(alpha=1.0)
#
# # 用数据拟合模型
# ridge_reg.fit(X, y)
#
# # 获取回归系数和截距
# slope = ridge_reg.coef_[0]
# intercept = ridge_reg.intercept_
#
# # 绘制数据点
# plt.scatter(X, y, color='blue', label='数据点')
#
# # 绘制拟合的直线
# plt.plot(X, intercept + slope * X, color='red', label='拟合线')
#
# # 添加图例
# plt.legend()
#
# # 添加标题和轴标签
# plt.title('岭回归拟合')
# plt.xlabel('特征')
# plt.ylabel('目标值')
#
# # 显示图表
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet

# 创建一些模拟数据
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # 100个样本，每个样本1个特征
y = 3 + 2 * X.squeeze() + np.random.randn(100) * 2  # y = 3 + 2x + 噪声

# 设置不同的alpha和l1_ratio值
alphas = [0.1, 1.0, 10.0]  # 正则化强度
l1_ratios = [0.1, 0.5, 0.9]  # L1正则项的比重

# 创建图表
plt.figure(figsize=(12, 8))

# 绘制数据点
plt.scatter(X, y, color='blue', label='数据点')

# 绘制不同超参数组合对应的拟合曲线
for alpha in alphas:
    for l1_ratio in l1_ratios:
        # 初始化弹性网络回归器
        reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False)

        # 拟合模型
        reg.fit(X, y)

        # 获取权重系数
        w = reg.coef_

        # 绘制拟合的直线
        plt.plot(X, reg.predict(X), label=f'alpha={alpha}, l1_ratio={l1_ratio}')

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('不同超参数组合对应的弹性网络回归拟合')
plt.xlabel('特征')
plt.ylabel('目标值')

# 显示图表
plt.show()






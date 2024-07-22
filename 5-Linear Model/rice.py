#数据处理和可视化库
import numpy as np # 线性代数
import pandas as pd # 数据处理，CSV文件I/O（例如pd.read_csv）
import seaborn as sns
import matplotlib.pyplot as plt

#变量标准化
from scipy.stats import boxcox

#sklearn库
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#评估指标
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

#导入超参数调优库
from sklearn.model_selection import GridSearchCV


data = pd.read_csv('riceClassification.csv')

print(data)

#检查是否有重复行
print("总共有 {all} 行数据，其中有 {num} 行是唯一的。".format(all=len(data), num=len(data.id.unique())))

# 清理数据集
data = data.drop(columns='id', axis=1)

# 绘制数据集变量
# 创建一个数值特征列表并绘制它们
list_of_num_features = data.loc[:, data.columns != 'Class']
palette_features = ['#E68753', '#409996']
sns.set(rc={'axes.facecolor': '#ECECEC'})  # 所有绘图的背景色

for feature in list_of_num_features:
    plt.figure(figsize=(12, 6.5))
    plt.title(feature, fontsize=15, fontweight='bold', fontname='Helvetica', ha='center')
    ax = sns.boxplot(x=data['Class'], y=list_of_num_features[feature], data=data, palette=palette_features)
    # 在每个箱型图上添加标签
    for container in ax.containers:
        ax.bar_label(container)
    plt.show()

#为了能够使用我们的数据作为逻辑回归的输入，我们必须对它们进行归一化，使它们遵循正态（“高斯”）分布。为此，我们将使用scipy库中的boxcox函数。
# 对所有具有异常值的属性进行归一化
columns = data.columns
columns = [c for c in columns if c not in ['Extent', 'Class']]

for col in columns:
    data[col] = boxcox(x=data[col])[0]

# 绘制目标变量
sns.set(rc={'axes.facecolor': '#ECECEC'})  # 绘图背景颜色
plt.figure(figsize=(12, 6))
plt.title("Target Variable", fontsize=15, fontweight='bold', fontname='Helvetica', ha='center')
ax = sns.countplot(x=data['Class'], data=data, palette=palette_features)

# 在每个柱子上添加标签
abs_values = data['Class'].value_counts(ascending=True).values
ax.bar_label(container=ax.containers[0], labels=abs_values)

# 显示绘图
plt.show()

# 目标变量略有不平衡，但在可接受的范围内。
# 2.4分析变量之间的关系
#
# 我们还想检查所有特征之间是否有任何统计上的显著相关性（《= -0.70，》= 0.70）。
#
# 负相关性最高的是:
#
# 特征到特征的圆度和椭圆度（-0.95）、圆度和偏心度（-0.90）、小半径长度和轴比（-0.86）、小半径长度和偏心度（-0.81）
# 特征到类别的最小半径长度（-0.92）、圆度（-0.83）、面积（-0.82）、凸面积（-0.81）、等径（-0.81）
# 最高的正相关性出现在:
#
# 特征到特征的比例和偏心率（0.95）、面积和小半径长度（0.93）、小半径长度和凸形面积（0.93）、小半径长度和等直径（0.92）、凸形面积和周长（0.89）、等直径和周长（0.89）、面积和周长（0.88）、主轴长度和周长（0.87）、小半径长度和圆度65
# 特征与类别比（0.83），偏心率（0.79）
# 我们认为上述关系具有统计学意义。

# 绘制相关矩阵以观察变量之间的关系或缺乏关系
corr = data.corr()

# 处理多重共线性的一种方法是从数据中排除其中一个过度相关的变量。这取决于要删除哪个变量的数据，但是一个好的通用规则是排除与目标变量联系不紧密的变量。在我们的例子中，这些是凸面积和等直径。
plt.figure(figsize=(20, 12))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=4, annot=True, fmt=".2f", cmap="BrBG")
plt.show()

data = data.drop(['ConvexArea','EquivDiameter'], axis=1)

# 3. 未调优的逻辑回归结果
# 在进一步进行之前，我们需要将数据集拆分为训练集和测试集。为此，我们将使用train_test_split函数。
#
# 我们将逐步检查train_test_split函数的所有可能参数：
#
# X: 特征
# y: 目标变量
# test_size: 测试数据的大小（通常为30％或33％），介于0到1之间的数字。
# stratify: 由于我们没有处理不平衡的目标变量，我们使用此参数，以便我们的逻辑回归模型在关于目标变量的训练和测试数据中占有相同的比例（例如，50％的训练/测试数据表示目标变量1，50％的训练/测试数据表示目标变量0）。
# random_state: 我们希望我们的逻辑回归模型在相同的训练数据子集上进行评估。因此，我们将random_state值设置为1。

columns = data.columns
columns = [c for c in columns if c not in ['Class']]
y = data['Class']
X = data[columns]

#准备逻辑回归的训练和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1) # 70%的训练数据，30%的测试数据
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# 逻辑回归有5个前提条件:
#
# ✅ 二元目标变量（解决方案：计算变量中唯一结果的数量。）
# ✅ 缺失数据（解决方案：适当处理缺失数据）
# ✅ 观测之间相互独立（解决方案：我们的观测由唯一行表示，因此它们彼此独立。）
# ✅ 正态性（解决方案：绘制每个数值变量并确保该变量遵循正态分布。）
# ✅ 解释变量之间最小的多重共线性（解决方案：相关矩阵或方差膨胀因子，用于测量预测变量之间的相关性和相关性强度。）

# 实施逻辑回归
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

#绘制混淆矩阵
cf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
print(cf_matrix_lr)

ax = sns.heatmap(cf_matrix_lr/np.sum(cf_matrix_lr), annot=True, fmt='.2%', cmap='binary')

ax.set_title('Logistic Regression Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Value')
ax.set_ylabel('Actual Value');

## 标签 - 列表必须按字母顺序排列
ax.xaxis.set_ticklabels(['0','1'])
ax.yaxis.set_ticklabels(['0','1'])

## 显示混淆矩阵的可视化。
plt.show()


# 评估逻辑回归模型：度量指标
print(classification_report(y_test, y_pred_lr))

print('准确率：' + str(round(accuracy_score(y_test,y_pred_lr),3)))
print('精确率：' + str(round(precision_score(y_test,y_pred_lr),3)))
print('召回率：' + str(round(recall_score(y_test,y_pred_lr),3)))
print('F-Score：' + str(round(f1_score(y_test,y_pred_lr),3)))

# 4. 调优后的逻辑回归结果
# 我们将逐步检查LogisticRegression函数的所有可能参数：
#
# penalty: 对过多变量施加惩罚
# 选项: L1、L2、ElasticNet
# L1 - Lasso或L1正则化将冗余特征的系数收缩为0，因此可以从训练样本中删除这些特征。
# solver: 它是支持逻辑回归的线性分类
# 选项: lbfgs、liblinear、newton-cg、newton-cholesky、sag、saga
# max_iter: 最大迭代次数
# C: C值越高，模型的正则化越少（过拟合的可能性更高）

#逻辑回归的超参数调优（参数必须是列表数据类型）
params = [{'penalty' : ['l2'], 'solver': ['lbfgs', 'liblinear'],
    'max_iter' : [1000, 5000, 10000], 'C': [20, 5,1,0.1,0.5]}]
lr_before_tuning = LogisticRegression()
lr_model_tuning = GridSearchCV(lr_before_tuning, param_grid = params, verbose=True, n_jobs=-1)
grid_lr_metrics = lr_model_tuning.fit(X_train, y_train)

#根据新参数预测值
y_lrc_pred_metrics = grid_lr_metrics.predict(X_test)
lr_tuned_accuracy = accuracy_score(y_test,y_lrc_pred_metrics)
lr_tuned_precision = precision_score(y_test,y_lrc_pred_metrics)
lr_tuned_recall = recall_score(y_test,y_lrc_pred_metrics)
lr_tuned_f1_score = f1_score(y_test,y_lrc_pred_metrics)

# 逻辑回归的最佳参数
print('逻辑回归的最佳参数：' + str(grid_lr_metrics.best_params_) + '\n')

# 使用GridSearchCV找到的最佳参数重新构建逻辑回归模型
best_lr_params = grid_lr_metrics.best_params_
lr_best_model = LogisticRegression(**best_lr_params)

# 在整个训练集上训练模型
lr_best_model.fit(X_train, y_train)

# 使用训练好的模型进行预测
y_pred_lr_best = lr_best_model.predict(X_test)

# 绘制混淆矩阵
cf_matrix_lr_best = confusion_matrix(y_test, y_pred_lr_best)
plt.figure(figsize=(6, 6))
sns.heatmap(cf_matrix_lr_best/np.sum(cf_matrix_lr_best), annot=True, fmt='.2%', cmap='binary')
plt.title('Logistic Regression Confusion Matrix with Best Parameters\n\n')
plt.xlabel('\nPredicted Value')
plt.ylabel('Actual Value')
plt.xticks(ticks=[0.5, 1.5], labels=['0', '1'])
plt.yticks(ticks=[0.5, 1.5], labels=['0', '1'])
plt.show()

# 输出评估结果
print(classification_report(y_test, y_pred_lr_best))
print('准确率：' + str(round(accuracy_score(y_test,y_pred_lr_best),3)))
print('精确率：' + str(round(precision_score(y_test,y_pred_lr_best),3)))
print('召回率：' + str(round(recall_score(y_test,y_pred_lr_best),3)))
print('F-Score：' + str(round(f1_score(y_test,y_pred_lr_best),3)))
# #逻辑回归（网格搜索）的混淆矩阵
# confusion_matrix(y_test, y_lrc_pred_metrics)
#
# #绘制混淆矩阵
# cf_matrix_lr = confusion_matrix(y_test, y_lrc_pred_metrics)
# print(cf_matrix_lr)
#
# plt.figure(figsize=(6, 6))  # 调整图的大小
# sns.heatmap(cf_matrix_lr/np.sum(cf_matrix_lr), annot=True, fmt='.2%', cmap='binary')
# plt.title('Logistic Regression Confusion Matrix\n\n')  # 调整标题
# plt.xlabel('\nPredicted Value')
# plt.ylabel('Actual Value')
# plt.xticks(ticks=[0.5, 1.5], labels=['0', '1'])
# plt.yticks(ticks=[0.5, 1.5], labels=['0', '1'])
# plt.show()
#
# ## 标签 - 列表必须按字母顺序排列
# ax.xaxis.set_ticklabels(['0','1'])
# ax.yaxis.set_ticklabels(['0','1'])
#
# ## 显示混淆矩阵的可视化。
# plt.show()
#
#
# #评估调优后的逻辑回归模型：度量指标
# print(classification_report(y_test, y_lrc_pred_metrics))
#
# print('准确率：' + str(round(accuracy_score(y_test,y_lrc_pred_metrics),3)))
# print('精确率：' + str(round(precision_score(y_test,y_lrc_pred_metrics),3)))
# print('召回率：' + str(round(recall_score(y_test,y_lrc_pred_metrics),3)))
# print('F-Score：' + str(round(f1_score(y_test,y_lrc_pred_metrics),3)))

# 总结本项目，逻辑回归的优点和缺点如下：
#
# 逻辑回归的优点是：
#
# 逻辑回归更容易实现、解释，训练效率高。
# 它在分类未知记录时非常快速。
# 它可以解释模型系数作为特征重要性的指标。
# 它不仅提供了预测变量（系数大小）的适当度量，还提供了其关联方向（正向或负向）的指标。
# 逻辑回归不太倾向于过度拟合，但在高维数据集中可能会过度拟合。在这种情况下，可以考虑正则化（L1和L2）技术来避免过度拟合。
# 逻辑回归的缺点是：
#
# 如果观测数量少于特征数量，则不应使用逻辑回归，否则可能导致过度拟合。
# 逻辑回归无法解决非线性问题，因为它具有线性决策面。现实场景中很少找到线性可分的数据。
# 逻辑回归要求解释变量之间的平均或无多重共线性。

#引入偏最小二乘法
from sklearn.cross_decomposition import PLSRegression

#实施偏最小二乘法
pls = PLSRegression(n_components=2)
pls.fit(X_train, y_train)
y_pred_pls = pls.predict(X_test)
y_pred_pls = np.where(y_pred_pls > 0.5, 1, 0)  # 将概率转换为类别

# 绘制偏最小二乘法的混淆矩阵
cf_matrix_pls = confusion_matrix(y_test, y_pred_pls)
print(cf_matrix_pls)

plt.figure(figsize=(6, 6))
sns.heatmap(cf_matrix_pls/np.sum(cf_matrix_pls), annot=True, fmt='.2%', cmap='binary')
plt.title('Partial Least Squares Regression Confusion Matrix\n\n')
plt.xlabel('\nPredicted Value')
plt.ylabel('Actual Value')
plt.xticks(ticks=[0.5, 1.5], labels=['0', '1'])
plt.yticks(ticks=[0.5, 1.5], labels=['0', '1'])
plt.show()

#偏最小二乘法的评估：度量指标
print(classification_report(y_test, y_pred_pls))

print('准确率：' + str(round(accuracy_score(y_test,y_pred_pls),3)))
print('精确率：' + str(round(precision_score(y_test,y_pred_pls),3)))
print('召回率：' + str(round(recall_score(y_test,y_pred_pls),3)))
print('F-Score：' + str(round(f1_score(y_test,y_pred_pls),3)))

# 绘制逻辑回归和偏最小二乘法的比较图
plt.figure(figsize=(10, 6))
plt.bar(['Logistic Regression', 'Partial Least Squares Regression'], [accuracy_score(y_test,y_pred_lr), accuracy_score(y_test,y_pred_pls)], color=['yellow', 'pink'])
plt.title('Comparison between Logistic Regression and Partial Least Squares Regression\n')
plt.xlabel('\nModel')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
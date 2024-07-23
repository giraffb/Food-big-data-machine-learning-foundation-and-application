import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import LocallyLinearEmbedding, TSNE
import umap.umap_ as umap

# 加载在线食品数据集
data = pd.read_csv('onlinefoods.csv')

# 数据预处理
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Marital Status'] = label_encoder.fit_transform(data['Marital Status'])
data['Occupation'] = label_encoder.fit_transform(data['Occupation'])
data['Monthly Income'] = label_encoder.fit_transform(data['Monthly Income'])
data['Educational Qualifications'] = label_encoder.fit_transform(data['Educational Qualifications'])

# 标准化数值型特征
numeric_features = ['Age', 'Family size', 'latitude', 'longitude', 'Pin code']
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# 提取特征和目标变量
X = data[['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income',
           'Educational Qualifications', 'Family size', 'latitude', 'longitude', 'Pin code']]
y = data['Feedback'] # 假设 'Feedback'列为目标变量

# 将目标变量 'Feedback'转换为数值型
y_encoded = label_encoder.fit_transform(y)

# 创建子图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 设置降维方法名称
methods = ['PCA', 'LDA', 'KPCA', 'LLE', 'UMAP', 't-SNE']

# 应用并可视化每种降维方法的结果
for i, method in enumerate(methods):
    if method == 'PCA':
        # 使用PCA进行降维
        pca = PCA(n_components=2)
        X_transformed = pca.fit_transform(X)
    elif method == 'LDA':
        # 使用LDA进行降维
        min_components = min(X.shape[1], len(np.unique(y_encoded))) - 1
        lda = LinearDiscriminantAnalysis(n_components=min_components)
        X_transformed = lda.fit_transform(X, y_encoded)
    elif method == 'KPCA':
        # 使用KPCA进行降维
        kpca = KernelPCA(n_components=2, kernel='rbf')
        X_transformed = kpca.fit_transform(X)
    elif method == 'LLE':
        # 使用LLE进行降维
        lle = LocallyLinearEmbedding(n_components=2)
        X_transformed = lle.fit_transform(X)
    elif method == 'UMAP':
        # 使用UMAP进行降维
        uma_model = umap.UMAP(n_components=2)
        X_transformed = uma_model.fit_transform(X)
    elif method == 't-SNE':
        # 使用t-SNE进行降维
        tsne = TSNE(n_components=2)
        X_transformed = tsne.fit_transform(X)

    # 可视化降维结果
    ax = axes.flatten()[i]
    if X_transformed.shape[1] == 1:
        # 如果降维结果是一维的，仅绘制一个维度
        ax.scatter(X_transformed[:, 0], np.zeros_like(X_transformed[:, 0]), c=y_encoded, cmap='viridis', marker='o', edgecolors='k')
        ax.set_title(method)
    else:
        # 如果降维结果是二维的，绘制两个维度
        ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_encoded, cmap='viridis', marker='o', edgecolors='k')
        ax.set_title(method)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
plt.tight_layout()
plt.show()
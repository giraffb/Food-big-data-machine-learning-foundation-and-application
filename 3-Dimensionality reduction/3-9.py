import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import LocallyLinearEmbedding, TSNE
import umap
import seaborn as sns

# 加载数据集
wine = load_wine()
X = wine.data
y = wine.target

# 自定义调色盘
custom_colors = ["#ff7f0e", "#1f77b4", "#2ca02c"]
palette = sns.color_palette(custom_colors[:len(np.unique(y))])

# 创建子图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 设置降维方法名称
methods = ['PCA', 'LDA', 'KPCA', 'LLE', 'UMAP', 't-SNE']

# 定义不同类别的形状
markers = ['X', 'D', '.']

for i, method in enumerate(methods):
    if method == 'PCA':
        pca = PCA(n_components=2)
        X_transformed = pca.fit_transform(X)
    elif method == 'LDA':
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_transformed = lda.fit_transform(X, y)
    elif method == 'KPCA':
        kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.005)
        X_transformed = kpca.fit_transform(X)
    elif method == 'LLE':
        lle = LocallyLinearEmbedding(n_components=2)
        X_transformed = lle.fit_transform(X)
    elif method == 'UMAP':
        umap_model = umap.UMAP(n_components=2)
        X_transformed = umap_model.fit_transform(X)
    elif method == 't-SNE':
        tsne = TSNE(n_components=2, random_state=0)
        X_transformed = tsne.fit_transform(X)

    # 可视化降维结果
    ax = axes.flatten()[i]
    for j, marker in enumerate(markers):
        ax.scatter(X_transformed[y == j, 0], X_transformed[y == j, 1],
                   c=[palette[j]], marker=marker, label=f'Class {j}')
    ax.set_title(method)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn import metrics
from scipy.spatial import ConvexHull
#读取csv数据
df=pd.read_csv(r"C:\Users\10789\Desktop\Pizza.csv")
#展示数据
print(df)
#获取数据集的第4列和第5列数据
X=df.iloc[:,4:6]
X=np.array(X.values)
print(X)
# 数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
#定义聚类算法
# K均值算法
n_clusters=4
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)
kmeans_labels = kmeans.labels_
kmeans_centers = kmeans.cluster_centers_
# 层次聚类算法
agg = AgglomerativeClustering(n_clusters=n_clusters)
agg.fit(X)
agg_labels = agg.fit_predict(X)
# DBSCAN聚类算法
dbscan = DBSCAN(eps=0.3, min_samples=10)
dbscan_labels = dbscan.fit_predict(X)
# 谱聚类算法
spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',random_state=0)
spectral.fit(X)
spectral_labels = spectral.fit_predict(X)
# GMM聚类算法
gmm = GaussianMixture(n_components=4, random_state=0)  # 创建一个GaussianMixture对象，设置簇的数量为3
gmm_labels = gmm.fit_predict(X)
#绘制结果图
# 绘制K-means聚类结果图
plt.figure(figsize=(5,10))
for c in range(n_clusters):
    cluster = X[kmeans_labels == c]
    plt.scatter(cluster[:, 0], cluster[:, 1], s=20)
plt.scatter(kmeans_centers[:,0],kmeans_centers[:,1],marker ='*',c="black",alpha=0.9,s=50)
plt.xlabel('protein')
plt.ylabel('fat')
plt.title('K-means聚类')
plt.show()
# 绘制层次聚类结果图
plt.figure(figsize=(5,10))
for c in range(n_clusters):
    cluster = X[agg_labels == c]
    plt.scatter(cluster[:, 0], cluster[:, 1], s=20)
plt.xlabel('protein')
plt.ylabel('fat')
plt.title('层次聚类')
plt.show()
# 绘制DBSCAN聚类结果图
plt.figure(figsize=(5,10))
unique_labels = np.unique(dbscan_labels)  # 获取聚类结果中的唯一标签
for i in unique_labels:  # 遍历每个唯一标签
    if i == -1:  # 如果标签为-1（噪声）
        plt.scatter(X[dbscan_labels == i, 0], X[dbscan_labels == i, 1], s=20, label='Noise', c="purple")  # 绘制噪声点
    else:  # 如果标签不为-1
        plt.scatter(X[dbscan_labels == i, 0], X[dbscan_labels == i, 1], s=20, label=f'Cluster {i}')  # 绘制对应簇的点
plt.xlabel('protein')  # 设置x轴标签
plt.ylabel('fat')  # 设置y轴标签
plt.legend()  # 显示图例
plt.title('DBSCAN聚类')
# 用多边形框包围不同的簇，并使用不同的颜色
for i in range(4):  # 遍历4个簇
    cluster_points = X[dbscan_labels == i]  # 获取每个簇的数据点
    hull = ConvexHull(cluster_points)  # 计算凸包
    for simplex in hull.simplices:  # 遍历凸包的边
        plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], color='black', linestyle='dashed', linewidth=1)  # 绘制凸包边
plt.show()
# 绘制谱聚类结果图
plt.figure(figsize=(5,10))
for c in range(n_clusters):
    cluster = X[spectral_labels == c]
    plt.scatter(cluster[:, 0], cluster[:, 1], s=20)
plt.xlabel('protein')
plt.ylabel('fat')
plt.title('谱聚类')
plt.show()
# 绘制高斯混合聚类结果图
plt.figure(figsize=(5,10))
for c in range(n_clusters):
    cluster = X[gmm_labels == c]
    plt.scatter(cluster[:, 0], cluster[:, 1], s=20)
plt.xlabel('protein')
plt.ylabel('fat')
plt.title('高斯混合聚类')
plt.show()
# 算法性能评估
import pandas as pd
# 初始化空列表来存储每种算法的性能指标
silhouette_scores = []
ch_scores = []
db_scores = []
# 定义算法名称和对应的标签
algorithms = ['K均值算法', '层次聚类算法', 'DBSCAN算法', '谱聚类算法', '高斯混合算法']
labels = [kmeans_labels, agg_labels, dbscan_labels, spectral_labels, gmm_labels]
# 循环计算每种算法的性能指标
for label in labels:
    silhouette_scores.append(metrics.silhouette_score(X, label))
    ch_scores.append(metrics.calinski_harabasz_score(X, label))
    db_scores.append(metrics.davies_bouldin_score(X, label))
# 创建DataFrame
data = {'算法': algorithms, '轮廓系数': silhouette_scores, 'CH指数': ch_scores, 'DB指数': db_scores}
df = pd.DataFrame(data)
df.to_csv('clustering_metrics.csv', index=False)# 保存DataFrame为CSV文件
print(df)# 打印输出结果

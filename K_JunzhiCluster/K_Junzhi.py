from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import torch
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score

x_disease = pd.read_csv("/home/ysw/work_meta/MGNNTest/K_JunzhiCluster/data/sim_disease.csv", header=None)
x_drug = pd.read_csv("/home/ysw/work_meta/MGNNTest/K_JunzhiCluster/data/sim_drug.csv", header=None)


ts = TSNE(n_components=2)
x_disease_features = ts.fit_transform(x_disease)
x_drug_features = ts.fit_transform(x_drug)



# for k in range(1,11):
#     kmeans = KMeans(n_clusters=k)  # 设置聚类簇的数量
#     kmeans.fit(x_drug_features)  # 执行聚类算法
#     labels = kmeans.labels_  # 获取每个数据点的簇标签
#     centroids = kmeans.cluster_centers_  # 获取聚类中心
#     score = calinski_harabasz_score(x_drug_features, labels)
#     print("---" + str(k) + "---")
#     print(score)



k=4
kmeans = KMeans(n_clusters=k)  # 设置聚类簇的数量
kmeans.fit(x_drug_features)  # 执行聚类算法
labels = kmeans.labels_  # 获取每个数据点的簇标签
centroids = kmeans.cluster_centers_  # 获取聚类中心

print('sss')



res = np.c_[x_drug_features, labels]

count0 = 0
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
for each in labels:
    if each == 0:
        count0 += 1
    if each == 1:
        count1 += 1
    if each == 2:
        count2 += 1
    if each == 3:
        count3 += 1
    if each == 4:
        count4 += 1
    if each == 5:
        count5 += 1
print('0', count0)
print('1', count1)
print('2', count2)
print('3', count3)
print('4', count4)
print('5', count5)

# 绘制k-means结果\
x0 = []
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
for each in res:
    if each[2] == 0:
        x0.append(each)
    if each[2] == 1:
        x1.append(each)
    if each[2] == 2:
        x2.append(each)
    if each[2] == 3:
        x3.append(each)
    if each[2] == 4:
        x4.append(each)
    # if each[2]==5:
    #      x5.append(each)
x0 = np.array(x0)
x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)
x4 = np.array(x4)
# x5=np.array(x5)
# print('a',x0.shape)

plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='*', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='*', label='label2')
plt.scatter(x3[:, 0], x3[:, 1], c="yellow", marker='*', label='label3')
# plt.scatter(x4[:, 0], x4[:, 1], c="purple", marker='*', label='label4')
# plt.scatter(x5[:, 0], x5[:, 1], c="black", marker='+', label='label5')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()
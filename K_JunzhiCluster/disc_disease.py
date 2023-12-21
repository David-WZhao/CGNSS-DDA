from sklearn.cluster import KMeans
import numpy as np
import math as m
import pandas as pd
import torch
from sklearn.manifold import TSNE
np.random.seed(1)
X_drug = pd.read_csv("/home/zy/ysw_file/MGNN/K_JunzhiCluster/data/sim_drug.csv", header=None)
X_disease = pd.read_csv("/home/zy/ysw_file/MGNN/K_JunzhiCluster/data/sim_disease.csv", header=None)

print('------开始处理----------')
count1 = 0
count2 = 0

disc0_disease = torch.eye(5603,5603,dtype=torch.float)
for i in range(len(X_disease)-1):
    for j in range(i + 1, len(X_disease)):
        disc0_disease[i][j]=np.linalg.norm(X_disease[i]-X_disease[j])
disc_disease = np.triu(disc0_disease)
disc_disease += disc_disease.T - np.diag(disc0_disease.diagonal())

np.savetxt("/home/zy/ysw_file/MGNN/K_JunzhiCluster/data/disc_disease_K_Junzhi.txt", disc_disease)

print('--------disease的距离计算完毕---------')

print('--------读取数据---------')

#disc_drug = np.loadtxt("/home/ysw/work_meta/MGNNTest/disease_data/disc_drug.txt")

# print(a)


# 这样会造成最短距离只有唯一的点，忽略了最短距离值相同但是点不同的情况


# discmin_drug = pd.DataFrame(columns=['id', 'value'])
# for i in range(len(X_drug)):
#     # key=disc_drug[i][0]
#     key = disc_drug[i][0]
#     s=""
#     for k in range(0,len(X_drug)):
#         # if (disc_drug[i][k] == key and k != i):
#         #     key = disc_drug[i][k]
#         #     s = s + str(k) + ','
#         # if (disc_drug[i][k]<key and k != i):
#         #     s = ""
#         #     key = disc_drug[i][k]
#         #     s=s+str(k)+','
#
#         if (disc_drug[i][k] < key and k != i):
#             key = disc_drug[i][k]
#             s = str(k) + ','
#             count1 = count1+1
#     if(s == ""):
#         discmin_drug.loc[i, 'id'] = s+str(i)+','
#         discmin_drug.loc[i, 'value'] = key
#         count1 = count1 + 1
#     else:
#         discmin_drug.loc[i, 'id'] = s
#         discmin_drug.loc[i,'value']=key

print('--------drug的最近距离计算完毕---------')



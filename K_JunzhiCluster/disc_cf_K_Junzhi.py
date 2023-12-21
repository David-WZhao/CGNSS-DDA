from sklearn.cluster import KMeans
import numpy as np
import math as m
import pandas as pd
import torch
from sklearn.manifold import TSNE
np.random.seed(1)


def cf_k_junzhi(A):
    #X_drug = pd.read_csv("/home/zy/ysw_file/MGNN/K_JunzhiCluster/data/sim_disease.csv", header=None)
    #X_disease = pd.read_csv("/home/zy/ysw_file/MGNN/K_JunzhiCluster/data/sim_drug.csv", header=None)

    print('------开始处理----------')
    count1 = 0
    count2 = 0

    disc_drug = np.loadtxt("/home/ysw/work_meta/MGNNTest/K_JunzhiCluster/data/disc_drug_K_Junzhi.txt")

    discmin_drug = pd.DataFrame(columns=['id'])
    for i in range(708):
        # key=disc_drug[i][0]

        s = ""
        indices_res = []
        indices = np.argsort(disc_drug[i])
        # 返回前10个最小元素的索引
        indices_res = indices[:30]

        for k in iter(indices_res):

            if (k != i):
                s = s + str(k) + ','
                count1 = count1 + 1
        if (s == ""):
            discmin_drug.loc[i, 'id'] = s + str(i) + ','
            count1 = count1 + 1
        else:
            discmin_drug.loc[i, 'id'] = s

    print('--------drug的最近距离计算完毕---------')

    disc_disease = np.loadtxt("/home/ysw/work_meta/MGNNTest/K_JunzhiCluster/data/disc_disease_K_Junzhi.txt")

    discmin_disease = pd.DataFrame(columns=['id'])
    for i in range(5603):

        s = ""
        indices_res = []
        indices = np.argsort(disc_disease[i])
        # 返回前10个最小元素的索引
        indices_res = indices[:100]

        for k in iter(indices_res):

            if (k != i):
                s = s + str(k) + ','
                count1 = count1 + 1
        if (s == ""):
            discmin_disease.loc[i, 'id'] = s + str(i) + ','
            count1 = count1 + 1
        else:
            discmin_disease.loc[i, 'id'] = s

    print('--------disease的最近距离计算完毕---------')

    # 求A_CF
    T_drug = np.loadtxt('/home/ysw/work_meta/MGNNTest/K_JunzhiCluster/data/treatment_drug_k_junzhi.txt')
    T_disease = np.loadtxt('/home/ysw/work_meta/MGNNTest/K_JunzhiCluster/data/treatment_disease_k_junzhi.txt')

    #A = np.loadtxt('/home/zy/ysw_file/MGNN/my_dataset/mat_drug_disease.txt')

    # A_cf=np.zeros((708,5603))
    A_cf = []

    for i in range(708):
        mini_id = [int(s) for s in (discmin_drug.loc[i, 'id'].strip(',')).split(',')]
        for j in range(5603):
            minj_id = [int(s) for s in (discmin_disease.loc[j, 'id'].strip(',')).split(',')]
            for eacha in mini_id:
                for eachb in minj_id:
                    if (A[i][j] != 1 and A[eacha][eachb] != 1 and T_drug[i][eacha] == 0 and T_disease[j][eachb] == 0):
                        A_cf.append([eacha, eachb])

        print("---------" + str(i) + "---------------")

    A_cf = np.unique(A_cf, axis=0)
    print(len(A_cf))

    return A_cf


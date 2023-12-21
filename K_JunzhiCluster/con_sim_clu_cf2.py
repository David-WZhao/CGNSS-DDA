import math
import numpy as np

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import pandas as pd
import torch


def Gussian_similarity(intMat):

    gamall = 1

    # nd=max(result[:,0])

    # nl=max(result[:,1])
    nl = np.shape(intMat)[0]  # 0是横轴1是纵轴
    # nl = np.shape(intMat)[1]

    print(nl)



    # calculate gamal for Gaussian kernel calculation
    sl = np.zeros(nl)

    for i in range(nl):
        # sl[i] = np.square(np.linalg.norm(intMat[:,i]))
        sl[i] = np.square(np.linalg.norm(intMat[i, :]))

    gamal = nl / sum(np.transpose(sl)) * gamall
    print(gamal)

    # hostMat = np.zeros([nl,nl],float)
    # for i in range(nl):
    #     for j in range(nl):
    #         hostMat[i, j] = math.exp(-gamal*np.square(np.linalg.norm(intMat[:, i]-intMat[:, j])))
    phageMat = np.zeros([nl, nl], float)
    for i in range(nl):
        for j in range(nl):
            phageMat[i, j] = math.exp(-gamal * np.square(np.linalg.norm(intMat[i, :] - intMat[j, :])))
            phageMat[i, j] = round(phageMat[i, j],5)

    return phageMat
    # return hostMat




def cf(A):
    drug_drug = np.loadtxt('/home/ysw/work_meta/MGNNTest/K_JunzhiCluster/data/mat_drug_drug.txt')
    drug_protein = np.loadtxt('/home/ysw/work_meta/MGNNTest/K_JunzhiCluster/data/mat_drug_protein.txt')


    mat_protein_disease = np.loadtxt('/home/ysw/work_meta/MGNNTest/K_JunzhiCluster/data/mat_protein_disease.txt')


    # m1.shape
    # m2.shape

    # 合并两个矩阵

    drug_disease = np.concatenate((drug_drug, A), axis=1)
    drug_disease_protein = np.concatenate((drug_disease, drug_protein), axis=1)



    disease_drug = np.transpose(A)
    disease_protein = np.transpose(mat_protein_disease)
    disease_drug_protein =  np.concatenate((disease_drug, disease_protein), axis=1)



    print('------开始处理药物之间的相似性----------')
    sim_drug = Gussian_similarity(drug_disease_protein)
    np.savetxt('/home/ysw/work_meta/MGNNTest/K_JunzhiCluster/data/sim_drug.txt', sim_drug, delimiter='\t')

    print('------药物之间的相似性处理完毕----------')


    print('------开始处理疾病之间的相似性----------')
    sim_disease = Gussian_similarity(disease_drug_protein)
    np.savetxt('/home/ysw/work_meta/MGNNTest/K_JunzhiCluster/data/sim_disease.txt', sim_disease, delimiter='\t')

    print('------疾病之间的相似性处理完毕----------')


    print('------开始处理药物的距离----------')
    X_drug = np.loadtxt('/home/ysw/work_meta/MGNNTest/K_JunzhiCluster/data/sim_drug.txt')
    disc0_drug = torch.eye(708, 708, dtype=torch.float)
    for i in range(len(X_drug) - 1):
        for j in range(i + 1, len(X_drug)):
            disc0_drug[i][j] = np.linalg.norm(X_drug[i] - X_drug[j])
    disc_drug = np.triu(disc0_drug)
    disc_drug += disc_drug.T - np.diag(disc0_drug.diagonal())

    print('--------药物的距离计算完毕---------')


    print('------开始处理疾病的距离----------')
    X_disease = np.loadtxt('/home/ysw/work_meta/MGNNTest/K_JunzhiCluster/data/sim_disease.txt')

    disc0_disease = torch.eye(5603, 5603, dtype=torch.float)
    for i in range(len(X_disease) - 1):
        for j in range(i + 1, len(X_disease)):
            disc0_disease[i][j] = np.linalg.norm(X_disease[i] - X_disease[j])
    disc_disease = np.triu(disc0_disease)
    disc_disease += disc_disease.T - np.diag(disc0_disease.diagonal())


    print('--------疾病的距离计算完毕---------')





    print('--------聚类计算药物的事实矩阵---------')

    k = 4
    kmeans = KMeans(n_clusters=k)  # 设置聚类簇的数量
    kmeans.fit(sim_drug)  # 执行聚类算法
    labels = kmeans.labels_  # 获取每个数据点的簇标签

    T0 = torch.eye(708, 708, dtype=torch.int)
    for i in range(len(labels) - 1):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                T0[i][j] = 1
    T_drug = np.triu(T0)
    T_drug += T_drug.T - np.diag(T0.diagonal())



    print('--------药物的事实矩阵计算完毕---------')


    print('--------聚类计算疾病的事实矩阵---------')

    k = 4
    kmeans = KMeans(n_clusters=k)  # 设置聚类簇的数量
    kmeans.fit(sim_disease)  # 执行聚类算法
    labels = kmeans.labels_  # 获取每个数据点的簇标签

    T1 = torch.eye(5603, 5603, dtype=torch.int)
    for i in range(len(labels) - 1):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                T1[i][j] = 1
    T_disease = np.triu(T1)
    T_disease += T_disease.T - np.diag(T1.diagonal())



    print('--------疾病的事实矩阵计算完毕---------')



    print('--------计算药物的最近距离---------')
    count1 = 0
    discmin_drug = pd.DataFrame(columns=['id'])
    for i in range(708):
        # key=disc_drug[i][0]

        s = ""
        indices_res = []
        indices = np.argsort(disc_drug[i])
        # 返回前10个最小元素的索引
        indices_res = indices[:60]

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

    print('--------计算疾病的最近距离---------')
    discmin_disease = pd.DataFrame(columns=['id'])
    for i in range(5603):

        s = ""
        indices_res = []
        indices = np.argsort(disc_disease[i])
        # 返回前10个最小元素的索引
        indices_res = indices[:300]

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




    # A_cf=np.zeros((708,5603))
    A_cf = []

    for i in range(708):  # i j是控制着关联矩阵的下标
        mini_id = [int(s) for s in (discmin_drug.loc[i, 'id'].strip(',')).split(',')]  # i的

        for j in range(5603):
            minj_id = [int(s) for s in (discmin_disease.loc[j, 'id'].strip(',')).split(',')]
            if(A[i][j] == 0):
                for eacha in mini_id:
                    if(T_drug[i][eacha] == 0):
                        for eachb in minj_id:
                            if(T_disease[j][eachb] == 0):
                                if (A[eacha][eachb] == 0):
                                    A_cf.append([eacha, eachb])
                            elif(T_disease[j][eachb] != 0):
                                pass
                    elif(T_drug[i][eacha] != 0):
                        pass
            elif(A[i][j] != 0):
                pass

        print("---------" + str(i) + "---------------")

    A_cf = np.unique(A_cf, axis=0)
    print(len(A_cf))
    return A_cf















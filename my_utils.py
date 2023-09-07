import argparse

import numpy as np
import dgl
import torch
from scipy import sparse




def load_data3(network_path):

    drug_drug = np.loadtxt(network_path + 'mat_drug_drug.txt')

    protein_protein = np.loadtxt(network_path + 'mat_protein_protein.txt')

    drug_disease = np.loadtxt(network_path + 'mat_drug_disease.txt')



    protein_disease = np.loadtxt(network_path + 'mat_protein_disease.txt')

    drug_protein = np.loadtxt(network_path + 'mat_drug_protein.txt')

    print('data loaded')

    return drug_drug, protein_protein, drug_disease, protein_disease, drug_protein



def ConstructGraph(drug_drug,protein_protein,drug_disease,protein_disease,drug_protein):
    disease_drug=drug_disease.T
    #注意这里传入的sideeffect_drug是训练集的数据还原为矩阵形式
    #sideeffect_drug=drug_sideeffect.T
    disease_protein=protein_disease.T
    protein_drug=drug_protein.T

    dr_dr=dgl.graph(sparse.csr_matrix(drug_drug),ntype='drug', etype='drdr')
    pr_pr=dgl.graph(sparse.csr_matrix(protein_protein),ntype='protein', etype='prpr')

    #di_di=dgl.graph(sparse.csr_matrix(disease_disease),ntype='disease', etype='didi')

    dr_di=dgl.bipartite(sparse.csr_matrix(drug_disease),'drug', 'drdi', 'diease')
    di_dr=dgl.bipartite(sparse.csr_matrix(disease_drug),'disease', 'didr', 'drug')
    #dr_si=dgl.bipartite(sparse.csr_matrix(drug_sideeffect),'drug', 'drsi', 'sideeffect')
    #si_dr=dgl.bipartite(sparse.csr_matrix(sideeffect_drug),'sideeffect', 'sidr', 'drug')
    dr_pr=dgl.bipartite(sparse.csr_matrix(drug_protein),'drug', 'drpr', 'protein')
    pr_dr=dgl.bipartite(sparse.csr_matrix(protein_drug),'protein', 'prdr', 'drug')
    pr_di=dgl.bipartite(sparse.csr_matrix(protein_disease),'protein', 'prdi', 'disease')
    di_pr=dgl.bipartite(sparse.csr_matrix(disease_protein),'disease', 'dipr', 'protein')

    graph=dgl.hetero_from_relations([dr_dr,pr_pr,dr_di,di_dr,dr_pr,pr_dr,pr_di,di_pr])

    return graph

def parse_args():
    parser = argparse.ArgumentParser(description='MGNN')

    parser.add_argument("--epochs", type=int, default=2000,
                        help="number of training epochs")
    parser.add_argument("--dim-embedding", type=int, default=128,
                        help="dimension of embeddings")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--patience', type=int, default=50,
                        help="Early stopping patience")
    parser.add_argument('--layer', type=int, default=3,
                        help="layers L")
    return parser.parse_args()

def meta_path_matrixs(g,meta_paths):
    meta_path_matrixs=[]
    for meta_path in meta_paths:
        new_g=dgl.metapath_reachable_graph(g,meta_path)
        meta_path_matrixs.append(new_g.adjacency_matrix().to_dense().numpy().T)
    return meta_path_matrixs
def to_ones(array):
    #array=np.array(array)
    j = i=0
    while i < len(array):
        j=0
        while j<len(array[i]):
            if array[i][j]!=0:
                array[i][j]=1
            j+=1
        i+=1
    #print(array,type(array))
    return array
'''
c=np.array([[0 ,1],
            [1,1]])
d=np.array([[1,0],
            [1,1]])
g=np.array([[1],
            [0]])
e=np.array([[1,0],
            [0,1]])
f=np.array([[0,0],
            [0,1]])
m=np.array([[c,d,g],
            [e,f]])

r=meta_path_matrixs(m)
print(r)'''
'''
c=np.array([[0 ,12],
            [3,5]])
print(to_ones(c))'''

'''
all_meta_paths=[[('drdr','drdi','dipr'),('drpr','prdr')],
                [('dipr','prdr'),('didr','drdr')]]
g=load_data('my_dataset/')
new_g=dgl.metapath_reachable_graph(
    g,all_meta_paths[0][0])
print(new_g.adjacency_matrix().to_dense().numpy().shape)'''


'''
all_meta_paths=[[('drdr','drdi','dipr'),('drpr','prdr')],
                [('dipr','prdr'),('didr','drdr')]]
g=load_data('my_dataset/')

print(meta_path_matrixs(g,all_meta_paths[0])[0].shape)'''

'''def concat_tool(drug_feature,sideeffect_feature,mat_drug_se):
    arr=[]
    for i,drug in enumerate(drug_feature):
        print(f'正在拼接第{i}行药物')
        for j,sideeffect in enumerate(sideeffect_feature):
            prediction_vector=torch.cat((drug, sideeffect), dim=-1)
            arr.append((prediction_vector,mat_drug_se[i][j]))
    return arr'''
def row_normalize(t):
    t = t.float()
    row_sums = t.sum(1) + 1e-12
    output = t / row_sums[:, None]
    output[torch.isnan(output) | torch.isinf(output)] = 0.0
    return output

def concat_tool(drug_feature,sideeffect_feature):
    prediction_vectors_list=[]
    #prediction_vectors=torch.cat((drug_feature[0], sideeffect_feature[0]), dim=-1)

    for drug in drug_feature:
        #print(f'正在拼接第{i}行药物')
        for sideeffect in sideeffect_feature:
            prediction_vector=torch.cat((drug, sideeffect), dim=-1)
            prediction_vectors_list.append(prediction_vector)
    prediction_vectors=torch.stack(prediction_vectors_list,dim=0)
                #prediction_vectors=torch.stack([prediction_vectors,prediction_vector],dim=0)

            #prediction_vector=torch.cat((drug, sideeffect), dim=-1)


    return prediction_vectors
'''
drug_feature=torch.tensor([[4,5,6],
                        [7,8,9]])
sideefect_feature=torch.tensor([[1,2,3],
                                [4,5,6],
                                [7,8,9]])
mat_drug_se=np.array([[1,1,0],
                      [1,0,1]])
a=concat_tool(drug_feature,sideefect_feature,mat_drug_se)
#a=torch.tensor(a)
print(a)
a=torch.tensor([1,1,2])
b=torch.tensor([2,2,3])
'''
#print(torch.stack([a,b],dim=0))


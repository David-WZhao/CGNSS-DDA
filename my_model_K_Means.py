import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from K_JunzhiCluster.con_sim_clu_cf2 import cf

from my_utils import *
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

import time
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold

np.random.seed(5)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from Att_hin import *

# print(data_set,data_set.shape)

'''all_meta_paths=[[[drug_drug,drug_disease],[drug_protein,protein_drug]],
                [[protein_drug,drug_disease],[protein_protein,protein_drug]],
                [[sideeffect_drug,drug_drug],[sideeffect_drug,drug_disease]],
                [[disease_protein,protein_drug],[disease_drug,drug_drug]]]'''

# print(all_meta_paths[0][0][0].shape)


t1 = time.time()


class MGNN(nn.Module):
    def __init__(self, d, p, h, dim, layer):
        super(MGNN, self).__init__()

        self.layer = 1
        self.dim_embedding = dim

        self.fc_drug_drug = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_drug_disease = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_disease_drug = nn.Linear(self.dim_embedding, self.dim_embedding).float()

        self.fc_protein_protein = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_protein_disease = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_disease_protein = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_drug_protein = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_protein_drug = nn.Linear(self.dim_embedding, self.dim_embedding).float()

        self.fc_disease_disease = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_protein_sideeffect = nn.Linear(self.dim_embedding, self.dim_embedding).float()

        self.project_drug_feature = nn.Linear(d, self.dim_embedding)
        self.project_protein_feature = nn.Linear(p, self.dim_embedding)
        self.project_disease_feature = nn.Linear(h, self.dim_embedding)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.attention = Attention(in_size=dim)

    def forward(self, drug_feat, protein_feat, disease_feat, drug_agg_matrixs, protein_agg_matrixs, disease_agg_matrixs,
                drug_disease, mask):
        # print(type(disease_feat))
        drug_feat = self.project_drug_feature(drug_feat)
        protein_feat = self.project_protein_feature(protein_feat)
        disease_feat = self.project_disease_feature(disease_feat)
        # print(drug_agg_matrixs[3].shape)
        for i in range(self.layer):
            # print(f'layer:{i}')

            # print(disease_agg_matrixs)
            drug_features = []
            # print(torch.from_numpy(drug_agg_matrixs[0]).to(torch.float32).type())
            # print(F.relu(self.fc_disease_drug(disease_feat))[0].type())
            # print(type(drug_agg_matrixs[0]))

            # print(to_ones(drug_agg_matrixs[0]))

            drug_features.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[0])).float()).to(device),
                                          F.relu(self.fc_drug_drug(drug_feat))) + drug_feat)
            drug_features.append(torch.mm((row_normalize(drug_agg_matrixs[1]).float()).to(device),
                                          F.relu(self.fc_disease_drug(disease_feat))) + drug_feat)
            # drug_features.append(attLayer((drug_agg_matrixs[1]).float().to(device),F.relu(self.fc_disease_drug(disease_feat)),drug_feat) + drug_feat)
            drug_features.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[2])).float()).to(device),
                                          F.relu(self.fc_protein_drug(protein_feat))) + drug_feat)

           
            drug_features = torch.stack(drug_features, dim=1)
            # print(drug_feats.size())
            # print(drug_feats[0][0])

            protein_features = []
            protein_features.append(
                torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[0])).float()).to(device),
                         F.relu(self.fc_protein_protein(protein_feat))) + protein_feat)
            protein_features.append(
                torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[1])).float()).to(device),
                         F.relu(self.fc_disease_protein(disease_feat))) + protein_feat)
            protein_features.append(
                torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[2])).float()).to(device),
                         F.relu(self.fc_drug_protein(drug_feat))) + protein_feat)
            

            protein_features = torch.stack(protein_features, dim=1)

            disease_features = []
            # disease_features.append(
            #     torch.mm((row_normalize(torch.from_numpy(disease_agg_matrixs[0])).float()).to(device),
            #              F.relu(self.fc_disease_disease(disease_feat))) + disease_feat)
            disease_features.append(torch.mm((row_normalize(disease_agg_matrixs[0]).float()).to(device),
                                             F.relu(self.fc_drug_disease(drug_feat))) + disease_feat)
            disease_features.append(
                torch.mm((row_normalize(torch.from_numpy(disease_agg_matrixs[1])).float()).to(device),
                         F.relu(self.fc_protein_disease(protein_feat))) + disease_feat)

            # disease_features.append(
            #     torch.mm((row_normalize(torch.from_numpy(disease_agg_matrixs[3])).float()).to(device),
            #              F.relu(self.fc_drug_disease(drug_feat))) + disease_feat)
            # disease_features.append(
            #     torch.mm((row_normalize(torch.from_numpy(disease_agg_matrixs[4])).float()).to(device),
            #              F.relu(self.fc_drug_disease(drug_feat))) + disease_feat)
            # disease_features.append(
            #     torch.mm((row_normalize(torch.from_numpy(disease_agg_matrixs[5])).float()).to(device),
            #              F.relu(self.fc_protein_disease(protein_feat))) + disease_feat)
            # disease_features.append(
            #     torch.mm((row_normalize(torch.from_numpy(disease_agg_matrixs[6])).float()).to(device),
            #              F.relu(self.fc_disease_disease(disease_feat))) + disease_feat)
            # disease_features.append(
            #     torch.mm((row_normalize(torch.from_numpy(disease_agg_matrixs[7])).float()).to(device),
            #              F.relu(self.fc_protein_disease(protein_feat))) + disease_feat)
            #
            # disease_features.append(
            #     torch.mm((row_normalize(torch.from_numpy(disease_agg_matrixs[8])).float()).to(device),
            #              F.relu(self.fc_protein_disease(protein_feat))) + disease_feat)

            disease_features = torch.stack(disease_features, dim=1)

            drug_feat, drug_alpha = self.attention(drug_features)
            protein_feat, protein_alpha = self.attention(protein_features)
            # sideeffect_feat,sideeffect_alpha=self.attention(sideeffect_features)
            disease_feat, disease_alpha = self.attention(disease_features)
            # np.savetxt('alpha/drug_alpha.txt',drug_alpha.detach().cpu().numpy().reshape(708,10),fmt='%1.4f')
            # np.savetxt('alpha/protein_alpha.txt',protein_alpha.detach().cpu().numpy().reshape(1512,9),fmt='%1.4f')
            # np.savetxt('alpha/sideeffect_alpha.txt',sideeffect_alpha.detach().cpu().numpy().reshape(4192,8),fmt='%1.4f')
            # np.savetxt('alpha/disease_alpha.txt',disease_alpha.detach().cpu().numpy().reshape(5603,9),fmt='%1.4f')
            '''drug_feat=torch.mean(drug_features,dim=1)
            protein_feat=torch.mean(protein_features,dim=1)
            sideeffect_feat=torch.mean(sideeffect_features,dim=1)
            disease_feat=torch.mean(disease_features,dim=1)'''
        predict = self.sigmoid(torch.mm(drug_feat, torch.transpose(disease_feat, 0, 1)))

        tmp = torch.mul(mask.float(), (predict - drug_disease.float()))

        loss = torch.sum(tmp ** 2)
        return predict, loss


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Attention, self).__init__()
        # print(in_size)
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)

        alpha = torch.softmax(w, dim=1)

        return (alpha * z).sum(1), alpha


# drug_drug,protein_protein,sideeffect_sideeffect,disease_disease,drug_disease_ori,drug_sideeffect,protein_disease,drug_protein=load_data('my_dataset/')

def train_and_evaluate(DDAtrain, DDAvalid, DDAtest, drug_feat, protein_feat, disease_feat, args,k, test):
    drug_disease = torch.zeros((708, 5603))
    mask = torch.zeros((708, 5603)).to(device)
    for ele in DDAtrain:
        drug_disease[ele[0], ele[1]] = ele[2]
        mask[ele[0], ele[1]] = 1

    # print(DSAvalid.shape)

    best_valid_aupr = 0.
    best_valid_auc = 0
    best_test_aupr = 0.
    best_test_auc = 0.
    patience = 0.

    g = ConstructGraph(drug_drug, protein_protein, drug_disease, protein_disease, drug_protein)

    # train_and_evaluate(DSAtrain, DSAvalid, DSAtest,DSAtrain_index,train_lab,0.001,0,5,device)
    all_meta_paths = [
        [('drdr', 'drdi'), ('drpr', 'prdr'), ('drdi', 'dipr'), ('drpr', 'prpr'), ('drdr', 'drpr', 'prdi')],
        [('prdr', 'drdi'), ('prpr', 'prdr'), ('prdi', 'didr'), ('prdr', 'drdr'), ('prpr', 'prdi'),
         ('prdr', 'drpr', 'prpr')],
        [('dipr', 'prdr'), ('didr', 'drdr'), ('dipr', 'prdi'), ('didr', 'drpr'),
         ('didr', 'drdi', 'dipr')]]

    drug_agg_matrixs = [drug_drug, drug_disease, drug_protein]
    protein_agg_matrixs = [protein_protein, protein_disease, drug_protein.T]
    # sideeffect_agg_matrixs=[sideeffect_sideeffect,drug_side.T]
    disease_agg_matrixs = [drug_disease.T, protein_disease.T]

    drug_meta_path_matrixs = meta_path_matrixs(g, all_meta_paths[0])
    protein_meta_path_matrixs = meta_path_matrixs(g, all_meta_paths[1])
    # sideeffect_meta_path_matrixs=meta_path_matrixs(g,all_meta_paths[2])
    disease_meta_path_matrixs = meta_path_matrixs(g, all_meta_paths[2])

    drug_agg_matrixs.extend(drug_meta_path_matrixs)
    protein_agg_matrixs.extend(protein_meta_path_matrixs)
    # sideeffect_agg_matrixs.extend(sideeffect_meta_path_matrixs)
    disease_agg_matrixs.extend(disease_meta_path_matrixs)
    # print('*************',drug_agg_matrixs[2].shape)

    drug_disease = drug_disease.to(device)

    model = MGNN(708, 1512, 5603, dim=args.dim_embedding, layer=args.layer)
    model = model.to(device)
    # loss_fcn=nn.MSELoss()
    # loss_fcn=loss_fcn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):

        t1 = time.time()
        model.train()
        predict, loss = model(drug_feat, protein_feat, disease_feat, drug_agg_matrixs, protein_agg_matrixs,
                              disease_agg_matrixs, drug_disease, mask)
        ##predict=torch.mul(mask.float(),predict)
        # print('predict:',predict)

        # loss=loss_fcn(predict,drug_sideeffect)
        results = predict.detach().cpu()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_list = []
            ground_truth = []

            for ele in DDAvalid:
                pred_list.append(results[ele[0], ele[1]])
                ground_truth.append(ele[2])

            # print('*******',type_of_target(results))
            valid_auc = roc_auc_score(ground_truth, pred_list)
            valid_aupr = average_precision_score(ground_truth, pred_list)
            # if valid_auc >= best_valid_auc:
            #   best_valid_auc=valid_auc
            if valid_aupr >= best_valid_aupr:
                best_valid_aupr = valid_aupr
                best_epoch = epoch
                patience = 0

                pred_list = []
                ground_truth = []
                for ele in DDAtest:
                    pred_list.append(results[ele[0], ele[1]])
                    ground_truth.append(ele[2])
                test_auc = roc_auc_score(ground_truth, pred_list)
                test_aupr = average_precision_score(ground_truth, pred_list)
                if test_auc >= best_test_auc:
                    best_test_auc = test_auc
                if test_aupr >= best_test_aupr:
                    best_test_aupr = test_aupr

            else:
                patience += 1
                if patience > args.patience:
                    print("Early Stopping")




                    pre_test_list = []
                    for ele in test:
                        pre_test_list.append(results[ele[0], ele[1]])
                    # np.savetxt("result3/001/seed1/pre_test_005_" + str(k) + ".txt", pre_test_list, fmt='%1.4f')
                    # #
                    #re_test = []
                    #test_len = len(pre_test_list)
                    #pre_test_list = np.array(pre_test_list)
                    #re_test.append(np.sum(pre_test_list > 0.5))
                    #re_test.append(np.sum(pre_test_list > 0.5) / test_len)
                    #re_test.append(np.sum(pre_test_list > 0.6))
                    #re_test.append(np.sum(pre_test_list > 0.6) / test_len)
                    #re_test.append(np.sum(pre_test_list > 0.7))
                    #re_test.append(np.sum(pre_test_list > 0.7) / test_len)
                    #re_test.append(np.sum(pre_test_list > 0.8))
                    #re_test.append(np.sum(pre_test_list > 0.8) / test_len)
                    #re_test.append(np.sum(pre_test_list > 0.9))
                    #re_test.append(np.sum(pre_test_list > 0.9) / test_len)
                    #np.savetxt("/home/ysw/work_meta/MGNNTest/K_JunzhiCluster/data2/zhunque_test_" + str(k) + ".txt", re_test, fmt='%1.4f')




                    # auc_aupr = []
                    # auc_aupr.append(test_auc)
                    # auc_aupr.append(test_aupr)
                    # np.savetxt("result3/001/seed1/auc_aupr_005_" + str(k) + ".txt", auc_aupr, fmt='%1.4f')



                    break
            print('Valid auc & aupr:', valid_auc, valid_aupr, ";  ", 'Test auc & aupr:', test_auc, test_aupr,
                  'best_epoch:', best_epoch + 1)

        t2 = time.time()
        print(f'epoc: {epoch + 1} loss:{loss.item()} time consum:{t2 - t1} s')

    return best_test_auc, best_test_aupr


def main(args):
    drug_feat = torch.eye(708, 708, dtype=torch.float)
    protein_feat = torch.eye(1512, 1512, dtype=torch.float)
    # sideeffect_feat = torch.eye(4192,4192,dtype=torch.float)
    disease_feat = torch.eye(5603, 5603, dtype=torch.float)

    drug_feat = drug_feat.to(device)
    protein_feat = protein_feat.to(device)
    # sideeffect_feat=sideeffect_feat.to(device)
    disease_feat = disease_feat.to(device)

    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(drug_disease_ori)[0]):
        for j in range(np.shape(drug_disease_ori)[1]):
            if int(drug_disease_ori[i][j]) == 1:
                whole_positive_index.append([i, j])






    len_data_index = len(whole_positive_index)







    data_set_pos = np.zeros((len(whole_positive_index), 3), dtype=int)
    count = 0
    # 将所有正样本按[i,j,lab]的形式存入data_set
    for i in whole_positive_index:
        data_set_pos[count][0] = i[0]
        data_set_pos[count][1] = i[1]
        data_set_pos[count][2] = 1
        count += 1



    fold = 1
    kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
    best_test_aucs = []
    best_test_auprs = []

    for train_index, test_index in kf.split(data_set_pos[:, :2], data_set_pos[:, 2]):
        # 此时训练集，测试集已经划分好

        train, DDAtest = data_set_pos[train_index], data_set_pos[test_index]

        train_pos = [[x[0], x[1]] for x in train]
        test_pos = [[x[0], x[1]] for x in DDAtest]
        #np.savetxt('/home/zy/ysw_file/MGNN/K_JunzhiCluster/test_pos.txt', test_pos, fmt='%d')

        # 获取矩阵的行数和列数
        A_train_row = 708
        A_train_col = 5603

        # 创建一个以0填充的二维矩阵
        A_train_pos = [[0] * A_train_col for _ in range(A_train_row)]

        # 遍历二元组列表，将对应位置的值设为1
        # 将cv的正样本恢复成矩阵
        for row, col in train_pos:
            A_train_pos[row][col] = 1


        # 选出有效负样本
        train_neg = cf(A_train_pos)

        #train_neg = np.loadtxt('/home/zy/ysw_file/MGNN/disease_data/A_cf30_120.txt')

        #  选出和正样本一样长度的负样本
        negative_sample_index = np.random.choice(np.arange(len(train_neg)), size=len(data_set_pos),replace=False)
        np.random.shuffle(negative_sample_index)

        # 选出训练集的负样本
        negative_sample_index2 = negative_sample_index[0:179292]
        # 选出测试集的负样本
        negative_sample_index_test = negative_sample_index[179292:199214]

        # 构造新的训练集
        train3 = np.zeros((len(negative_sample_index2) + len(train), 3), dtype=int)
        count2 = 0
        # 将所有正样本按[i,j,lab]的形式存入data_set
        for i in train_pos:
            train3[count2][0] = i[0]
            train3[count2][1] = i[1]
            train3[count2][2] = 1
            count2 += 1

        for i in negative_sample_index2:
            train3[count2][0] = train_neg[i][0]
            train3[count2][1] = train_neg[i][1]
            train3[count2][2] = 0
            count2+=1

        #train3 = np.concatenate((train, train_neg2))

        np.random.shuffle(train3)


        # 构造新的测试集
        test3 = np.zeros((len(negative_sample_index_test) + len(DDAtest), 3), dtype=int)
        test_pos = [[x[0], x[1]] for x in DDAtest]
        count3=0
        for i in test_pos:
            test3[count3][0] = i[0]
            test3[count3][1] = i[1]
            test3[count3][2] = 1
            count3 += 1

        for i in negative_sample_index_test:
            test3[count3][0] = train_neg[i][0]
            test3[count3][1] = train_neg[i][1]
            test3[count3][2] = 0
            count3+=1

        np.random.shuffle(test3)

        DDAtrain, DDAvalid = train_test_split(train3, test_size=0.05, random_state=None)

        print("#############%d fold" % fold + "#############")
        fold = fold + 1

        # 此处开始替换

        best_test_auc, best_test_aupr = train_and_evaluate(DDAtrain, DDAvalid, test3, drug_feat, protein_feat,
                                                           disease_feat, args,fold-1,DDAtest)

        best_test_aucs.append(best_test_auc)
        best_test_auprs.append(best_test_aupr)

    print('best_test_aucs:', best_test_aucs)
    print('best_test_aucprs:', best_test_auprs)

    average_test_auc = np.mean(best_test_aucs)
    average_test_aupr = np.mean(best_test_auprs)
    print('average_test_auc:', average_test_auc, 'average_test_aupr:', average_test_aupr)
    print('Program Finished')


if __name__ == "__main__":
    Patience = 20
    drug_drug, protein_protein, drug_disease_ori, protein_disease, drug_protein = load_data3(
        'my_dataset/')
    args = parse_args()
    print(args)
    main(args)



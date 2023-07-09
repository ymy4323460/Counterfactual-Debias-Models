import os
import math
import time
import random
import torch
import argparse
import numpy as np
import torch
import torch.nn as nn
import utils as ut
import torch.optim as optim
from torch import autograd
from torch.utils import data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.autograd import Variable
import dataloader as dataload
from codebase.ipw.config import get_config
from codebase.ipw.learner import Learner
import codebase.ipw.models as md
import sklearn.metrics as skm
import warnings
# import save_imputation_data

warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(torch.cuda.is_available())

args, _ = get_config()
workstation_path = './'
# args.model_dir = os.path.join(workstation_path, args.model_dir)
if args.dataset[:9] == 'logitdata' or args.dataset[:7] == 'invdata' or args.dataset[:3] == 'non':
    dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'train', 'data_nonuniform.csv')
    test_dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'data_uniform.csv')
else:
    dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'train', 'train.csv')
    test_dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'dev.csv')
# dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'train', 'train.csv')

imputation_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'imputation.csv')
# test_dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'dev.csv')
model = Learner(args)
if args.feature_data:
    data_feature_path = os.path.join(workstation_path, 'dataset', args.dataset)
else:
    data_feature_path = None
user_dic, item_dic = np.load(os.path.join(data_feature_path, 'user_features.npy'), allow_pickle=True).item(), np.load(
    os.path.join(data_feature_path, 'item_features.npy'), allow_pickle=True).item()
# dataset_dir, batch_size, shuffle=False, num_workers=8, pin_memory=True, propensity=None, imputation=None,data_feature_path=None
if args.debias_mode in ['Pretrain']:
    if args.pretrain_mode == "uniform_imputation":
        pretrain_dataloader = dataload.pretrain_dataload(
            os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'data_uniform.csv'), args.batch_size,
            data_feature_path=data_feature_path, pretrain_mode=args.pretrain_mode, syn=True)
    pretrain_dataloader = dataload.pretrain_dataload(dataset_dir, args.batch_size, data_feature_path=data_feature_path,
                                                     pretrain_mode=args.pretrain_mode, syn=True)
elif args.debias_mode in ['Propensity_Mode']:
    # If the data provide propensity score directly we choose this mode, currently only the dataset coat provide propensity score.
    propensity_dir = os.path.join(workstation_path, 'dataset', 'coat', 'propensity.ascii')
    train_dataloader = dataload.dataload(dataset_dir, args.batch_size, propensity=propensity_dir,
                                         data_feature_path=data_feature_path, shuffle=True)

elif args.debias_mode in ['DoublyRobust_Mode']:
    # If the data provide propensity score directly we choose this mode, currently only the dataset coat provide propensity score.
    propensity_dir = os.path.join(workstation_path, 'dataset', 'coat', 'propensity.ascii')
    train_dataloader = dataload.dataload(dataset_dir, args.batch_size, propensity=propensity_dir,
                                         data_feature_path=data_feature_path, shuffle=True)

elif args.debias_mode in ['Direct', 'Propensity_DR_Mode']:
    train_dataloader = dataload.dataload(dataset_dir, args.batch_size, data_feature_path=data_feature_path, shuffle=True)
elif args.debias_mode in ['Uniform_DR_Mode']:
    train_dataloader = dataload.dataload(dataset_dir, args.batch_size, data_feature_path=data_feature_path, shuffle=True)
# train_dataloader = dataload.dataload(dataset_dir, args.batch_size, imputation=os.path.join(workstation_path, 'dataset', 'yahoo', 'uniformimputation.csv'), data_feature_path=data_feature_path)
else:
    # pretrain_dataloader = dataload.dataload(dataset_dir, args.batch_size)
    train_dataloader = dataload.dataload(dataset_dir, args.batch_size, data_feature_path=data_feature_path, shuffle=True)
if args.debias_mode in ['Pretrain']:
    test_dataloader = dataload.pretrain_dataload(test_dataset_dir, 1000, data_feature_path=data_feature_path,
                                                 pretrain_mode=args.pretrain_mode, shuffle=False, syn=True)
else:
    test_dataloader = dataload.dataload(test_dataset_dir, 1000, data_feature_path=data_feature_path, shuffle=False)

print("=======================")
print(len(test_dataloader))
print("dev mode:{} debias mode:{} \n".format(args.train_mode, args.debias_mode))
if args.train_mode == 'pretrain':
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    for epoch in range(args.epoch_max):
        model.train()
        loss = 0
        total_loss = 0
        auc = 0

        total_auc = 0
        acc = 0

        total_acc = 0
        save_flag = True
        for x, a, y, r in pretrain_dataloader:
            optimizer.zero_grad()
            x, a, y, r = x.to(device), a.to(device), y.to(device), r.to(device)
            loss = model.pretrain(x, a, y, epoch, mode=args.pretrain_mode, save_flag=save_flag)

            save_flag = False
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            auc = skm.roc_auc_score(y.cpu().numpy(), torch.argmax(model.predict(x, a, pretrain_mode=args.pretrain_mode),
                                                                  dim=1).cpu().detach().numpy())
            acc = skm.accuracy_score(y.cpu().numpy(),
                                     torch.argmax(model.predict(x, a, pretrain_mode=args.pretrain_mode),
                                                  dim=1).cpu().detach().numpy())
            total_acc += acc
            total_auc += auc
            m = len(pretrain_dataloader)

        test_loss = 0
        test_total_loss = 0
        test_auc = 0

        test_total_auc = 0
        test_acc = 0

        test_total_acc = 0
        best_test_auc = 0
        total_test_logloss = 0
        for x, a, y, r in test_dataloader:
            x, a, y, r = x.to(device), a.to(device), y.to(device), r.to(device)
            test_loss = model.pretrain(x, a, y, epoch, mode=args.pretrain_mode)
            test_total_loss += test_loss.item()
            test_auc = skm.roc_auc_score(y.cpu().numpy(),
                                         torch.argmax(model.predict(x, a, pretrain_mode=args.pretrain_mode),
                                                      dim=1).cpu().detach().numpy())
            test_acc = skm.accuracy_score(y.cpu().numpy(),
                                          torch.argmax(model.predict(x, a, pretrain_mode=args.pretrain_mode),
                                                       dim=1).cpu().detach().numpy())
            test_logloss = skm.log_loss(y.cpu().numpy(), F.sigmoid(
                model.predict(x, a, pretrain_mode=args.pretrain_mode)[:, 1]).cpu().detach().numpy())
            test_total_acc += test_acc
            test_total_auc += test_auc
            total_test_logloss += test_logloss
            test_m = len(test_dataloader)
        best_test_auc = max(best_test_auc, best_test_auc)
        if test_total_auc >= best_test_auc:
            if args.pretrain_mode == 'propensity':
                ut.save_model_by_name(model_dir='Pretrain+'+args.dataset, model=model.propensity, global_step=0)
            elif args.pretrain_mode == 'imputation':
                ut.save_model_by_name(model_dir='Pretrain+'+args.dataset, model=model.imputation, global_step=0)
                ut.save_model_by_name(model_dir='Pretrain+'+args.dataset, model=model.imputation1, global_step=0)
            elif args.pretrain_mode == 'uniform_imputation':
                ut.save_model_by_name(model_dir='Pretrain+'+args.dataset, model=model.unbias_imputation, global_step=0)
        if epoch % 1 == 0:
            print("Epoch:{}\n test_auc:{}, test_acc:{}, test_logloss:{}".format(epoch, test_total_auc / test_m, \
                                                                                test_total_acc / test_m, \
                                                                                total_test_logloss / test_m))

elif args.train_mode == 'train':
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    for epoch in range(args.epoch_max):
        model.train()
        loss = 0
        total_loss = 0
        auc = 0

        total_auc = 0
        acc = 0

        total_acc = 0
        # for step in range(args.max_step):
        save_flag = True
        if args.debias_mode in ['Propensity_Mode']:
            for x, a, y, r, w in train_dataloader:

                optimizer.zero_grad()
                x, a, y, r, w = x.to(device), a.to(device), y.to(device), r.to(device), w.to(device)
                loss = model.learn(x, a, r, y, w=w, savestep=epoch, save_flag=save_flag)
                save_flag = False
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                m = len(train_dataloader)
        elif args.debias_mode in ['DoublyRobust_Mode']:
            for x, a, y, r, w in train_dataloader:
                optimizer.zero_grad()
                x_u, a_u, y_u = model.data_sampler_feature(x.size()[0], user_dic, item_dic)
                x, a, y, r, w, x_u, a_u, y_u, r_u = x.to(device), a.to(device), y.to(device), r.to(device), w.to(
                    device), x_u.to(device), a_u.to(device), y_u.to(device), r_u.to(device)
                loss = model.learn(x, a, r, y, x_u, a_u, y_u, w=w, savestep=epoch, save_flag=save_flag)
                save_flag = False
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                m = len(train_dataloader)
        elif args.debias_mode in ['Direct', 'Propensity_DR_Mode', 'Uniform_DR_Mode', 'Propensitylearnt_Mode',
                                  'SNIPSlearnt_Mode', 'CVIB', 'ATT']:
            for x, a, y, r, a_propensity in train_dataloader:
                print('the propensity is--------', a_propensity)
                optimizer.zero_grad()
                x_u, a_u, y_u = model.data_sampler_feature(x.size()[0], user_dic, item_dic)
                x, a, y, r, x_u, a_u, y_u, a_propensity = x.to(device), a.to(device), y.to(device), r.to(
                    device), x_u.to(device), a_u.to(device), y_u.to(device), a_propensity.to(device)
                loss = model.learn(x, a, r, y, x_u, a_u, y_u, a_propensity=a_propensity, savestep=epoch,
                                   save_flag=save_flag)
                save_flag = False
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                m = len(train_dataloader)

        else:
            for x, a, y, r, a_propensity in train_dataloader:
                optimizer.zero_grad()
                x, a, y, r, a_propensity = x.to(device), a.to(device), y.to(device), r.to(device), a_propensity.to(
                    device)
                loss = model.learn(x, a, r, y, a_propensity=a_propensity, w=None, savestep=epoch, save_flag=save_flag)
                save_flag = False
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                m = len(train_dataloader)
        if epoch % 1 == 0:
            total_test_auc = 0
            total_test_acc = 0
            total_test_logloss = 0
            total_test_ndcg = 0
            total_test_recall = 0
            total_test_precision = 0
            test_dataloader_len = len(test_dataloader)

            for test_x, test_a, test_y, test_r, negative_a in test_dataloader:
                if args.downstream == 'MLP':
                    test_y_pred = F.sigmoid(model.predict(test_x, test_a)).reshape(-1).cpu().detach().numpy()
                else:
                    test_y_pred = F.sigmoid(model.predict(test_x, test_a)).cpu().detach().numpy()
                test_auc = skm.roc_auc_score(test_y.cpu().numpy(), test_y_pred)
                test_acc = skm.accuracy_score(test_y.cpu().numpy(),
                                              np.where(test_y_pred > 0.5, np.ones_like(test_y_pred),
                                                          np.zeros_like(test_y_pred)))
                test_logloss = skm.log_loss(test_y.cpu().detach().numpy(), test_y_pred)

                top = np.repeat(np.sort(test_y_pred.reshape(-1, args.impression_len))[:, 3].reshape(-1, 1), args.impression_len, axis=1).reshape(-1)
                test_ndcg = skm.ndcg_score(test_y.reshape(-1, args.impression_len), test_y_pred.reshape(-1, args.impression_len), k=args.impression_len)

                test_pre = skm.precision_score(test_y.cpu().numpy(), np.where(test_y_pred > top, np.ones_like(test_y_pred),
                                                                      np.zeros_like(test_y_pred)))
                test_rec = skm.recall_score(test_y.cpu().numpy(), np.where(test_y_pred > top, np.ones_like(test_y_pred),
                                                                      np.zeros_like(test_y_pred)))

                total_test_auc += test_auc
                total_test_acc += test_acc
                total_test_logloss += test_logloss
                total_test_ndcg += test_ndcg
                total_test_recall += test_rec
                total_test_precision += test_pre

            print(
            "Epoch {}\n test_auc:{}, test_acc:{}, test_logloss:{}, test_ndcg:{}, test_recall:{}, test_precision:{}".format(epoch, float(
                total_test_auc / test_dataloader_len), float(total_test_acc / test_dataloader_len), float(
                total_test_logloss / test_dataloader_len), float(
                total_test_ndcg / test_dataloader_len), float(
                total_test_recall / test_dataloader_len), float(
                total_test_precision / test_dataloader_len))
                )


elif args.train_mode == 'test':
    # define optimizer for max
    model = Learner(args)
    model.load_model()
    loss = 0
    total_loss = 0
    auc = 0

    total_auc = 0
    acc = 0

    total_acc = 0
    for x, a, y, r in test_dataloader:
        x, a, y, r = x.to(device), a.to(device), y.to(device), r.to(device)
        loss = model.learn(x, a, r, y, w=None, savestep=0, lamb=1)
        total_loss += loss.item()
        auc = skm.roc_auc_score(y.cpu().numpy(), torch.argmax(model.predict(x, a), dim=1).cpu().detach().numpy())
        acc = skm.accuracy_score(y.cpu().numpy(), torch.argmax(model.predict(x, a), dim=1).cpu().detach().numpy())
        total_acc += acc
        total_auc += auc
        m = len(test_dataloader)

    print("total_loss:{} total_auc:{} total_acc:{}".format(total_loss / m, total_auc / m, total_acc / m))
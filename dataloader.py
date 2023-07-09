import os
import math
import random
import torch.nn as nn
from torch.utils import data
import argparse
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
from random import randint, sample

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def fileread(data_dir):
    print(data_dir)
    with open(data_dir, 'rb') as f:
        reader = f.readlines()
        data = []
        for line in reader:
            data.append(line.decode().strip().split(','))
    # print(len(data))
    data = np.array(data, dtype=float)
    return data  # .reshape([len(data),4,1])


def read_csv(data_path):
    print("reading csv : %s" % (data_path))
    use_cols = [line.strip() for line in open('config/used_header_info')]
    print('used_header_info 使用特征{}个'.format(len(use_cols)))

    # get dtype
    dtype = dict()
    for name in use_cols:
        if 'CATEGORY' in name:
            print(name)
            dtype[name] = 'category'
    if dtype:
        print('使用特征指定数据类型::', dtype)

    df = pd.read_csv(data_path,
                     header=0, usecols=use_cols, dtype=dtype, keep_default_na=False, nrows=500000)
    # print(df.dtypes)
    return df


def select_user_data(df):
    use_cols = [line.strip() for line in open('config/user_feature_info')]
    print('user_feature_info 使用特征{}个'.format(len(use_cols)))
    return df[use_cols].to_numpy(dtype='float32')


def select_item_data(df):
    use_cols = [line.strip() for line in open('config/product_feature_info')]
    print('product_feature_info 使用特征{}个'.format(len(use_cols)))
    # 待修改为item_id, torch.float对应np.float32, np默认np.float64,对应torch.double
    return df[use_cols].to_numpy(dtype='float32')


def fileread_propensity(data_dir):
    print(data_dir)
    with open(data_dir, 'rb') as f:
        reader = f.readlines()
        data = []
        for line in reader:
            data.append(line.decode().strip().split(' '))
    # print(len(data))
    data = np.array(data, dtype=float)
    return data  # .reshape([len(data),4,1])

class DataLoad(data.Dataset):
    def __init__(self, root, propensity=None, imputation=None, data_feature_path=None, a_propensity=False, Flag=False, mode='train', nagetive_path=None):
        # self.user_item = fileread(os.path.join(root,'data'))
        self.user_item = fileread(root)
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.propensity = propensity
        self.imputation = imputation
        self.data_feature_path = data_feature_path
        self.a_propensity = a_propensity
        self.mode = mode
        self.negative_sample = False
        if nagetive_path is not None:
            self.negative = np.load(nagetive_path)
            self.negative_sample = True
        '''
        if data_feature_path is not None:
            self.propensity_list = fileread_propensity(propensity)
        '''
        if imputation is not None:
            self.imputation_list = np.c_[self.user_item, fileread(imputation)]
            self.len_imputation_list = len(self.imputation_list)
        if data_feature_path is not None:
            self.userfeature = np.load(os.path.join(data_feature_path, 'user_features.npy'),
                                       allow_pickle=True).item()
            self.itemfeature = np.load(os.path.join(data_feature_path, 'item_features.npy'),
                                       allow_pickle=True).item()

    def __getitem__(self, idx):


        data = torch.from_numpy(self.user_item[idx])
        if self.negative_sample:
            self.negative_data = torch.from_numpy(self.negative[idx])

        if self.propensity is not None:
            propensity_data = torch.from_numpy(int(self.propensity_list[data[0]]), int(self.propensity_list[data[1]]))
            '''
            if self.imputation is not None:
                return imputation_data[0], imputation_data[1], imputation_data[2], imputation_data[3], imputation_data[4], imputation_data[5], imputation_data[6], imputation_data[7], propensity_data
            '''
            if self.data_feature_path is not None:
                return torch.cat((torch.from_numpy(self.userfeature[int(data[0].numpy())]),
                                      torch.from_numpy(self.itemfeature[int(data[1].numpy())])), 0), torch.from_numpy(
                    self.itemfeature[int(data[1].numpy())]), data[2], data[3], propensity_data
            if self.mode == 'train':
                return data[0], data[1], data[2], data[3], propensity_data
            else:
                return data[0], data[1], data[2], data[3], propensity_data, self.negative_data
        # print(data[0])
        else:
            '''
            if self.imputation is not None:
                return imputation_data[0], imputation_data[1], imputation_data[2], imputation_data[3], imputation_data[4], imputation_data[5], imputation_data[6], imputation_data[7]
            '''
            # print(data[0])
            if self.data_feature_path is not None:
                # print(data[0], data[1])
                return torch.cat((torch.from_numpy(self.userfeature[int(data[0].numpy())]),
                                      torch.from_numpy(self.itemfeature[int(data[1].numpy())])), 0), torch.from_numpy(
                    self.itemfeature[int(data[1].numpy())]), data[2], data[3], data[1]
            if self.mode == 'train' or not self.negative_sample:
                return data[0], data[1], data[2], data[3]
            else:
                return data[0], data[1], data[2], data[3], self.negative_data

    def __len__(self):
        if self.imputation is not None:
            # print(len(self.imputation_list), len(self.user_item))
            return len(self.imputation_list)
        return len(self.user_item)


def dataload(dataset_dir, batch_size, shuffle=False, num_workers=8, pin_memory=True, propensity=None, imputation=None,
             data_feature_path=None, a_propensity=False, Flag=False, mode='train', nagetive_path=None):
    dataset = DataLoad(dataset_dir, propensity, imputation, data_feature_path, a_propensity, Flag, mode=mode, nagetive_path=nagetive_path)
    dataset = Data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
    return dataset


class DataLoadPretrain(data.Dataset):
    def __init__(self, root, propensity=None, imputation=None, data_feature_path=None, pretrain_mode='propensity', syn=False):
        # self.user_item = fileread(os.path.join(root,'data'))
        self.user_item = fileread(root)
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.propensity = propensity
        self.imputation = imputation
        self.pretrain_mode = pretrain_mode
        self.data_feature_path = data_feature_path
        if propensity is not None:
            self.propensity_list = fileread_propensity(propensity)
        if imputation is not None:
            self.imputation_list = np.c_[self.user_item, fileread(imputation)]
            self.len_imputation_list = len(self.imputation_list)
        if data_feature_path is not None and not syn:
            self.userfeature = np.load(os.path.join(data_feature_path, 'user_features.npy'),
                                       allow_pickle=True)
            self.itemfeature = np.load(os.path.join(data_feature_path, 'item_features.npy'),
                                       allow_pickle=True)
        elif data_feature_path is not None and syn:
            self.userfeature = np.load(os.path.join(data_feature_path, 'user_features.npy'),
                                       allow_pickle=True).item()
            self.itemfeature = np.load(os.path.join(data_feature_path, 'item_features.npy'),
                                       allow_pickle=True).item()

    def __getitem__(self, idx):
        # print(idx)
        '''
        if self.imputation is not None:#%min(len(self.imputation_list), len(self.user_item))
            imputation_data = torch.from_numpy(self.imputation_list[idx])
            if self.data_feature_path is not None:
                imputation_data[0], imputation_data[1], imputation_data[4], imputation_data[5] = torch.from_numpy(self.userfeature[int(imputation_data[0].numpy())]), torch.from_numpy(self.userfeature[int(imputation_data[1].numpy())]), torch.from_numpy(self.userfeature[int(imputation_data[4].numpy())]), torch.from_numpy(self.userfeature[int(imputation_data[5].numpy())])
            #print(imputation_data)
        '''

        data = torch.from_numpy(self.user_item[idx])

        if self.propensity is not None:

            propensity_data = torch.from_numpy(int(self.propensity_list[data[0]]), int(self.propensity_list[data[1]]))
            '''
            if self.imputation is not None:
                return imputation_data[0], imputation_data[1], imputation_data[2], imputation_data[3], imputation_data[4], imputation_data[5], imputation_data[6], imputation_data[7], propensity_data
            '''
            if self.data_feature_path is not None:
                if self.pretrain_mode == 'propensity':
                    return torch.from_numpy(np.concatenate((self.userfeature[int(data[0].numpy())], self.itemfeature[int(data[1].numpy())]), axis=0)), data[1], data[2], data[
                        3], propensity_data
                return torch.from_numpy(np.concatenate((self.userfeature[int(data[0].numpy())], self.itemfeature[int(data[1].numpy())]), axis=0)), torch.from_numpy(
                    self.itemfeature[int(data[1].numpy())]), data[2], data[3], propensity_data
            return data[0], data[1], data[2], data[3], propensity_data
        # print(data[0])
        else:
            '''
            if self.imputation is not None:
                return imputation_data[0], imputation_data[1], imputation_data[2], imputation_data[3], imputation_data[4], imputation_data[5], imputation_data[6], imputation_data[7]
            '''
            # print(data[0])
            if self.data_feature_path is not None:
                # print(data[0], data[1])
                if self.pretrain_mode == 'propensity':
                    return torch.from_numpy(np.concatenate((self.userfeature[int(data[0].numpy())], self.itemfeature[int(data[1].numpy())]), axis=0)), data[1], data[2], data[3]
                return torch.from_numpy(np.concatenate((self.userfeature[int(data[0].numpy())], self.itemfeature[int(data[1].numpy())]), axis=0)), torch.from_numpy(
                    self.itemfeature[int(data[1].numpy())]), data[2], data[3]
            return data[0], data[1], data[2], data[3]

    def __len__(self):
        if self.imputation is not None:
            # print(len(self.imputation_list), len(self.user_item))
            return len(self.imputation_list)
        return len(self.user_item)


class DataLoad_Sample(data.Dataset):
    def __init__(self, root, propensity=None, imputation=None, data_feature_path=None, imputation_model=None):
        # self.user_item = fileread(os.path.join(root,'data'))
        self.user_item = fileread(root)
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.propensity = propensity
        self.imputation = imputation
        self.imputation_model = imputation_model
        self.data_feature_path = data_feature_path
        if propensity is not None:
            self.propensity_list = fileread_propensity(propensity)
        if imputation_model is not None:
            self.full_data = np.load(os.path.join('./', 'dataset', 'yahoo', 'full_data.txt.npy'))
#             print(self.full_data.shape)
        if data_feature_path is not None:
            self.userfeature = np.load(os.path.join(data_feature_path, 'user_features.npy'), allow_pickle=True)
            self.itemfeature = np.load(os.path.join(data_feature_path, 'item_features.npy'), allow_pickle=True)

    def __getitem__(self, idx):
        # print(idx)
        if self.imputation_model is not None:  # %min(len(self.imputation_list), len(self.user_item))
            x = np.random.randint(0, 15400, size=1, dtype='l')
            a = np.random.randint(0, 1000, size=1, dtype='l')
            imputation_data = torch.argmax(self.imputation_model.predict(torch.tensor([int(x)]).to(device),
                                                                         torch.tensor([int(a)]).to(
                                                                             device))).detach().numpy() if \
            self.full_data[x, a] == -1 else self.full_data[x, a]
            if self.data_feature_path is not None:
                imputation_data[0], imputation_data[1], imputation_data[4], imputation_data[5] = torch.from_numpy(
                    self.userfeature[int(imputation_data[0].numpy())]), torch.from_numpy(
                    self.userfeature[int(imputation_data[1].numpy())]), torch.from_numpy(
                    self.userfeature[int(imputation_data[4].numpy())]), torch.from_numpy(
                    self.userfeature[int(imputation_data[5].numpy())])
            # print(imputation_data)
        else:
            data = torch.from_numpy(self.user_item[idx])
            if self.data_feature_path is not None:
                data[0], data[1] = torch.from_numpy(self.userfeature[int(data[0].numpy())]), torch.from_numpy(
                    self.userfeature[int(data[1].numpy())])

        if self.propensity is not None:
            propensity_data = torch.from_numpy(int(self.propensity_list[data[0]]), int(self.propensity_list[data[1]]))
            if self.imputation is not None:
                return imputation_data[0], imputation_data[1], imputation_data[2], imputation_data[3], imputation_data[
                    4], imputation_data[5], imputation_data[6], imputation_data[7], propensity_data
            return data[0], data[1], data[2], data[3], propensity_data
        # print(data[0])
        else:
            if self.imputation is not None:
                return imputation_data[0], imputation_data[1], imputation_data[2], imputation_data[3], imputation_data[
                    4], imputation_data[5], imputation_data[6], imputation_data[7]
            return data[0], data[1], data[2], data[3]

    def __len__(self):
        return len(self.user_item)


def dataload_sample(dataset_dir, batch_size, shuffle=False, num_workers=8, pin_memory=True, propensity=None,
                    imputation=None, data_feature_path=None, imputation_model=None):
    dataset = DataLoad_Sample(dataset_dir, propensity, imputation, data_feature_path, imputation_model)
    dataset = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    return dataset


def pretrain_dataload(dataset_dir, batch_size, shuffle=False, num_workers=8, pin_memory=True, propensity=None,
                      imputation=None, data_feature_path=None, pretrain_mode='propensity', syn=False):
    dataset = DataLoadPretrain(dataset_dir, propensity, imputation, data_feature_path, pretrain_mode, syn=syn)
    dataset = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    return dataset


if __name__ == '__main__':
    dataload_huawei('./dataset/part_0_test')

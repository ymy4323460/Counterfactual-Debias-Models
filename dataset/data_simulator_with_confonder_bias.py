import numpy as np
import pandas as pd
from scipy.stats import binom
import torch
import os


def user_item_feature_withconfounder(context_dim=32, sample_size=128, item_size=100):
    confounder = np.random.uniform(-1, 1, (sample_size, context_dim))
    x = np.random.uniform(-1, 1, (sample_size, context_dim))
    a = np.random.uniform(-1, 1, (item_size, context_dim))
    # x = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, context_dim))
    # a = np.random.normal(loc=0.0, scale=1.0, size=(item_size, context_dim))
    return x, a, confounder



def get_impression_feature(x, parameters=None):
    # Todo: should make fixed name (nonliearlogitdata_) of dataset more flex
    item_feature_dict = np.load(
        './nonliearlogitdata_{}_{}_{}/item_features.npy'.format(parameters[1], parameters[2], parameters[3]),
        allow_pickle=True).item()
    # print(item_feature_dict)
    # item_feature_dict = np.load('./lineardata/item_feature_dict.npy', allow_pickle=True).item()
    return item_feature_dict[x]


def cal_similarity(x, y):
    # print(np.dot(x.reshape([1, x.shape[0]]), y.T).shape)
    return np.dot(x.reshape([1, x.shape[0]]), y.T)


def sigmoid_fun(x):
    return 1 / (1 + np.exp(-x))


def cal_similarity_uniform(x, y):
    # print(np.dot(x.reshape([1, x.shape[0]]), y.T).shape)
    return np.where(sigmoid_fun(np.dot(x.reshape([1, x.shape[0]]), y.T)) > 0.5, 1, 0)


def logit_policy(x, y, confounder, gamma=0.5):
    # print(x)
    #
    # print(y)
    x = x.reshape(1, x.shape[0])
    all_x = (1 - gamma) * np.concatenate((np.tile(x, (y.shape[0], 1)), y), axis=1) + gamma * np.tile(confounder,
                                                                                                     (y.shape[0], 2))
    indicate = np.where(all_x > 0, 1, 0)
    # print(indicate)
    # result = np.sum(np.multiply(all_x, indicate)-0.5*indicate, axis=1)
    result = np.sum(all_x, axis=1)
    # print(result)
    # print(sigmoid_fun(result))
    # exit()
    return sigmoid_fun(result)


def nonlinear_reward_function(x, y, confounder, gamma=0.5):
    x = x.reshape(1, x.shape[0])
    all_x = np.concatenate((np.tile(x, (y.shape[0], 1)), y), axis=1)  # + gamma*np.tile(confounder, (y.shape[0],2))
    indicate = np.where(all_x > 0, 1, 0)
    indicate_neg = np.where(all_x < 0, 1, 0)
    pos = np.multiply(all_x, indicate) - 0.5 * indicate
    neg = np.multiply(all_x, indicate_neg) + 0.5 * indicate
    # print(np.mean(np.matmul(pos.reshape(all_x.shape[0], all_x.shape[1], 1), neg.reshape(all_x.shape[0], 1, all_x.shape[1])), axis=1).shape)
    # print(np.mean(pos, axis=1).shape)
    result = np.sum(pos, axis=1) + (1 / 32.) * np.sum(
        np.matmul(pos.reshape(all_x.shape[0], all_x.shape[1], 1), neg.reshape(all_x.shape[0], 1, all_x.shape[1])),
        axis=(1, 2))
    # print(result)
    # exit()
    # result = gamma*confounder + (1-gamma)*result
    return np.where(sigmoid_fun(result) > 0.5, 1, 0)


def nonlinear_reward_function_logit(x, y, confounder, gamma=0.5):
    x = x.reshape(1, x.shape[0])
    # x = (1-gamma)*x.reshape(1, x.shape[0])+gamma*confounder.reshape(1, x.shape[0])
    # x = x.reshape(1, x.shape[0])
    all_x = np.concatenate((np.tile(x, (y.shape[0], 1)), y), axis=1)  # + gamma*np.tile(confounder, (y.shape[0],2))
    indicate = np.where(all_x > 0, 1, 0)
    indicate_neg = np.where(all_x < 0, 1, 0)
    pos = np.multiply(all_x, indicate) - 0.5 * indicate
    neg = np.multiply(all_x, indicate_neg) + 0.5 * indicate
    # print(np.mean(np.matmul(pos.reshape(all_x.shape[0], all_x.shape[1], 1), neg.reshape(all_x.shape[0], 1, all_x.shape[1])), axis=1).shape)
    # print(np.mean(pos, axis=1).shape)
    result = np.sum(pos, axis=1) + (1 / 32.) * np.sum(
        np.matmul(pos.reshape(all_x.shape[0], all_x.shape[1], 1), neg.reshape(all_x.shape[0], 1, all_x.shape[1])),
        axis=(1, 2))
    # print(result)
    # exit()
    return sigmoid_fun(result)


def linear_reward_function(x, y):
    # print(np.dot(x.reshape([1, x.shape[0]]), y.T).shape)
    return np.where(sigmoid_fun(np.dot(x.reshape([1, x.shape[0]]), y.T)) > 0.5, 1, 0)


def sample_from_multinomial(probability, impression_len=5, item_size=32):
    # print(probability)
    probability = probability + np.abs(np.random.normal(loc=0.0, scale=0.2, size=(probability.shape)))
    return np.random.choice(np.arange(item_size), size=impression_len, replace=False, p=probability / probability.sum())
    # return np.random.multinomial(impression_len, probability/probability.sum(), size=1)


def random_policy(impression_len=5, item_size=32):
    # print(probability)
    probability = 1. / impression_len * np.ones(item_size)
    return np.random.choice(np.arange(item_size), size=impression_len, replace=False, p=probability / probability.sum())
    # return np.random.multinomial(impression_len, probability/probability.sum(), size=1)


def get_feedbacks(x, y, threshold=None):
    if threshold is not None:
        # print(threshold)
        for i in range(threshold.shape[0]):
            if threshold[i] == 1:
                x[y[i]] = 1
    else:
        x[y] = 1
    return x


def logit_impression_list_new(context_dim=32, impression_len=5, sample_size=128, item_size=32, step=10, mode='dev',
                              sample_num=50, policy='logit', gamma=0.):
    sample_size = sample_num * 200
    x, a, confounder = user_item_feature_withconfounder(context_dim=context_dim, sample_size=sample_num * 200,
                                                        item_size=item_size)
    # if mode == 'train':
    #     x = np.load('./{}data_{}_{}_{}/user.npy'.format(policy, sample_num, context_dim, gamma), allow_pickle=True)
    #     # print(x)
    #     a = np.load('./{}data_{}_{}_{}/item.npy'.format(policy, sample_num, context_dim, gamma), allow_pickle=True)
    #     confounder = np.load('./{}data_{}_{}_{}/confounder.npy'.format(policy, sample_num, context_dim, gamma),
    #                          allow_pickle=True)
    user_feature_dict = dict(zip([j for j in range(sample_size)], list(x)))
    item_feature_dict = dict(zip([j for j in range(item_size)], list(a)))
    # print(x)
    if not os.path.exists('./nonliearlogitdata_{}_{}_{}/'.format(sample_num, context_dim, gamma)):  # 判断所在目录下是否有该文件名的文件夹
        os.makedirs('./nonliearlogitdata_{}_{}_{}/dev/'.format(sample_num, context_dim, gamma))
        os.makedirs('./nonliearlogitdata_{}_{}_{}/train/'.format(sample_num, context_dim, gamma))
    np.save('./nonliearlogitdata_{}_{}_{}/user'.format(sample_num, context_dim, gamma), x)
    np.save('./nonliearlogitdata_{}_{}_{}/item'.format(sample_num, context_dim, gamma), a)
    np.save('./nonliearlogitdata_{}_{}_{}/confounder'.format(sample_num, context_dim, gamma), confounder)
    np.save('./nonliearlogitdata_{}_{}_{}/user_features'.format(sample_num, context_dim, gamma), user_feature_dict)
    np.save('./nonliearlogitdata_{}_{}_{}/item_features'.format(sample_num, context_dim, gamma), item_feature_dict)
    all_ = None
    for i in range(1):
        # impression_p = 1/(1+np.exp(-np.dot(x, a.T) ))#np.random.normal(loc=0.0, scale=0.01, size=(sample_size, item_size))
        # impression_p = 1./16*np.ones((sample_size, item_size))#np.array(list(map(logit_policy, x, [a for j in range(sample_size)])))
        impression_p = np.array(list(
            map(logit_policy, x, [a for j in range(sample_size)], confounder, [gamma for j in range(sample_size)])))

        # print(impression_p.shape)
        # 获得impression list 的feature
        # print(impression_p[0].sum())
        impression_list = np.array(list(map(sample_from_multinomial, impression_p)))
        # print(impression_list)
        impression_information = np.array(list(map(get_impression_feature, impression_list.reshape(-1),
                                                   [['logit', sample_num, context_dim, gamma] for j in
                                                    range(sample_size * impression_len)]))).reshape(
            [sample_size, impression_len, context_dim])
        # print(impression_list.reshape(-1))
        # 用 true function 获得 result
        pair_matrix = np.array(list(map(nonlinear_reward_function_logit, x, impression_information, confounder,
                                        [gamma for j in range(sample_size)]))).reshape(sample_size, impression_len)
        # print(pair_matrix, pair_matrix.shape)
        value, index = torch.topk(torch.from_numpy(pair_matrix), 2, dim=1, largest=True, sorted=True, out=None)[
                           0].numpy(), \
                       torch.topk(torch.from_numpy(pair_matrix), 2, dim=1, largest=True, sorted=True, out=None)[
                           1].numpy()
        # print(value)
        # value = np.where(value > 0.5, 1, 0)# 重要
        # user_feedback = np.zeros([sample_size, impression_len]) #重要
        # assert index.shape[0] == user_feedback.shape[0]
        user_feedback = np.where(pair_matrix > 0.5, 1, 0)
        # user_feedback = np.array(list(map(get_feedbacks, user_feedback, index, value))) #重要
        # print(user_feedback)

        assert impression_list.shape == user_feedback.shape
        user_list = np.repeat(np.arange(sample_size).reshape([sample_size, 1]), impression_len, axis=1).reshape(-1, 1)
        # print(user_list)
        impression_list = impression_list.reshape(-1, 1)
        impression_indicate = np.ones([sample_size, impression_len]).reshape(-1, 1)
        user_feedback = user_feedback.reshape(-1, 1)
        assert user_list.shape == impression_list.shape
        # print(user_list.shape, impression_list.shape, impression_indicate.shape, user_feedback.shape)
        batch = np.concatenate((user_list, impression_list, user_feedback, impression_indicate), axis=1)
        if all_ is None:
            all_ = batch
        else:
            all_ = np.concatenate((all_, batch), axis=0)
        # print(batch)

    if not os.path.exists('./nonliearlogitdata_{}_{}_{}/'.format(sample_num, context_dim, gamma)):  # 判断所在目录下是否有该文件名的文件夹
        os.makedirs('./nonliearlogitdata_{}_{}_{}/dev/'.format(sample_num, context_dim, gamma))
        os.makedirs('./nonliearlogitdata_{}_{}_{}/train/'.format(sample_num, context_dim, gamma))
    partition = int(0.75 * all_.shape[0])
    pd.DataFrame(all_[: partition]).to_csv(
        './nonliearlogitdata_{}_{}_{}/dev/data_nonuniform.csv'.format(sample_num, context_dim, gamma), header=False,
        index=False)
    pd.DataFrame(all_[partition:]).to_csv(
        './nonliearlogitdata_{}_{}_{}/train/data_nonuniform.csv'.format(sample_num, context_dim, gamma), header=False,
        index=False)
    return all_, user_feature_dict, item_feature_dict


def random_impression_list(context_dim=32, impression_len=5, sample_size=128, item_size=32, step=10, mode='dev',
                           policy='logit', sample_num=50, gamma=0.5):
    sample_size = sample_num * 200
    x = np.load('./{}data_{}_{}_{}/user.npy'.format(policy, sample_num, context_dim, gamma), allow_pickle=True)
    # print(x)
    a = np.load('./{}data_{}_{}_{}/item.npy'.format(policy, sample_num, context_dim, gamma), allow_pickle=True)
    confounder = np.load('./{}data_{}_{}_{}/confounder.npy'.format(policy, sample_num, context_dim, gamma),
                         allow_pickle=True)
    all_ = None
    for i in range(1):
        # 随机采样出来一组item
        # impression_list = np.random.randint(0, high = item_size, size = (sample_size, impression_len), dtype = 'l')
        impression_list = np.array(list(map(random_policy, [impression_len for j in range(sample_size)])))
        # 获得impression list 的feature
        # print(impression_list)
        impression_information = np.array(list(map(get_impression_feature, impression_list.reshape(-1),
                                                   [[policy, sample_num, context_dim, gamma] for j in
                                                    range(sample_size * impression_len)]))).reshape(
            [sample_size, impression_len, context_dim])
        # print(x.shape)
        # 获得 result
        pair_matrix = np.array(list(map(nonlinear_reward_function, x, impression_information, confounder,
                                        [gamma for j in range(sample_size)]))).reshape(impression_list.shape)
        # print(pair_matrix.shape, impression_list.shape)
        # index = torch.topk(torch.from_numpy(pair_matrix), 1, dim=1, largest=True, sorted=True, out=None)[1].numpy()
        user_feedback = pair_matrix
        # assert index.shape[0] == user_feedback.shape[0]

        # user_feedback = np.array(list(map(get_feedbacks, user_feedback, index)))

        assert impression_list.shape == user_feedback.shape
        user_list = np.repeat(np.arange(sample_size).reshape([sample_size, 1]), impression_len, axis=1).reshape(-1, 1)
        impression_list = impression_list.reshape(-1, 1)
        impression_indicate = np.ones([sample_size, impression_len]).reshape(-1, 1)
        user_feedback = user_feedback.reshape(-1, 1)
        assert user_list.shape == impression_list.shape
        batch = np.concatenate((user_list, impression_list, user_feedback, impression_indicate), axis=1)
        if all_ is None:
            all_ = batch
        else:
            all_ = np.concatenate((all_, batch), axis=0)

    partition = int(0.75 * all_.shape[0])
    pd.DataFrame(all_[: partition]).to_csv(
        './{}data_{}_{}_{}/dev/data_uniform.csv'.format(policy, sample_num, context_dim, gamma), header=False,
        index=False)
    pd.DataFrame(all_[partition:]).to_csv(
        './{}data_{}_{}_{}/train/data_uniform.csv'.format(policy, sample_num, context_dim, gamma), header=False,
        index=False)

    return all_  # , user_feature_dict, item_feature_dict


for sam in [10]:
    for cdim in [32]:
        # The confonder influenced
        for gm in [0.5]:
            # bias 
            for bs in [0, 0.01, 0.1, 0.5, 1]
                logit_impression_list_new(mode='train', step=sam, policy='nonliearlogit', sample_num=sam,
                                          context_dim=cdim, gamma=gm)

            random_impression_list(mode='dev', step=sam, policy='nonliearlogit', sample_num=sam, context_dim=cdim,
                                   gamma=gm)
       
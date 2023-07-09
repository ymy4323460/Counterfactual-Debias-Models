import numpy as np
import torch
import torch.nn.functional as F
import utils as ut
from torch import autograd, nn, optim
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
import codebase.ipw.models as md
import os
from operator import itemgetter

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


class Learner(nn.Module):
    def __init__(self, args, debias_model=None, model=None):
        super().__init__()
        self.args = args
        self.name = self.args.name
        # self.full_data = np.load(os.path.join('./', 'dataset', 'yahoo', 'full_data.txt.npy'))
        self.propensity = md.Propensity(x_dim=self.args.user_dim, x_size=self.args.user_size,
                                        emb_dim=self.args.user_emb_dim, layer_dim=self.args.ipm_layer_dims,
                                        a_space=self.args.item_size).to(device)
        # if self.args.user_dim > 1:
        #    self.imputation = md.Imputation(name='Imputation', x_dim = self.args.user_size, a_dim= self.args.user_size, emb_dim1=self.args.user_dim, emb_dim2=self.args.user_dim, layer_dim=self.args.ipm_layer_dims, side_information=True).to(device)
        #    self.side_information=True
        self.imputation = md.Imputation(name='Imputation', x_dim=self.args.user_dim, a_dim=self.args.item_dim,
                                        x_size=self.args.user_size, a_size=self.args.item_size,
                                        emb_dim1=self.args.user_emb_dim, emb_dim2=self.args.item_emb_dim,
                                        layer_dim=self.args.ipm_layer_dims).to(device)
        self.imputation1 = md.Imputation(name='Imputation1', x_dim=self.args.user_dim, a_dim=self.args.item_dim,
                                        x_size=self.args.user_size, a_size=self.args.item_size,
                                        emb_dim1=self.args.user_emb_dim, emb_dim2=self.args.item_emb_dim,
                                        layer_dim=self.args.ipm_layer_dims).to(device)
        self.unbias_imputation = md.Imputation(name='Uniform_Imputation', x_dim=self.args.user_dim,
                                               a_dim=self.args.item_dim, x_size=self.args.user_size,
                                               a_size=self.args.item_size, emb_dim1=self.args.user_emb_dim,
                                               emb_dim2=self.args.item_emb_dim, layer_dim=self.args.ipm_layer_dims).to(
            device)
        self.acl_discriminito_o = md.Imputation(name='Discriminitor_Oracle', x_dim=self.args.user_dim, a_dim=self.args.item_dim,
                                        x_size=self.args.user_size, a_size=self.args.item_size,
                                        emb_dim1=self.args.user_emb_dim, emb_dim2=self.args.item_emb_dim,
                                        layer_dim=self.args.ipm_layer_dims, y_space=1).to(device)

        self.acl_discriminito_t = md.Imputation(name='Discriminitor_Target', x_dim=self.args.user_dim, a_dim=self.args.item_dim,
                                        x_size=self.args.user_size, a_size=self.args.item_size,
                                        emb_dim1=self.args.user_emb_dim, emb_dim2=self.args.item_emb_dim,
                                        layer_dim=self.args.ipm_layer_dims, y_space=1).to(device)
        print('experiment_id:{} dev mode:{} debias_mode: {} dataset: {} downstream: {}'.format(
            args.experiment_id, args.train_mode, args.debias_mode, args.dataset, self.args.downstream))
        if debias_model is not None:
            self.debias_model = debias_model
        if self.args.use_weight:
            if self.args.debias_mode in ['Propensity_Mode', 'SNIPS_Mode']:
                self.debias_model = md.DirectMethod(name=args.debias_mode, args=args).to(device)


            elif self.args.debias_mode in ['Propensitylearnt_Mode', 'SNIPSlearnt_Mode']:
                ut.load_model_by_name('Pretrain+'+self.args.dataset, self.propensity, 0)

            elif args.debias_mode == 'Direct':
                ut.load_model_by_name('Pretrain+'+self.args.dataset, self.imputation, 0)
                # self.full_data = np.load(os.path.join('./', 'dataset', 'yahoo', 'full_data.txt.npy'))

            elif args.debias_mode == 'DoublyRobust_Mode':  # propensity are given in dataset\
                ut.load_model_by_name('Pretrain+'+self.args.dataset, self.imputation, 0)
                # self.full_data = np.load(os.path.join('./', 'dataset', 'yahoo', 'full_data.txt.npy'))

            elif args.debias_mode == 'Propensity_DR_Mode':
                ut.load_model_by_name('Pretrain+'+self.args.dataset, self.propensity, 0)

                ut.load_model_by_name('Pretrain+'+self.args.dataset, self.imputation, 0)
                # self.full_data = np.load(os.path.join('./', 'dataset', 'yahoo', 'full_data.txt.npy'))

            elif args.debias_mode in ['Uniform_DR_Mode', 'CVIB']:
                ut.load_model_by_name('Pretrain+'+self.args.dataset, self.propensity, 0)
                ut.load_model_by_name('Pretrain+'+self.args.dataset, self.unbias_imputation, 0)
            elif args.debias_mode == ['ACL']:
                ut.load_model_by_name('Pretrain+'+self.args.dataset, self.imputation, 0)
                self.acl_discriminito_o = self.imputation

            elif args.debias_mode == ["ATT"]:
                ut.load_model_by_name('Pretrain+'+self.args.dataset, self.imputation, 0)
                ut.load_model_by_name('Pretrain+'+self.args.dataset, self.imputation1, 0)
            if self.args.downstream == 'MLP':
                self.debias_model = md.DirectMethod(name=args.debias_mode, args=args).to(device)
                # print('hahahahahahahahahahha', self.debias_model)
            elif self.args.downstream == 'LightGCN':
                self.debias_model = md.NeuBPR(name=args.debias_mode, args=args).to(device)
            else:

                self.debias_model = md.NeuBPR(name=args.debias_mode, args=args).to(device)
                # print('hahahahahahahahahahha', self.debias_model)
        else:
            self.debias_model = md.DirectMethod(name=args.debias_mode, args=args).to(device)

        # if args.debias_mode = 'dev':
        #     self.debias_model = debias_model

    def data_sampler(self, batch):
        x = torch.from_numpy(np.random.randint(0, self.args.user_size, size=batch))
        # print(x)
        a = torch.from_numpy(np.random.randint(0, self.args.item_size, size=batch))
        x = torch.tensor(x, dtype=torch.int64).to(device).reshape([x.size()[0], 1])
        a = torch.tensor(a, dtype=torch.int64).to(device).reshape([a.size()[0], 1])
        if self.args.debias_mode in ['Direct', 'DoublyRobust_Mode', 'Propensity_DR_Mode', 'CVIB', 'ACL']:
            sample = torch.argmax(self.imputation.predict(x, a).to(device), dim=1).detach()
        elif self.args.debias_mode == 'CVIB':
            data = np.random.shuffle(torch.cat((x, a), axis=1).numpy()).from_numpy()
            sample = torch.zeros_like(a)
            x = data[:, 0]
            a = data[:, 1]

        elif self.args.debias_mode == 'Uniform_DR_Mode':
            sample = torch.argmax(self.unbias_imputation.predict(x, a).to(device), dim=1).detach()

        elif self.args.debias_mode == 'ATT':
            epsilon = 0.01
            sample = torch.argmax(self.imputation.predict(x, a).to(device), dim=1).detach()
            sample1 = torch.argmax(self.imputation1.predict(x, a).to(device), dim=1).detach()
            x = x[torch.where(sample-sample1<epsilon)]
            a = a[torch.where(sample-sample1<epsilon)]
            sample = sample[torch.where(sample-sample1<epsilon)]
        # print(x.size(), a.size(), self.imputation.predict(x, a).size())
        '''
        for i in range(batch):
            x = np.random.randint(0, self.args.user_size, size=1, dtype='l')
            a = np.random.randint(0, self.args.item_size, size=1, dtype='l')
            if self.args.debias_mode in ['Direct', 'DoublyRobust_Mode', 'Propensity_DR_Mode']:
                #print(sample[i, 0],x)
                sample[i, 0] *= -x[0]
                sample[i, 1] *= -a[0]
                sample[i, 2] = torch.from_numpy(torch.argmax(self.imputation.predict(torch.tensor([int(x[0])]).to(device), torch.tensor([int(a[0])]).to(device))).cpu().detach().numpy()) if self.full_data[x[0],a[0]] == -1 else self.full_data[x[0],a[0]]
            elif self.args.debias_mode in ['Direct', 'DoublyRobust_Mode', 'Propensity_DR_Mode']:
                sample[i, 0] *= -x[0]
                sample[i, 1] *= -a[0]
                sample[i, 2] = torch.from_numpy(torch.argmax(self.unbias_imputation.predict(torch.tensor([int(x[0])]).to(device), torch.tensor([int(a[0])]).to(device))).cpu().detach().numpy()) if self.full_data[x[0],a[0]] == -1 else self.full_data[x[0],a[0]]
                '''
        return x, a, sample

    def data_sampler_feature(self, batch, user_dic, item_dic):
        # print(self.args.user_size, self.args.item_size)
        x_index = list(np.random.randint(0, self.args.user_size, size=batch))
        a_index = list(np.random.randint(0, self.args.item_size, size=batch))
        # need make sure keys in dict_key
        #     print(user_dic)
        x = torch.from_numpy(np.array(itemgetter(*x_index)(user_dic)))
        a = torch.from_numpy(np.array(itemgetter(*a_index)(item_dic)))
        x = torch.concat((x, a), axis=1)
        x = torch.tensor(x, dtype=torch.float32).to(device).reshape([x.size()[0], self.args.user_dim])
        a = torch.tensor(a, dtype=torch.float32).to(device).reshape([a.size()[0], self.args.item_dim])

        if self.args.debias_mode == 'Uniform_DR_Mode':
            sample = torch.argmax(self.unbias_imputation.predict(x, a).to(device), dim=1).detach()
        else:
            sample = torch.argmax(self.imputation.predict(x, a).to(device), dim=1).detach()
        # print(x.size(), a.size(), self.imputation.predict(x, a).size())
        '''
        for i in range(batch):
            x = np.random.randint(0, self.args.user_size, size=1, dtype='l')
            a = np.random.randint(0, self.args.item_size, size=1, dtype='l')
            if self.args.debias_mode in ['Direct', 'DoublyRobust_Mode', 'Propensity_DR_Mode']:
                #print(sample[i, 0],x)
                sample[i, 0] *= -x[0]
                sample[i, 1] *= -a[0]
                sample[i, 2] = torch.from_numpy(torch.argmax(self.imputation.predict(torch.tensor([int(x[0])]).to(device), torch.tensor([int(a[0])]).to(device))).cpu().detach().numpy()) if self.full_data[x[0],a[0]] == -1 else self.full_data[x[0],a[0]]
            elif self.args.debias_mode in ['Direct', 'DoublyRobust_Mode', 'Propensity_DR_Mode']:
                sample[i, 0] *= -x[0]
                sample[i, 1] *= -a[0]
                sample[i, 2] = torch.from_numpy(torch.argmax(self.unbias_imputation.predict(torch.tensor([int(x[0])]).to(device), torch.tensor([int(a[0])]).to(device))).cpu().detach().numpy()) if self.full_data[x[0],a[0]] == -1 else self.full_data[x[0],a[0]]
                '''
        return x, a, sample

    def load_model(self):
        ut.load_model_by_name(self.args.model_dir, self.debias_model, 0)

    def pretrain(self, x, a, y, savestep=0, mode='propensity', save_flag=True):
        if self.args.feature_data:
#             print(x.size())
            x = torch.tensor(x, dtype=torch.float32).to(device).reshape([x.size()[0], self.args.user_dim])
            if mode == 'propensity':
                a = torch.tensor(a, dtype=torch.int64).to(device).reshape([x.size()[0], 1])
            else:
                a = torch.tensor(a, dtype=torch.float32).to(device).reshape([x.size()[0], self.args.item_dim])
            y = torch.tensor(y, dtype=torch.int64).to(device)
        else:
            x = torch.tensor(x, dtype=torch.int64).to(device).reshape([x.size()[0], 1])
            a = torch.tensor(a, dtype=torch.int64).to(device).reshape([a.size()[0], 1])
            y = torch.tensor(y, dtype=torch.int64).to(device)

        if mode == 'propensity':
            # print(self.propensity.loss(x, a, y))
            return self.propensity.loss(x, a, y)
        if mode == 'imputation':
            # print('imputation')
#             print(self.imputation1.name)
            return self.imputation.loss(x, a, y) + self.imputation1.loss(x, a, y)

        if mode == 'uniform_imputation':
            return self.unbias_imputation.loss(x, a, y)

            '''

    def sample_data(self, x_size):
        if args.debias_mode in ['Direct', 'DoublyRobust_Mode', 'Propensity_DR_Mode']:
            self.full_data = np.load(os.path.join('./', 'dataset', 'yahoo', 'full_data.txt.npy'))
            '''

    def learn(self, x, a, r, y, x_u=None, a_u=None, y_u=None, w=None, a_propensity=None, savestep=0, lamb=0.1,
              save_flag=True, test=False, train_turn='max'):
        if self.args.feature_data:
            x = torch.tensor(x, dtype=torch.float32).to(device).reshape([x.size()[0], self.args.user_dim])
            a = torch.tensor(a, dtype=torch.float32).to(device).reshape([x.size()[0], self.args.item_dim])
            y = torch.tensor(y, dtype=torch.int64).to(device)
        else:
            x = torch.tensor(x, dtype=torch.int64).to(device).reshape([x.size()[0], 1])
            a = torch.tensor(a, dtype=torch.int64).to(device).reshape([a.size()[0], 1])
            y = torch.tensor(y, dtype=torch.int64).to(device)
        # print(x_u)
        if a_propensity is not None:
            a_propensity = torch.tensor(a_propensity, dtype=torch.int64).to(device).reshape([a.size()[0], 1])
        if x_u is not None:
            # print(x_u)
            if self.args.feature_data:
                x_u = torch.tensor(x_u, dtype=torch.float32).to(device).reshape([x_u.size()[0], self.args.user_dim])
                a_u = torch.tensor(a_u, dtype=torch.float32).to(device).reshape([x_u.size()[0], self.args.item_dim])
                y_u = torch.tensor(y_u, dtype=torch.int64).to(device)
            else:
                # print('xxxxxxxx')
                x_u = torch.tensor(x_u, dtype=torch.int64).to(device).reshape([x_u.size()[0], 1])
                a_u = torch.tensor(a_u, dtype=torch.int64).to(device).reshape([a_u.size()[0], 1])
                y_u = torch.tensor(y_u, dtype=torch.int64).to(device)
        elif test:
            # print('xxxxxxxx')
            if not self.args.feature_data:
                x_u = torch.tensor(x, dtype=torch.int64).to(device).reshape([y.size()[0], 1])
                a_u = torch.tensor(a, dtype=torch.int64).to(device).reshape([y.size()[0], 1])
                y_u = torch.tensor(y, dtype=torch.int64).to(device)
            else:
                x_u = torch.tensor(x, dtype=torch.float32).to(device).reshape([y.size()[0], self.args.user_dim])
                a_u = torch.tensor(a, dtype=torch.float32).to(device).reshape([y.size()[0], self.args.item_dim])
                y_u = torch.tensor(y, dtype=torch.int64).to(device)
            # print(y_u)
        if savestep % 15 == 0 and savestep > 0 and save_flag:
            # print("=================================")
            ut.save_model_by_name(model_dir=self.args.model_dir, model=self.debias_model, global_step=savestep)
        if w is not None:
            w = torch.tensor(1.0 / max(0.3, w), dtype=torch.float32).to(device)
        if self.args.debias_mode in ['Propensity_Mode', 'SNIPS_Mode']:
            if self.args.debias_mode == 'Propensity_Mode':
                return self.debias_model.weighted_loss(x, a, y, w)
            else:
                return self.debias_model.snips_loss(x, a, y, w)


        elif self.args.debias_mode in ['Propensitylearnt_Mode', 'SNIPSlearnt_Mode']:
            # print(a_propensity.max())
            # exit()
            if a_propensity is not None:
                w = torch.gather(self.propensity.propensity(x), 1, a_propensity)
            else:
                w = torch.gather(self.propensity.propensity(x), 1, a)
            # print(self.propensity.propensity(x).size(), a.size(), w.size())
            # print("=================================")
            if self.args.debias_mode == 'Propensitylearnt_Mode':
                return self.debias_model.weighted_loss(x, a, y, w)
            else:
                return self.debias_model.snips_loss(x, a, y, w)

        elif self.args.debias_mode == 'Direct':  # �Ȱ�train_data�岹��Ȼ����ѵ��
            # self.imputation = md.Imputation(x_dim=self.args.user_dim, layer_dim=self.args.ipm_layer_dims)
            # ut.load_model_by_name(self.args.model_dir, self.imputation, 0)
            # print(x_u)
            return self.debias_model.loss(x_u, a_u, y_u)

        elif self.args.debias_mode == 'DoublyRobust_Mode':  # propensity are given in dataset # �Ȱ�train_data�岹��ȫ����Ȼ����ѵ��
            propensity, imputation = self.debias_model.weighted_loss(x, a, y, w), self.debias_model.loss(x_u, a_u, y_u)
            return propensity + lamb * imputation

        elif self.args.debias_mode == 'Propensity_DR_Mode':
            if a_propensity is not None:
                w = torch.gather(self.propensity.propensity(x), 1, a_propensity)
            else:
                w = torch.gather(self.propensity.propensity(x), 1, a)
            propensity, imputation = self.debias_model.weighted_loss(x, a, y, w), self.debias_model.loss(x_u, a_u, y_u)
            return propensity + lamb * imputation

        elif self.args.debias_mode == 'Uniform_DR_Mode':
            if a_propensity is not None:
                w = torch.gather(self.propensity.propensity(x), 1, a_propensity)
            else:
                w = torch.gather(self.propensity.propensity(x), 1, a)
            propensity, imputation = self.debias_model.weighted_loss(x, a, y, w), self.debias_model.loss(x_u, a_u, y_u)
            return propensity + lamb * imputation

        elif self.args.debias_mode == 'CVIB':
            alpha = 5.0
            gamma = 1e-5
            pred = F.sigmoid(self.debias_model.predict(x, a))

            pred_ul = F.sigmoid(self.debias_model.predict(x_u, a_u))
            logp_hat = pred.log()
            pred_avg = pred.mean()
            pred_ul_avg = pred_ul.mean()
            info_loss = alpha * (- pred_avg * pred_ul_avg.log() - (1-pred_avg) * (1-pred_ul_avg).log()) + gamma* torch.mean(pred * logp_hat)
            loss = self.debias_model.loss(x, a, y)
            return loss + lamb*info_loss

        elif self.args.debias_mode == 'ACL':
            if train_turn=='max':
                observe_target = self.acl_discriminito_t.predict(x, a)
                observe_oracle = self.acl_discriminito_o.predict(x, a)
                discriminitor_loss = torch.norm(observe_target.reshape(-1) - observe_oracle.reshape(-1))
                w = F.sigmoid(observe_target)
                acl_loss = self.debias_model.weighted_loss(x, a, y, w)
                return -acl_loss + 0.01*discriminitor_loss
            else:
                w = F.sigmoid(self.acl_discriminito_t.predict(x, a))
                acl_loss = self.debias_model.weighted_loss(x, a, y, w)
                return acl_loss

#         elif self.args.debias_mode = 'ACL':
#             if train_turn='max':
#                 return
#             else:
#                 return self.debias_model.loss(x, a, y)

        elif self.args.debias_mode == "ATT":
            return self.debias_model.loss(x, a, y) + self.debias_model.loss(x_u, a_u, y_u)

    def predict(self, x, a, pretrain_mode=None):
        if self.args.feature_data:
            x = torch.tensor(x, dtype=torch.float32).to(device).reshape([x.size()[0], self.args.user_dim])
            if pretrain_mode == 'propensity':
                a = torch.tensor(a, dtype=torch.int64).to(device).reshape([a.size()[0], 1])
            else:  # pretrain_mode is None:
                a = torch.tensor(a, dtype=torch.float32).to(device).reshape([a.size()[0], self.args.item_dim])
        else:
            x = torch.tensor(x, dtype=torch.int64).to(device).reshape([x.size()[0], 1])
            a = torch.tensor(a, dtype=torch.int64).to(device).reshape([a.size()[0], 1])
        # print(x)

        if pretrain_mode == 'propensity':
            # print(self.propensity.predict(x.to(device)))
            return self.propensity.predict(x.to(device))
        if pretrain_mode == 'imputation':
            # print('imputation')
            return self.imputation.predict(x.to(device), a.to(device))
        if pretrain_mode == 'uniform_imputation':
            return self.unbias_imputation.predict(x.to(device), a.to(device))
        if pretrain_mode is None:
            y = self.debias_model.predict(x.to(device), a.to(device))
            return y
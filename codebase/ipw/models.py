import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import autograd, nn, optim
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

# propensity method
class Propensity(nn.Module):
    def __init__(self, x_dim, x_size, emb_dim, layer_dim, a_space):
        super().__init__()
        self.name = 'Propensity_Model'
        self.a_space = a_space
        self.x_dim = x_dim
        self.user_embedding = nn.Embedding(x_size, emb_dim).to(device)
        self.user_embedding_dim = emb_dim
        self.user_embedding_net = nn.Sequential(
                nn.Linear(self.x_dim, layer_dim[0]),
                nn.ReLU(),
                nn.Linear(layer_dim[0], self.user_embedding_dim)
            )
        self.predict_net = nn.Sequential(
            nn.Linear(emb_dim, layer_dim[2]),
            nn.ReLU(),
            nn.Linear(layer_dim[2], a_space)
        )
        self.sigmd = torch.nn.Sigmoid()
        self.sftcross = torch.nn.CrossEntropyLoss()


    def predict(self, x):
        if self.x_dim > 1:
            user_emb = self.user_embedding_net(x).reshape([x.size()[0], self.user_embedding_dim])
        else:
            #print("xxxxxxxxxxxx",x.size())
            user_emb = self.user_embedding(x).reshape([x.size()[0], self.user_embedding_dim])
        return self.predict_net(user_emb.reshape(-1, self.user_embedding_dim).to(device))

    def propensity(self, x):
        if self.x_dim > 1:
            user_emb = self.user_embedding_net(x).reshape([x.size()[0], self.user_embedding_dim])
        else:

            user_emb = self.user_embedding(x).reshape([x.size()[0], self.user_embedding_dim])
        return (F.softmax(self.predict_net(user_emb))*x.size()[0]).reshape(-1, self.a_space)

    def loss(self, x, a, y):
        #print(self.predict(x).size())
        return self.sftcross(self.predict(x), a.reshape(-1).to(device))

    def weighted_loss(self, x, a, y, w):
        return (w*self.sftcross(self.predict(x,a), y)).mean()

class Imputation(nn.Module):
    def __init__(self, name, x_dim, a_dim, x_size, a_size, emb_dim1, emb_dim2, layer_dim, y_space=2, propensity=None):# add hyperparameter indicate the side_information=False
        super().__init__()
        self.name = name
        #self.side_information = side_information
        self.user_embedding_dim =  emb_dim1
        self.item_embedding_dim =  emb_dim2
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.user_embedding = nn.Embedding(x_size, self.user_embedding_dim)
        print(emb_dim1)
        self.item_embedding = nn.Embedding(a_size, self.item_embedding_dim)
        self.predict_net = nn.Sequential(
            nn.Linear(self.user_embedding_dim + self.item_embedding_dim, layer_dim[2]),
            nn.ELU(),
            nn.Linear(layer_dim[2], y_space)
        )
        self.user_embedding_net = nn.Sequential(
                nn.Linear(self.x_dim, layer_dim[0]),
                nn.ReLU(),
                nn.Linear(layer_dim[0], self.user_embedding_dim)
            )
        self.item_embedding_net = nn.Sequential(
                nn.Linear(self.a_dim, layer_dim[0]),
                nn.ReLU(),
                nn.Linear(layer_dim[0], self.item_embedding_dim)
            )
        self.propensity = propensity
        self.sigmd = torch.nn.Sigmoid()
        self.sftcross = torch.nn.CrossEntropyLoss()

    def get_embeddings(self, x, a):
        user_emb = self.user_embedding(x)
        item_emb = self.item_embedding(a)
        return user_emb, item_emb

    def predict(self, x, a):
        #if not self.side_information:
            #
        if self.x_dim > 1:
            user_emb = self.user_embedding_net(x).reshape(-1, self.user_embedding_dim)
            item_emb = self.item_embedding_net(a).reshape(-1, self.item_embedding_dim)
        else:
            user_emb = self.user_embedding(x).reshape(-1, self.user_embedding_dim)
            item_emb = self.item_embedding(a).reshape(-1, self.item_embedding_dim)

        return self.predict_net(torch.cat((user_emb, item_emb),1))

    def loss(self, x, a, y):
        return self.sftcross(self.predict(x,a), y)

    def weighted_loss(self, x, a, y, w):
        w = torch.tensor(1.0/torch.clamp(w, 0.3, 1), dtype=torch.float32).to(device)
        return (w*self.sftcross(self.predict(x,a), y)).mean()

    def doubly_robust_loss(self, x, a, y, w):
        return self.weighted_loss(x, a, y, w), self.loss(x, a, y)

# direct method
class DirectMethod(nn.Module):
    def __init__(self, name, args):# add hyperparameter indicate the side_information=False
        super().__init__()
        self.name = 'Debias_Model'
        self.args = args
        self.x_emb_dim =  self.args.user_emb_dim
        self.a_emb_dim =  self.args.item_emb_dim
        self.x_dim = self.args.user_dim
        self.a_dim = self.args.item_dim
        self.user_embedding = nn.Embedding(self.args.user_size, self.args.user_emb_dim)
        #self.side_information = side_information
        self.item_embedding = nn.Embedding(self.args.item_size, self.args.item_emb_dim)

        self.layer_dim = self.args.ctr_layer_dims
        self.user_embedding_net = nn.Sequential(
            nn.Linear(self.x_dim, self.x_emb_dim),
            nn.ELU()
            )
        self.item_embedding_net = nn.Sequential(
                nn.Linear(self.a_dim, self.a_emb_dim),
                nn.ELU()
        )
        if self.x_dim == 1:
            self.predict_net = nn.Sequential(
                nn.Linear(self.x_emb_dim + self.a_emb_dim, self.args.ipm_layer_dims[1]),
                nn.ELU(),
                nn.Linear(self.args.ipm_layer_dims[1], self.args.ipm_layer_dims[2]),
                nn.ELU(),
                nn.Linear(self.args.ipm_layer_dims[2], 1)
            )
        else:
            self.predict_net = nn.Sequential(
                nn.Linear(self.x_dim + self.a_emb_dim, self.args.ipm_layer_dims[1]),
                nn.ELU(),
                nn.Linear(self.args.ipm_layer_dims[1], self.args.ipm_layer_dims[2]),
                nn.ELU(),
                nn.Linear(self.args.ipm_layer_dims[2], 1)
            )

        self.sigmd = torch.nn.Sigmoid()
        self.sftcross = torch.nn.CrossEntropyLoss(reduce=False)
        self.sftcross = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))


    def predict(self, x, a):
        if self.x_dim > 1:
            user_emb = x.reshape([x.size()[0], self.x_dim])
            item_emb = self.item_embedding_net(a).reshape(-1, self.a_emb_dim)
        else:
            user_emb = self.user_embedding(x).reshape(-1, self.x_emb_dim)
            item_emb = self.item_embedding(a).reshape(-1, self.a_emb_dim)

        return self.predict_net(torch.cat((user_emb, item_emb),1)).reshape(-1)

    def loss(self, x, a, y):
        y = torch.tensor(y, dtype=torch.float32).to(device)
        return self.sftcross(self.predict(x,a), y).mean()

    def weighted_loss(self, x, a, y, w):
        y = torch.tensor(y, dtype=torch.float32).to(device)
        w = torch.tensor(1.0/torch.clamp(w, 0.3, 1), dtype=torch.float32).to(device)
        return (w*self.sftcross(self.predict(x,a), y)).mean()

    def snips_loss(self, x, a, y, w):
        y = torch.tensor(y, dtype=torch.float32).to(device)
        w = torch.tensor(1.0/torch.clamp(w, 0.3, 1), dtype=torch.float32).to(device)
        return (w*self.sftcross(self.predict(x,a), y)).mean()/(w.mean())

    def doubly_robust_loss(self, x, a, y, w):
        y = torch.tensor(y, dtype=torch.float32).to(device)
        return self.weighted_loss(x, a, y, w), self.loss(x, a, y)


    def acl_loss(self, x, a, y, w):
        y = torch.tensor(y, dtype=torch.float32).to(device)


class NeuBPR(nn.Module):
    def __init__(self, name, args):
        super().__init__()
        self.name = 'Debias_Model'
        self.args = args
        # self.layers = [int(l) for l in args.layers.split('|')]
        # self.layers = args.layers
        if args.user_dim == 1:
            self.W_mlp = torch.nn.Embedding(num_embeddings=args.user_item_size[0], embedding_dim=args.user_emb_dim)
            self.W_mf = torch.nn.Embedding(num_embeddings=args.user_item_size[0], embedding_dim=args.user_emb_dim)
        else:
            self.W_mlp = torch.nn.Linear(self.args.user_dim, self.args.user_emb_dim)
            self.W_mf = torch.nn.Linear(self.args.user_dim, self.args.user_emb_dim)
        if args.item_dim == 1:
            self.H_mlp = torch.nn.Embedding(num_embeddings=args.user_item_size[1], embedding_dim=args.item_emb_dim)
            self.H_mf = torch.nn.Embedding(num_embeddings=args.user_item_size[1], embedding_dim=args.item_emb_dim)
        else:
            self.H_mlp = torch.nn.Linear(self.args.item_dim, self.args.item_emb_dim)
            self.H_mf = torch.nn.Linear(self.args.item_dim, self.args.item_emb_dim)

        nn.init.xavier_normal_(self.W_mlp.weight.data)
        nn.init.xavier_normal_(self.H_mlp.weight.data)
        nn.init.xavier_normal_(self.W_mf.weight.data)
        nn.init.xavier_normal_(self.H_mf.weight.data)

        if self.args.downstream == 'NeuBPR':
            self.fc_layers = torch.nn.ModuleList()
            for idx, (in_size, out_size) in enumerate(zip(self.args.ctr_layer_dims[:-1], self.args.ctr_layer_dims[1:])):
                self.fc_layers.append(torch.nn.Linear(in_size, out_size))

            self.affine_output = torch.nn.Linear(in_features=self.args.ctr_layer_dims[-1] + args.user_emb_dim, out_features=1)

        elif self.args.downstream == 'gmfBPR':
            self.affine_output = torch.nn.Linear(in_features=args.user_emb_dim, out_features=1)

        elif self.args.downstream == 'mlpBPR':
            self.fc_layers = torch.nn.ModuleList()
            for idx, (in_size, out_size) in enumerate(zip(self.args.ctr_layer_dims[:-1], self.args.ctr_layer_dims[1:])):
                self.fc_layers.append(torch.nn.Linear(in_size, out_size))

            self.affine_output = torch.nn.Linear(in_features=self.args.ctr_layer_dims[-1], out_features=1)

        self.logistic = torch.nn.Sigmoid()
        self.weight_decay = args.weight_decay
        self.dropout = torch.nn.Dropout(p=args.dropout)

        self.sftcross = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.args.ctr_classweight[1]))


    def loss(self, u, i, y, w=1):
        """Return loss value.

        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]

        Returns:s
            torch.FloatTensor
        """
        # print(u.size(), i.size())

        if self.args.user_dim == 1:
            u = torch.tensor(u, dtype=torch.int64).to(device).reshape(u.size()[0], 1)
        else:
            u = torch.tensor(u, dtype=torch.float32).to(device).reshape([u.size()[0], self.args.user_dim])
        if self.args.item_dim == 1:
            i = torch.tensor(i, dtype=torch.int64).to(device).reshape([i.size()[0], self.args.item_dim])
        else:
            i = torch.tensor(i, dtype=torch.float32).to(device).reshape([i.size()[0], self.args.item_dim])

        y = torch.tensor(y, dtype=torch.float32).to(device)
        x_ui = self.predict(u, i, mode='dev')
        # x_uj = self.predict(u, j, mode='dev')
        # x_uij = x_ui - x_uj
        # -------------------------------Mengyue Yang---------------------------------
        # # log_prob = F.logsigmoid(x_uij).mean()
        # log_prob = F.logsigmoid(x_uij)
        if not self.args.is_debias:
            Wu_mlp = self.W_mlp(u).reshape(u.size()[0], self.args.user_emb_dim)
            Wu_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)

        Hi_mlp = self.H_mlp(i).reshape(i.size()[0], self.args.item_emb_dim)
        Hi_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)


        # print('***'*10)
        # print('x_uij', x_uij.size(), Wu_mlp.size(), Hi_mlp.size())
        # print('***'*10)

        # log_prob = F.logsigmoid(x_uij).mean()

        # if self.args.model_name == 'NeuBPR':
        # 	regularization = self.weight_decay * (Wu_mlp.norm(dim=1).pow(2).mean() + \
        # 		Wu_mf.norm(dim=1).pow(2).mean() + Hi_mlp.norm(dim=1).pow(2).mean() + \
        # 		Hi_mf.norm(dim=1).pow(2).mean() + Hj_mlp.norm(dim=1).pow(2).mean() + \
        # 		Hj_mf.norm(dim=1).pow(2).mean())
        # elif self.args.model_name in ['gmfBPR', 'bprBPR']:
        # 	regularization = self.weight_decay * (Wu_mf.norm(dim=1).pow(2).mean() + \
        # 		Hi_mf.norm(dim=1).pow(2).mean() + Hj_mf.norm(dim=1).pow(2).mean())
        # elif self.args.model_name == 'mlpBPR':
        # 	regularization = self.weight_decay * (Wu_mlp.norm(dim=1).pow(2).mean() + \
        # 		Hi_mlp.norm(dim=1).pow(2).mean() + Hj_mlp.norm(dim=1).pow(2).mean())

        log_prob = self.sftcross(x_ui, y)

        # -----------------------------------------------------------------------
        if self.args.downstream == 'NeuBPR':
            if self.args.is_debias:
                regularization = self.weight_decay * (Hi_mlp.norm(dim=1) + Hi_mf.norm(dim=1))
            else:
                regularization = self.weight_decay * (Wu_mlp.norm(dim=1) + Wu_mf.norm(dim=1) + Hi_mlp.norm(dim=1) + Hi_mf.norm(dim=1))
        elif self.args.downstream in ['gmfBPR', 'bprBPR']:
            if self.args.is_debias:
                regularization = self.weight_decay * (Hi_mf.norm(dim=1))
            else:
                regularization = self.weight_decay * (Wu_mf.norm(dim=1) + Hi_mf.norm(dim=1))
        elif self.args.downstream == 'mlpBPR':
            if self.args.is_debias:
                regularization = self.weight_decay * (Hi_mlp.norm(dim=1))
            else:
                regularization = self.weight_decay * (Wu_mlp.norm(dim=1) + Hi_mlp.norm(dim=1))
        # ------------------------------------------------------------------------
        return w*(log_prob + regularization).mean()

    def weighted_loss(self, u, i, y, w):
        w = torch.tensor(1.0 / torch.clamp(w, 0.1, 1), dtype=torch.float32).to(device)
        return (w * self.loss(u, i, y, w)).mean()

    def snips_loss(self, u, i, y, w):
        w = torch.tensor(1.0 / torch.clamp(w, 0.1, 1), dtype=torch.float32).to(device)
        return (w * self.loss(u, i, y, w)).mean() / (w.mean())

    def doubly_robust_loss(self, u, i, y, w):
        return self.loss(u, i, y, w), self.loss(u, i, y)

    # ----------------------------Quanyu Dai----------------------------------
    def predict(self, u, i, mode='test'):
        #
        # if mode == 'test':
        #     u = torch.tensor(u, dtype=torch.int64).to(device).reshape(u.shape[0], 1)
        #     i = torch.tensor(i, dtype=torch.int64).to(device).reshape(i.shape[0], 1)

        if self.args.downstream == 'NeuBPR':
            if self.args.is_debias:
                user_embedding_mlp = u.reshape(u.size()[0], self.args.user_emb_dim)
                user_embedding_mf = u.reshape(u.size()[0], self.args.user_emb_dim)
            else:
                user_embedding_mlp = self.W_mlp(u).reshape(u.size()[0], self.args.user_emb_dim)
                user_embedding_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)
            item_embedding_mlp = self.H_mlp(i).reshape(i.size()[0], self.args.item_emb_dim)
            item_embedding_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)

            mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
            mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

            for idx, _ in enumerate(range(len(self.fc_layers))):
                mlp_vector = self.fc_layers[idx](mlp_vector)
                mlp_vector = torch.nn.ReLU()(mlp_vector)

            vector = torch.cat([mlp_vector, mf_vector], dim=-1)

        elif self.args.downstream == 'gmfBPR':
            if self.args.is_debias:
                user_embedding_mf = u.reshape(u.size()[0], self.args.user_emb_dim)
            else:
                user_embedding_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)
            item_embedding_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)

            vector = torch.mul(user_embedding_mf, item_embedding_mf)

        elif self.args.downstream == 'bprBPR':
            if self.args.is_debias:
                user_embedding_mf = u.reshape(u.size()[0], self.args.user_emb_dim)
            else:
                user_embedding_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)
            item_embedding_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)

            vector = torch.mul(user_embedding_mf, item_embedding_mf)

        elif self.args.downstream == 'mlpBPR':
            if self.args.is_debias:
                user_embedding_mlp = u.reshape(u.size()[0], self.args.user_emb_dim)
            else:
                user_embedding_mlp = self.W_mlp(u).reshape(u.size()[0], self.args.user_emb_dim)
            item_embedding_mlp = self.H_mlp(i).reshape(i.size()[0], self.args.item_emb_dim)

            vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector

            for idx, _ in enumerate(range(len(self.fc_layers))):
                vector = self.fc_layers[idx](vector)
                vector = torch.nn.ReLU()(vector)
                vector = self.dropout(vector)



        # print('###'*10)
        # print('user_emb, item_emb, vector', user_embedding_mf.size(), item_embedding_mf.size(), vector.size())
        # print('###'*10)

        if self.args.downstream in ['NeuBPR', 'gmfBPR', 'mlpBPR']:
            logits = self.affine_output(vector)
            rating = logits.reshape(logits.size()[0])
        elif self.args.downstream == 'bprBPR':
            rating = vector.sum(dim=1)
            rating = rating.reshape(rating.size()[0])

        if mode == 'test':
            # rating = self.logistic(rating)
            rating = rating#.detach().cpu().numpy()

        # print('rating', rating.shape, rating)

        return rating

    # ------------------------------------------------------------------------

    def load_pretrain_weights(self, gmf_model, mlp_model):
        """Loading weights from trained MLP model & GMF model for NeuBPR"""

        self.W_mlp.weight.data = mlp_model.W_mlp.weight.data
        self.H_mlp.weight.data = mlp_model.H_mlp.weight.data
        for idx in range(len(self.fc_layers)):
            self.fc_layers[idx].weight.data = mlp_model.fc_layers[idx].weight.data

        self.W_mf.weight.data = gmf_model.W_mf.weight.data
        self.H_mf.weight.data = gmf_model.H_mf.weight.data

        self.affine_output.weight.data = 0.5 * torch.cat(
            [mlp_model.affine_output.weight.data, gmf_model.affine_output.weight.data], dim=-1)
        self.affine_output.bias.data = 0.5 * (mlp_model.affine_output.bias.data + gmf_model.affine_output.bias.data)
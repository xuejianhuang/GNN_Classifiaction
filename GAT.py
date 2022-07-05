import torch.nn.functional as F
from torch import nn
import dgl.nn as dglnn

import config as config


class GraphAttnModel(nn.Module):

    def __init__(self, in_feats, hidden_dim, n_layers,fanouts, n_classes, num_heads, feat_dropout, attn_dropout,
                 dnn_unit_dropout,save_path=config.gat_model_path,loss_path=config.gat_loss_path,acc_path=config.gat_acc_path,
                 confusion_path=config.gat_confusion_path ,residual=True, activation=F.relu):
        '''
        :param in_feats: 输入特征维度
        :param hidden_dim: GAT隐含层维度
        :param n_layers:采用层数
        :param fanouts 每层最大采样邻居数量
        :param n_classes:分类类别数
        :param num_heads:注意力个数
        :param feat_dropout:特征dropout比例
        :param attn_dropout:attention dropout比例
        :param dnn_unit_dropout:DNN dropout比例
        :param save_path 模型保存路径
        :param loss_path 训练loss保存路径
        :param acc_path 训练acc保存路径
        :param confusion_path 混合矩阵保存路径
        :param residual:是否使用残差网络结构,defaut=True
        :param activation:激活函数 default=relu
        '''
        super(GraphAttnModel, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.fanouts=fanouts
        self.n_classes = n_classes
        self.heads = num_heads
        self.feat_dropout = feat_dropout
        self.attn_dropout = attn_dropout
        self.dnn_unit_dropout = dnn_unit_dropout
        self.save_path=save_path
        self.loss_path=loss_path
        self.acc_path=acc_path
        self.confusion_path=confusion_path
        self.residual = residual
        self.activation = activation


        self.linear_list = []
        for i in range(config.DNN_layers):
            if i == 0:
                self.linear_list.append(nn.Linear(self.hidden_dim * self.heads, config.DNN_hidden_dim[i]))
            else:
                self.linear_list.append(nn.Linear(config.DNN_hidden_dim[i - 1], config.DNN_hidden_dim[i]))
            self.linear_list.append(nn.LeakyReLU())
            self.linear_list.append(nn.Dropout(self.dnn_unit_dropout))

        self.DNN = nn.Sequential(*self.linear_list)

        self.layers = nn.ModuleList()

        # build multiple layers
        self.layers.append(dglnn.GATConv(in_feats=self.in_feats,
                                         out_feats=self.hidden_dim,
                                         num_heads=self.heads,
                                         feat_drop=self.feat_dropout,
                                         attn_drop=self.attn_dropout,
                                         residual=self.residual,
                                         activation=self.activation))

        for l in range(1, (self.n_layers)):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(dglnn.GATConv(in_feats=self.hidden_dim * self.heads,
                                             out_feats=self.hidden_dim,
                                             num_heads=self.heads,
                                             feat_drop=self.feat_dropout,
                                             attn_drop=self.attn_dropout,
                                             residual=self.residual,
                                             activation=self.activation))

    def forward(self, blocks, features):
        h = features
        for l in range(self.n_layers):
            h = self.layers[l](blocks[l], h).flatten(1)
        return self.DNN(h)







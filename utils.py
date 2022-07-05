#-*- coding:utf-8 -*-

"""
    Utilities to handel graph data
"""

import os
import dgl
import pickle
import numpy as np
import torch as th

import config
import pandas as pd
import random

def load_dgl_graph(base_path):
    """
    读取预处理的Graph，Feature和Label文件，并构建相应的数据供训练代码使用。

    :param base_path:
    :return:
    """
    graphs, _ = dgl.load_graphs(os.path.join(base_path, 'graph_pre.bin'))
    graph = graphs[0]
    print('################ Graph info: ###############')
    print(graph)

    with open(os.path.join(base_path, 'labels_pre.pkl'), 'rb') as f:
        label_data = pickle.load(f)

    labels = th.from_numpy(label_data['label'])
    train_labels_idx = label_data['train_labels_idx']
    test_labels_idx = label_data['test_labels_idx']
    print('################ Label info: ################')
    print('Total labels (including not labeled): {}'.format(labels.shape[0]))
    print('               Training label number: {}'.format(train_labels_idx.shape[0]))
    print('               Test label number: {}'.format(test_labels_idx.shape[0]))

    # get node features
    features = np.load(os.path.join(base_path, 'features_pre.npy'))
    node_feat = th.from_numpy(features).float()
    print('################ Feature info: ###############')
    print('Node\'s feature shape:{}'.format(node_feat.shape))

    return graph, labels, train_labels_idx, test_labels_idx, node_feat

def load_subtensor(node_feats, labels, seeds, input_nodes,):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = node_feats[input_nodes].to(config.device)
    batch_labels = labels[seeds].to(config.device)
    return batch_inputs, batch_labels

def saveLossAndAcc(test_loss_list,test_acc_list,train_loss_list,train_acc_list,loss_save_path,acc_save_path):
    loss=pd.DataFrame({"test_loss":test_loss_list,"train_loss":train_loss_list})
    acc=pd.DataFrame({"test_acc":test_acc_list,"train_acc":train_acc_list})
    loss.to_csv(loss_save_path,index=False)
    acc.to_csv(acc_save_path,index=False)

def saveConfusin(confusin,path):
    confusin_df=pd.DataFrame({config.label[i]:confusin[:,i] for i in range(config.n_classes)})
    confusin_df.to_csv(path,index=False)
def save_checkpoint(model,optimizer,epoch):
    checkpoint_new = {
        "net": model.state_dict (),
        "optimizer": optimizer.state_dict (),
        "epoch": epoch
    }
    th.save (checkpoint_new, config.checkpoint_path)
def time_diff(t_end, t_start):
    """
    计算时间差。t_end, t_start are datetime format, so use deltatime
    Parameters
    ----------
    t_end
    t_start

    Returns
    -------
    """
    diff_sec = t_end - t_start
    diff_min, rest_sec = divmod(diff_sec, 60)
    diff_hrs, rest_min = divmod(diff_min, 60)
    return (int(diff_hrs), int(rest_min), int(rest_sec))
def save_model(model,path):
    model_para_dict = model.state_dict()
    th.save(model_para_dict, path)


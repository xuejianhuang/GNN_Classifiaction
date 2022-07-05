
import time
import torch as th
import torch.nn as thnn
import torch.optim as optim

import dgl
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader

from GAT import GraphAttnModel
from GCN import GraphConvModel
from SAGE import GraphSageModel
from utils import load_dgl_graph, time_diff, load_subtensor, saveConfusin,save_model,save_checkpoint

from visdom import Visdom

import utils
import config
from visualize import Visualize



total_step = 0

def train(epoch, model, loss_fn, optimizer, train_dataloader, node_feat, labels,vis=None):
    model.train ()
    train_total_loss = 0  #
    train_total_correct = 0
    train_total_num = 0
    for step, (input_nodes, seeds, blocks) in enumerate (train_dataloader):  # input_nodes:参与计算的所有节点id,seeds:输出节点id
        global total_step  # 使用全局的total_step
        total_step += 1

        # forward
        batch_inputs, batch_labels = load_subtensor (node_feat, labels, seeds, input_nodes)
        blocks = [block.to (config.device) for block in blocks]
        # metric and loss

        train_batch_logits = model (blocks, batch_inputs)

        # print("train:",th.sum(train_batch_logits,axis=1).shape,th.sum(train_batch_logits,axis=1))

        train_loss = loss_fn (train_batch_logits, batch_labels)  # pytorch自动会把labels数字转化为one-hot
        # backward
        optimizer.zero_grad ()
        train_loss.backward ()
        optimizer.step ()

        train_total_loss += train_loss.cpu ().detach ().numpy ()*seeds.shape[0]  # 平均loss*bath_size

        preds = th.argmax (train_batch_logits, dim=1)
        train_total_correct += th.eq (preds, batch_labels).float ().sum ().item ()
        train_total_num += seeds.shape[0]
        # if step % 50 == 0:
        #     vis.append ([train_loss.item()], [total_step], name='train_loss', win='main_step_loss')

    mean_train_loss = train_total_loss / train_total_num
    mean__train_acc = train_total_correct / train_total_num

    print ('In epoch:{:03d}, train_loss:{:4f}, train_acc:{:.4f}'.format (epoch, mean_train_loss, mean__train_acc))

    # 动态可视化训练过程中loss和accuracy的变化
    # vis.append ([mean_train_loss], [epoch], name='train_loss', win='main_loss')
    # vis.append ([mean__train_acc], [epoch], name='train_acc', win='main_acc')
    return mean_train_loss, mean__train_acc


def test(epoch, model, loss_fn, test_dataloader, node_feat, labels,vis=None):
    # mini-batch for test
    model.eval ()
    confusin = th.zeros (config.n_classes, config.n_classes)  # 建立混合矩阵进行误差分析
    test_total_loss = 0
    test_total_correct = 0
    test_total_num = 0
    with th.no_grad ():
        for step, (input_nodes, seeds, blocks) in enumerate (test_dataloader):
            # forward
            batch_inputs, batch_labels = load_subtensor (node_feat, labels, seeds, input_nodes)
            blocks = [block.to (config.device) for block in blocks]
            # metric and loss
            test_batch_logits = model (blocks, batch_inputs)
            test_loss = loss_fn (test_batch_logits, batch_labels)

            test_total_loss += test_loss.cpu ().numpy ()*seeds.shape[0]  # 平均loss*bath_size
            preds = th.argmax (test_batch_logits, dim=1)

            for i in range (len (preds)):
                confusin[batch_labels[i], preds[i]] += 1

            test_total_correct += th.eq (preds, batch_labels).float ().sum ().item ()
            test_total_num += seeds.shape[0]

        mean_test_acc = test_total_correct / test_total_num
        mean_test_loss = test_total_loss / test_total_num
        print ('In epoch:{:03d}, test_loss:{:4f},test_acc:{:.4f}'.format (epoch, mean_test_loss, mean_test_acc))

        # 动态可视化训练过程中loss和accuracy的变化
        # vis.append ([mean_test_loss], [epoch], name='test_loss', win='main_loss')
        # vis.append ([mean_test_acc], [epoch], name='test_acc', win='main_acc')
        return mean_test_loss, mean_test_acc, confusin


def main(model):
   # vis = Visualize (server=config.server, host=config.port, wins=['main_loss', 'main_acc', 'main_step_loss'])
    vis=None
    graph, labels, train_nid, test_nid, node_feat = load_dgl_graph (config.base_path)
    graph = dgl.to_bidirected (graph, copy_ndata=True)  # 转化为双向图
    # graph = dgl.add_self_loop(graph)
    graph_data = (graph, labels, train_nid, test_nid, node_feat)

    sampler = MultiLayerNeighborSampler (model.fanouts)

    train_dataloader = NodeDataLoader (graph, train_nid, sampler, batch_size=config.batch_size, shuffle=True,
                                       num_workers=config.num_worker)

    test_dataloader = NodeDataLoader (graph, test_nid, sampler, batch_size=config.batch_size, shuffle=False,
                                      num_workers=config.num_worker)
    model.to (config.device)
    # 计算模型参数量
    count_parameters = sum (p.numel () for p in model.parameters () if p.requires_grad)
    print (f"The model has {count_parameters:,} trainable parameters")

    optimizer = optim.Adam (model.parameters (), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = thnn.CrossEntropyLoss ().to (config.device)
    best_acc = 0
    early_stop_cnt = 0
    test_loss_list=[]
    test_acc_list=[]
    train_loss_list=[]
    train_acc_list=[]
    start_epoch=0
    if config.resume:
        path_checkpoint =config.checkpoint_path  # 断点路径
        checkpoint = th.load (path_checkpoint)  # 加载断点
        model.load_state_dict (checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict (checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
    for epoch in range (start_epoch,config.epochs):
        t_start = time.time ()
        train_loss, train_acc = train (epoch, model, loss_fn, optimizer, train_dataloader, node_feat, labels,vis)
        if epoch % (config.checkpoint_epoch) == 0:
            save_checkpoint (model, optimizer, epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        t_end = time.time ()
        h, m, s = time_diff (t_end, t_start)
        print ("训练一个epoch需要：{}小时{}分{}秒".format (h, m, s))
        test_loss, test_acc, confusin = test (epoch, model, loss_fn, test_dataloader, node_feat, labels,vis)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            save_model (model, model.save_path)
            saveConfusin (confusin,model.confusion_path)

            print ("save model,acc:{}".format (best_acc))
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        if early_stop_cnt > config.early_stop_cnt:
            break
    utils.saveLossAndAcc(test_loss_list,test_acc_list,train_loss_list,train_acc_list,model.loss_path,model.acc_path)


if __name__ == '__main__':

    #-----------GAT---------------
    model_gat = GraphAttnModel (in_feats=config.in_feats, hidden_dim=config.GAT_hidden_dim, n_layers=config.GAT_layers,
                            fanouts=config.GAT_fanouts,n_classes=config.n_classes, num_heads=config.GAT_atte_num_heads,
                            feat_dropout=config.GAT_f_dropout,attn_dropout=config.GAT_atte_dropout,dnn_unit_dropout=config.DNN_dropout)

    # -----------GCN---------------
    model_gcn=GraphConvModel(in_feats=config.in_feats,hidden_dim=config.GCN_hidden_dim,n_layers=config.GCN_layers,fanouts=config.GCN_fanouts,
                             n_classes=config.n_classes,h_dropout=config.GCN_h_dropout,dnn_unit_dropout=config.DNN_dropout)

    # -----------SAGE---------------
    model_sage=GraphSageModel(in_feats=config.in_feats,hidden_dim=config.SAGE_hidden_dim,n_layers=config.SAGE_layers,fanouts=config.SAGE_fanouts,
                         n_classes=config.n_classes,feat_dropout=config.SAGE_f_dropout,dnn_unit_dropout=config.DNN_dropout,aggregator_type="pool")
    main (model_sage)






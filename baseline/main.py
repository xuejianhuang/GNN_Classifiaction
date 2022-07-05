import sys
sys.path.append("..")
import time
import  torch
from torch import optim,nn
import torch.nn as thnn
from torch.utils.data import DataLoader
from  config import server,port,in_feats,n_classes


from utils import save_model,time_diff,saveLossAndAcc

from visualize import Visualize

from baseline.DataSampler import DataSampler
from baseline.DNN import DNN
from baseline.lstm import LSTM
import baseline.baseline_config as config



total_step = 0
def train(epoch, model, loss_fn, optimizer, train_dataloader,vis):
    model.train()
    train_total_loss = 0  # 使用CrossEntropyLoss(reduction='sum')
    train_total_correct = 0
    train_total_num = 0
    for step, (x,label) in enumerate (train_dataloader):  # input_nodes:参与计算的所有节点id,seeds:输出节点id
        global total_step  # 使用全局的total_step
        total_step += 1
        x, label = x.to (config.device), label.to (config.device)
        out = model (x)
        train_loss = loss_fn(out, label)
        optimizer.zero_grad ()
        train_loss.backward ()
        optimizer.step ()

        train_total_loss += train_loss.cpu ().detach ().numpy ()

        preds = torch.argmax (out, dim=1)
        train_total_correct += torch.eq (preds, label).float ().sum ().item ()

        train_total_num += label.shape[0]

        if step % 50 == 0:
            vis.append ([train_loss.item () / label.shape[0]], [total_step], name='train_loss', win='base_step_loss')

    mean_train_loss = train_total_loss / train_total_num
    mean__train_acc = train_total_correct / train_total_num

    print ('In epoch:{:03d}, train_loss:{:4f}, train_acc:{:.4f}'.format (epoch, mean_train_loss, mean__train_acc))

    # 动态可视化训练过程中loss和accuracy的变化
    vis.append ([mean_train_loss], [epoch], name='train_loss', win='base_loss')
    vis.append ([mean__train_acc], [epoch], name='train_acc', win='base_acc')
    return mean_train_loss, mean__train_acc

def test(epoch, model, loss_fn, test_dataloader,vis):
        model.eval ()
        test_total_loss = 0
        test_total_correct = 0
        test_total_num = 0
        with torch.no_grad():
            for step, (x,label) in enumerate(test_dataloader):
                x, label = x.to(config.device), label.to(config.device)
                out = model(x)
                test_loss = loss_fn(out, label)
                test_total_loss += test_loss.cpu ().detach ().numpy ()

                preds = torch.argmax (out, dim=1)
                test_total_correct += torch.eq (preds, label).float ().sum ().item ()
                test_total_num += label.shape[0]

            mean_test_loss = test_total_loss / test_total_num
            mean_test_acc = test_total_correct / test_total_num
            print ('In epoch:{:03d}, test_loss:{:4f},test_acc:{:.4f}'.format (epoch, mean_test_loss, mean_test_acc))
            # 动态可视化训练过程中loss和accuracy的变化
            vis.append ([mean_test_loss], [epoch], name='test_loss', win='base_loss')
            vis.append ([mean_test_acc], [epoch], name='test_acc', win='base_acc')
            return mean_test_loss, mean_test_acc

def main(model):
    vis = Visualize (server=server, host=port, wins=['base_loss', 'base_acc', 'base_step_loss'])
    train_db = DataSampler (model='train')
    test_db = DataSampler (model='test')
    # test_db=DataSampler(model='test')

    train_dataloader = DataLoader(train_db, batch_size=config.batch_size, shuffle=True, num_workers=config.num_worker)
    test_dataloader = DataLoader(test_db, batch_size=config.batch_size, shuffle=False, num_workers=config.num_worker)
    model.to(config.device)
    # 计算模型参数量
    count_parameters = sum (p.numel () for p in model.parameters () if p.requires_grad)
    print (f"The model has {count_parameters:,} trainable parameters")

    optimizer = optim.Adam (model.parameters (), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = thnn.CrossEntropyLoss (reduction='sum').to (config.device)
    best_acc = 0
    early_stop_cnt = 0
    test_loss_list=[]
    test_acc_list=[]
    train_loss_list=[]
    train_acc_list=[]
    for epoch in range (config.epochs):
        t_start = time.time ()
        train_loss, train_acc = train (epoch, model, loss_fn, optimizer, train_dataloader,vis)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        t_end = time.time ()
        h, m, s = time_diff (t_end, t_start)
        print ("训练一个epoch需要：{}小时{}分{}秒".format (h, m, s))
        test_loss, test_acc = test (epoch, model, loss_fn, test_dataloader,vis)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            save_model(model, model.save_path)
            print ("save model,acc:{}".format (best_acc))
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        if early_stop_cnt > config.early_stop_cnt:
            break
    saveLossAndAcc(test_loss_list,test_acc_list,train_loss_list,train_acc_list,model.loss_path,model.acc_path)

if __name__ == '__main__':
    dnn=DNN(input_size=in_feats,output_size=n_classes,dnn_unit_dropout=config.DNN_dropout)
    lstm=LSTM(input_size=in_feats,output_size=n_classes,hidden_size=config.lstm_hidden_size,num_layers=config.lstm_layers,dropout=config.lstm_dropout)
    main(lstm)
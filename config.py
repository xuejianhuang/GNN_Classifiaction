import torch

#图数据存储文件
base_path="./data"
graph_file="graph_pre.bin"
label_file='labels_pre.pkl'
features_file="features_pre.npy"

#断点保存设置
resume=False  #是否从断点开始训练
checkpoint_path='./checkpoint/ckpt.pth'  #断点保存路径
checkpoint_epoch=5          #每隔训练多少epoch保存一次当前训练状态



#visdom配置
server='localhost'
port=8097



#GAT模型参数
GAT_layers=3                #采样层数数
GAT_fanouts=[10,10,10]       #每层最大采样邻居数量
GAT_hidden_dim=256      #隐含层维度
GAT_f_dropout=0.1      #特征随机置零概率
GAT_atte_dropout=0.1     #注意力机制随机置零概率
GAT_atte_num_heads=1      #多头注意力个数
gat_model_path="./results/models/gat_"+str(GAT_layers)+"_"+str(GAT_atte_num_heads)+".pkl"  #模型存储路径
gat_loss_path='./results/gat_'+str(GAT_layers)+"_"+str(GAT_atte_num_heads)+"_loss.csv"      #训练过程loss保存路径
gat_acc_path='./results/gat_'+str(GAT_layers)+"_"+str(GAT_atte_num_heads)+"_acc.csv"    #训练过程acc保存路径
gat_confusion_path='./results/gat_'+str(GAT_layers)+"_"+str(GAT_atte_num_heads)+'_confusin.csv'  #混合矩阵保存路径

#GCN模型参数
GCN_layers=3                #采样层数数
GCN_fanouts=[10,10,10]       #每层最大采样邻居数量
GCN_hidden_dim=256       #隐含层维度
GCN_h_dropout=0.1       #最后隐含层输出随机置零概率
gcn_model_path="./results/models/gcn_"+str(GCN_layers)+".pkl"  #模型存储位置
gcn_loss_path='./results/gcn_'+str(GCN_layers)+"_loss.csv"      #训练过程loss保存路径
gcn_acc_path='./results/gcn_'+str(GCN_layers)+"_acc.csv"    #训练过程acc保存路径
gcn_confusion_path='./results/gcn_'+str(GCN_layers)+'_confusin.csv'  #混合矩阵保存路径


#SAGE模型参数
SAGE_layers=3                #采样层数数
SAGE_fanouts=[10,10,10]       #每层最大采样邻居数量
SAGE_hidden_dim=256      #隐含层维度
SAGE_f_dropout=0.1        #特征随机置零概率
sage_model_path="./results/models/sage_"+str(SAGE_layers)+".pkl"  #模型存储位置
sage_loss_path='./results/sage_'+str(SAGE_layers)+"_loss.csv"      #训练过程loss保存路径
sage_acc_path='./results/sage_'+str(SAGE_layers)+"_acc.csv"    #训练过程acc保存路径
sage_confusion_path='./results/sage_'+str(SAGE_layers)+'_confusin.csv'  #混合矩阵保存路径




n_classes=23       #类别数
in_feats=300       #输入特征维度

label=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q', 'R', 'S', 'T', 'U', 'V', 'W']


#全连接层参数
DNN_layers=5         #全连接层数
DNN_hidden_dim =[512,256,128,64,n_classes]  #全连接层隐含层维度
DNN_dropout =0.1     #全连接层随机置零概率



#训练参数
learning_rate=0.001    #学习率
weight_decay=0     #L2正则化参数
epochs=100            #最大迭代次数
early_stop_cnt=20      #验证集准确率不提升时最多等待epoch，早停
batch_size=2048       #批量训练样本数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   #使用GPU还是CPU进行训练
num_worker=16

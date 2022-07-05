import torch

random_seed=22
#模型存储位置
dnn_model_path="base_dnn.pkl"
lstm_model_path="base_lstm.pkl"

#单文本分类模型数据文件（有标签的点）
feature_path = '../data/train_nodes_have_label_feature.npy'
label_path = '../data/train_nodes_have_label_l.npy'

#DNN模型参数
DNN_layers=5         #全连接层数
DNN_hidden_dim =[512,256,128,64,23]  #全连接层隐含层维度
DNN_dropout =0.1     #全连接层随机置零概率
DNN_model_path="./results/models/dnn_"+str(DNN_layers)+".pkl"  #模型存储位置
DNN_loss_path='./results/dnn_'+str(DNN_layers)+"_loss.csv"      #训练过程loss保存路径
DNN_acc_path='./results/dnn_'+str(DNN_layers)+"_acc.csv"    #训练过程acc保存路径

#LSTM模型参数
lstm_hidden_size=128
lstm_layers=2
lstm_dropout=0.1
bidirectional=False
lstm_model_path="./results/models/lstm_"+str(lstm_layers)+".pkl"  #模型存储位置
lstm_loss_path='./results/lstm_'+str(lstm_layers)+"_loss.csv"      #训练过程loss保存路径
lstm_acc_path='./results/lstm_'+str(lstm_layers)+"_acc.csv"    #训练过程acc保存路径





#训练参数
learning_rate=0.001    #学习率
weight_decay=0     #L2正则化参数
epochs=100            #最大迭代次数
early_stop_cnt=20      #验证集准确率不提升时最多等待epoch，早停
batch_size=2048       #批量训练样本数
device = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   #为了并行训练，把这baseline模型使用cpu
num_worker=16


import  baseline.baseline_config as config
from torch import nn


class DNN(nn.Module):
    def __init__(self,input_size,output_size,dnn_unit_dropout,save_path=config.DNN_model_path,
                 loss_path=config.DNN_loss_path, acc_path=config.DNN_acc_path):
        super(DNN, self).__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.dnn_unit_dropout=dnn_unit_dropout
        self.save_path=save_path
        self.loss_path = loss_path
        self.acc_path = acc_path
        self.linear_list = []
        for i in range (config.DNN_layers):
            if i == 0:
                self.linear_list.append (nn.Linear (self.input_size, config.DNN_hidden_dim[i]))
            else:
                self.linear_list.append (nn.Linear (config.DNN_hidden_dim[i - 1], config.DNN_hidden_dim[i]))
            self.linear_list.append (nn.LeakyReLU ())
            self.linear_list.append (nn.Dropout (self.dnn_unit_dropout))

        self.DNN = nn.Sequential (*self.linear_list)
    def forward(self,x):
        return self.DNN(x)
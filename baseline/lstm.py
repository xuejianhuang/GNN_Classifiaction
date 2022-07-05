
import baseline.baseline_config as config
from torch import nn
class LSTM(nn.Module):
    def __init__(self,input_size,output_size,hidden_size,num_layers,dropout,save_path=config.lstm_model_path,
                 loss_path=config.lstm_loss_path,acc_path=config.lstm_acc_path,bidirectional=False):
        super(LSTM, self).__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.dropout=dropout
        self.save_path=save_path
        self.loss_path = loss_path
        self.acc_path = acc_path
        self.bidirectional=bidirectional

       # self.pool1d=nn.MaxPool1d(kernel_size=2, stride=1, padding=0)  #300=>298
        self.norm=nn.BatchNorm1d(self.input_size)
        self.lstm= nn.LSTM(input_size=1,hidden_size=self.hidden_size,num_layers=self.num_layers,
                    dropout=self.dropout,bidirectional=self.bidirectional,batch_first=True)
        if self.bidirectional: 
            self.full_conn=nn.Linear(self.hidden_size*2, self.output_size)
        else:
            self.full_conn=nn.Linear(self.hidden_size, self.output_size)
           
    def forward(self,x):
        x=self.norm(x)
        x=x.unsqueeze(2) #4096*300->4096*300*1  N*L*H
        output, (h_n, c_n)=self.lstm(x)    #output=(N,L,D∗hidden_size)
        x=output[:,-1,:]
        x=x.squeeze(1)
        return  self.full_conn(x)                      
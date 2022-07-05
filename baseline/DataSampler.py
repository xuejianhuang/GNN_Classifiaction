import  torch
import numpy as np
from torch.utils.data import Dataset


class DataSampler(Dataset):
    data_train_feature_path = '../data/pre/train_nodes_have_label_feature.npy'  #使用训练集中所有有标签的节点
    data_train_label_path='../data/pre/train_nodes_have_label_l.npy'

    def __init__(self,model):
        
        self.model=model
        self.features=[]
        self.labels=[]
        ratio=0.05   #训练集95%，测试集5%

        features=np.load(DataSampler.data_train_feature_path)
        data_train_feaure=torch.from_numpy(features).float()
        label=np.load(DataSampler.data_train_label_path)
        data_train_label=torch.from_numpy(label)
        cut_idx = int(round(ratio* data_train_feaure.shape[0]))
        if self.model=='train':
            self.features=data_train_feaure[cut_idx:].tolist()
            self.labels=data_train_label[cut_idx:].tolist()

        else:
            self.features =data_train_feaure[:cut_idx].tolist()
            self.labels = data_train_label[:cut_idx].tolist()

    def __getitem__(self, idx):
        feature_, label_ = self.features[idx], self.labels[idx]
        return torch.tensor(feature_),torch.tensor(label_)

    def __len__(self):
        return  len(self.features)



if __name__ == '__main__':
    db = DataSampler('test')
    x, y = next(iter(db))
    print('sample:', x.shape, y.shape, y)







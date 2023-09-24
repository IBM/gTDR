from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

class GANF_Dataset(Dataset):
    def __init__(self, df, label=None, window_size=60, stride_size=10):
        super(GANF_Dataset, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.label = label
        if self.label is not None:
            self.data, self.idx, self.label = self.preprocess(df,self.label)
            self.label = 1.0-2*self.label 
        else:
            self.data, self.idx, self.time = self.preprocess(df)
    
    def preprocess(self, df):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)

        delat_time =  df.index[end_idx]-df.index[start_idx]
        
        print('Note you may need to create a class to override `preprocess()` if needed, \nthis is currently specific to SWaT or Traffic dataset')
        if self.label is not None:
            idx_mask = delat_time==pd.Timedelta(self.window_size,unit='s')
            return df.values, start_idx[idx_mask], self.label[start_idx[idx_mask]]
        else:
            idx_mask = delat_time==pd.Timedelta(5*self.window_size,unit='min')
            return df.values, start_idx[idx_mask], df.index[start_idx[idx_mask]]

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])
        if self.label is not None:
            return torch.FloatTensor(data).transpose(0,1),self.label[index]
        else:
            return torch.FloatTensor(data).transpose(0,1)
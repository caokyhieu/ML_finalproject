import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from operator import itemgetter
from torch.utils.data import Dataset, DataLoader

class RecSysData(Dataset):
  def __init__(self,train,item,mono_list,multi_list,item_mono_list,item_multi_list):

    self.df_user_train = pd.read_csv(train)
    self.df_item = pd.read_csv(item)
    self.mono_list = mono_list
    self.multi_list = multi_list
    self.item_mono_list = item_mono_list
    self.item_multi_list = item_multi_list
    self.process()
    self.label = self.df_user_train[self.df_user_train['action_type']=='clickout item']['reference']
    self.data = [(key,values.tolist()) for (key,values) in self.df_user_train.groupby(['session_id']).indices.items()]
    self.data = sorted(self.data,key=itemgetter(1))
    self.data_label = []
    for label_idx in self.label.index.values:
      for _,seq_idx in self.data:
        if seq_idx[0]<=label_idx<=seq_idx[-1]:
          temp = []
          for idx in seq_idx:
            if idx != label_idx:
              temp.append(idx)
            else:
              self.data_label.append((temp,label_idx))
              break
        else:
          continue


    
  def __len__(self):
    return len(self.data_label)

  def __getitem__(self,idx):
    return (df_user_train.loc[self.data_label[idx][0]],self.label.loc[self.data_label[idx][1]])

  def mono_emb(self,series):
    temp_set = set(series)
    dic = {j:i for (i,j) in enumerate(temp_set)}
    n_emb = int(np.power(len(dic),1/4))
    return (dic,n_emb)

  def multi_emb(self,series):
    temp_set = set(np.hstack(series.str.split('|').tolist()))
    dic = {j:i for (i,j) in enumerate(temp_set)}
    n_emb = int(np.power(len(dic),1/4))
    del temp_set
    return (dic,n_emb)

  def process(self):
    "for train  data"
    self.df_user_train[["current_filters","impressions","prices"]] = self.df_user_train[["current_filters","impressions","prices"]].fillna("NAN")
    self.embedding_mono = [self.mono_emb(self.df_user_train[i]) for i in self.mono_list]
    self.embedding_multi = [self.multi_emb(self.df_user_train[i]) for i in self.multi_list]
    replace_mono = {self.mono_list[i]:self.embedding_mono[i][0] for i in range(len(self.mono_list))}
    replace_multi = {self.multi_list[i]:self.embedding_multi[i][0] for i in range(len(self.multi_list))}
    self.df_user_train[self.mono_list] = self.df_user_train.replace(replace_mono)
    for idx,attr in enumerate(self.multi_list):
      self.df_user_train[attr] = self.df_user_train[attr].apply(lambda x: [self.embedding_multi[idx][0][i] for i in x.split('|')])

    "for label"
    self.item_embedding_mono = [self.mono_emb(df_item[i]) for i in self.item_mono_list]
    self.item_embedding_multi = [self.multi_emb(df_item[i]) for i in self.item_multi_list]
    replace_item_mono = {self.item_mono_list[i]:self.item_embedding_mono[i][0] for i in range(len(self.item_mono_list))}
    replace_item_multi = {self.item_multi_list[i]:self.item_embedding_multi[i][0] for i in range(len(self.item_multi_list))}
    self.df_item[self.item_mono_list] = self.df_item.replace(replace_item_mono)
    for idx,attr in enumerate(self.item_multi_list):
      self.df_item[attr] = self.df_item[attr].apply(lambda x: [self.item_embedding_multi[idx][0][i] for i in x.split('|')])

  #
  # def my_collate(self,batch):
  #   label = torch.stack([data[1] for data in batch],dim=0)
  #   idx = list(itertools.product(range(len(batch)),range(len(batch))))
  #   idx_1 = [i[0] for i in idx]
  #   idx_2 = [i[1] for i in idx]
  #   labels = label[idx_1] == label[idx_2]
  #   return((img[idx_1],img[idx_2]),labels.type(torch.float))

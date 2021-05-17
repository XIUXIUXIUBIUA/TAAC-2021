import os
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import re
import linecache
import dataloader.tokenization as tokenization

class MultimodaFeaturesDataset(Dataset):

    def __init__(self,dataset_config,job='training'):
        
        self.data_num_per_sample = 6 # 在train.txt中每个sample占6行
        self.device = dataset_config['device']
        if(job=='training'):
            self.meta_path = dataset_config['train_data_path']
        elif(job=='valdation'):
            self.meta_path = dataset_config['val_data_path']
        else:
            self.meta_path = dataset_config['test_data_path']
        self.tokenizer = tokenization.FullTokenizer(vocab_file=dataset_config['vocab_path'])
        self.label2id = {}
        with open(dataset_config['label_id_path'],'r') as f:
            for line in f:
                line = line.strip('\r\n')
                line = line.split('\t')
                self.label2id[line[0]] = int(line[1])
    def __getitem__(self, index):
        # 1. 从train.txt读取对应 idx 的path
        data_list = [] # 存储对于index的各个模态数据的路径和样本标签
        for line_i in range(self.data_num_per_sample*index+1,self.data_num_per_sample*(index+1)):
            line = linecache.getline(self.meta_path,line_i)
            line = line.strip('\r\n')
            data_list.append(line)
        video,audio,text_ids,label_ids = self.preprocess(data_list)
        video = video.to(self.device)
        audio = audio.to(self.device)
        text_ids = text_ids.to(self.device)
        label_ids = label_ids.to(self.device)
        return video,audio,text_ids,label_ids
    def __len__(self):
        # TODO 不能固定长度
        with open(self.meta_path,'r') as f:
            lines = f.readlines()
        return len(lines)//self.data_num_per_sample
    def preprocess(self,data_list):
        
        video_path,audio_path,image_path,text_path,label = data_list
        
        #--------------- video ----------------#
        video = torch.tensor(np.load(video_path).astype(np.float32))
        
        #--------------- audio ----------------#
        if os.path.exists(audio_path):
            audio = torch.tensor(np.load(audio_path).astype(np.float32))
        else:
            audio = torch.tensor(np.random.random((video.shape[0],128)).astype(np.float32))
            
        #--------------- text ----------------#
        
        text = ''
        with open(text_path,'r') as f:
            for line in f:
                dic = eval(line)
        for key in dic:
            dic[key] = ''.join(re.findall('[\u4e00-\u9fa5]',dic[key]))
            text += dic[key]
        tokens = ['[CLS]'] + self.tokenizer.tokenize(text)
        text_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        text_ids = torch.tensor(np.array(text_ids).astype('int64'))
        
        #--------------- label ----------------#
        label_ids = []
        label = label.split(',')
        np.random.shuffle(label)
        for i in label:
            label_ids.append(self.label2id[i])
        # label_ids = torch.tensor(np.array(label_ids).astype('int64'))
        dense_label_ids = torch.zeros(82)# ,dtype=torch.int64)
        dense_label_ids[label_ids] = 1
        # return video,audio,label_ids
        return video,audio,text_ids,dense_label_ids
    def collate_fn(self,batch):
        # 自定义dataloader 对一个batch的处理方式
        # 需要完成的任务有：
        # 1. 对video和audio的序列进行padding
        # 2. 对text，label_ids同样padding
        video_stacks = []
        audio_stacks = []
        text_stacks = []
        label_stacks = []
        for i in batch:
            video_stacks.append(i[0])
            audio_stacks.append(i[1])
            text_stacks.append(i[2])
            label_stacks.append(i[3])
        #print(video_stacks[0].size())
        video_stacks = pad_sequence(video_stacks,batch_first=True,padding_value=0)
        audio_stacks = pad_sequence(audio_stacks,batch_first=True,padding_value=0)
        text_stacks = pad_sequence(text_stacks,batch_first=True,padding_value=1) # bert 词表中 pad 是 1
        label_stacks = pad_sequence(label_stacks,batch_first=True,padding_value=0) # 83 表示没用的

        return video_stacks,audio_stacks,text_stacks,label_stacks
        # return video_stacks,audio_stacks,label_stacks
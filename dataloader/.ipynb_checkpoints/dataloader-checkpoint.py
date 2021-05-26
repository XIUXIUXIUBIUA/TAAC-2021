import os
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import re
import glob
import linecache
import dataloader.tokenization as tokenization
from transformers import BertTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import cv2
class MultimodaFeaturesDataset(Dataset):

    def __init__(self,dataset_config,job='training'):
        
        self.data_num_per_sample = 6 # 在train.txt中每个sample占6行
        self.text_max_len = dataset_config['text_max_len']
        self.device = dataset_config['device']
        
        if(job=='training'):
            self.meta_path = dataset_config['train_data_path']
        elif(job=='valdation'):
            self.meta_path = dataset_config['val_data_path']
        else:
            self.meta_path = dataset_config['test_data_path']
        self.tokenizer = BertTokenizer.from_pretrained(dataset_config['bert_path'])
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
        # return data_list[0]
        video,audio,text_ids,text_attention_mask,label_ids = self.preprocess(data_list)
        return video,audio,text_ids,text_attention_mask,label_ids
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
        
        # text = ''.join(re.findall('[\u4e00-\u9fa5]',dic['video_asr']))
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            truncation=True,
            max_length=self.text_max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        text_ids = inputs['input_ids']
        text_attention_mask = inputs['attention_mask']
        text_ids = torch.tensor(np.array(text_ids).astype('int64'))
        text_attention_mask = torch.tensor(np.array(text_attention_mask).astype('int64'))
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
        return video,audio,text_ids,text_attention_mask,dense_label_ids
    
    def collate_fn(self,batch):
        # 自定义dataloader 对一个batch的处理方式
        # 需要完成的任务有：
        # 1. 对video和audio的序列进行padding
        # 2. 对text，label_ids同样padding
        video_stacks = []
        audio_stacks = []
        text_stacks = []
        label_stacks = []
        text_attention_stacks = []
        for i in batch:
            video_stacks.append(i[0])
            audio_stacks.append(i[1])
            text_stacks.append(i[2])
            text_attention_stacks.append(i[3])
            label_stacks.append(i[4])
        
        video_stacks = pad_sequence(video_stacks,batch_first=True,padding_value=0)
        audio_stacks = pad_sequence(audio_stacks,batch_first=True,padding_value=0)
        text_stacks = pad_sequence(text_stacks,batch_first=True,padding_value=0) # 实际上没有pad
        # 实际上并没有padding，因为label变成multi-hot向量，长度都是82
        label_stacks = pad_sequence(label_stacks,batch_first=True,padding_value=0) 
        text_attention_stacks = pad_sequence(text_attention_stacks,batch_first=True,padding_value=0) # 实际上也没有pad
        return video_stacks,audio_stacks,text_stacks,text_attention_stacks,label_stacks
        # return video_stacks,audio_stacks,label_stacks
        
        

class TestingDataset(Dataset):

    def __init__(self,dataset_config):
        
        self.text_max_len = dataset_config['text_max_len']
        self.device = dataset_config['device']
        self.meta_path = dataset_config['test_data_path']
        self.feat_path = dataset_config['test_feat_path']
        self.tokenizer = BertTokenizer.from_pretrained(dataset_config['bert_path'])
        self.label2id = {}
        self.id2label = {}
        with open(dataset_config['label_id_path'],'r') as f:
            for line in f:
                line = line.strip('\r\n')
                line = line.split('\t')
                self.label2id[line[0]] = int(line[1])
                self.id2label[int(line[1])] = line[0]
        self.test_files = glob.glob(self.meta_path+'/*'+'.mp4')
        self.test_files.sort()
    def __getitem__(self, index):
        # 1. 从train.txt读取对应 idx 的path
        test_file =  self.test_files[index]
        feat_dict = self.preprocess(test_file,self.feat_path)
        return feat_dict
    def __len__(self):
        # TODO 不能固定长度
        return len(self.test_files)
    def preprocess(self,test_file,feat_path):
        feat_dict = {}
        file_name = os.path.basename(test_file)
        video_id = os.path.basename(test_file).split('.m')[0]
        #--------------- video/audio ----------------#
        
        feat_dict['video'] = torch.tensor(np.load(os.path.join(feat_path,'video_npy' ,'Youtube8M', 'tagging', video_id + '.npy')).astype(np.float32))
        # feat_dict['audio'] = torch.tensor(np.load(os.path.join(feat_path, 'audio_npy', 'Vggish', 'tagging', video_id + '.npy')).astype(np.float32))
        
        #--------------- text ----------------#
        text_path = os.path.join(feat_path, 'text_txt', 'tagging', video_id + '.txt')
        text = ''
        with open(text_path,'r') as f:
            for line in f:
                dic = eval(line)
        for key in dic:
            dic[key] = ''.join(re.findall('[\u4e00-\u9fa5]',dic[key]))
            text += dic[key]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            truncation=True,
            max_length=self.text_max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        text_ids = inputs['input_ids']
        text_attention_mask = inputs['attention_mask']
        feat_dict['text_ids'] = torch.tensor(np.array(text_ids).astype('int64'))
        feat_dict['text_attention_mask'] = torch.tensor(np.array(text_attention_mask).astype('int64'))
        feat_dict['file_name'] = file_name
        return feat_dict
    
    
class MultimodaRawDataset(Dataset):

    def __init__(self,dataset_config,job='training'):
        
        self.text_max_len = dataset_config['text_max_len']
        self.device = dataset_config['device']
        self.data_num_per_sample = 4
        if(job=='training'):
            self.meta_path = dataset_config['train_raw_data_path']
        elif(job=='valdation'):
            self.meta_path = dataset_config['val_raw_data_path']
        else:
            self.meta_path = dataset_config['test_data_path']
        self.tokenizer = BertTokenizer.from_pretrained(dataset_config['bert_path'])
        self.label2id = {}
        with open(dataset_config['label_id_path'],'r') as f:
            for line in f:
                line = line.strip('\r\n')
                line = line.split('\t')
                self.label2id[line[0]] = int(line[1])
        self.A_transform = A.Compose([
                # Resize(CFG.image_size, CFG.image_size),
                A.RandomResizedCrop(256, 256),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # mean on ImageNet
                    std=[0.229, 0.224, 0.225],  # std on ImageNet
                ),
                ToTensorV2(),
            ])
    def __getitem__(self, index):
        # 1. 从train.txt读取对应 idx 的path
        data_list = [] # 存储对于index的各个模态数据的路径和样本标签
        for line_i in range(self.data_num_per_sample*index+1,self.data_num_per_sample*(index+1)):
            line = linecache.getline(self.meta_path,line_i)
            line = line.strip('\r\n')
            data_list.append(line)
        #print(data_list)
        video,text_ids,text_attention_mask,label_ids = self.preprocess(data_list)
        return video,text_ids,text_attention_mask,label_ids
    def __len__(self):
        # TODO 不能固定长度
        with open(self.meta_path,'r') as f:
            lines = f.readlines()
        return len(lines)//self.data_num_per_sample
    def preprocess(self,data_list):
        
        video_path,text_path,label = data_list
        
        #--------------- video ----------------#
        
        frame_list = self.frame_iterator_list(video_path)
        
        video = []
        for frame in frame_list:
            augmented = self.A_transform(image=frame)
            video.append(augmented['image'].unsqueeze(0))
        video = torch.cat(video,dim=0)# shape(len,channel,H,W)
        #--------------- text ----------------#
        text = ''
        with open(text_path,'r') as f:
            for line in f:
                dic = eval(line)
           
        for key in dic:
            dic[key] = ''.join(re.findall('[\u4e00-\u9fa5]',dic[key]))
            text += dic[key]
        
        # text = ''.join(re.findall('[\u4e00-\u9fa5]',dic['video_asr']))
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            truncation=True,
            max_length=self.text_max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        text_ids = inputs['input_ids']
        text_attention_mask = inputs['attention_mask']
        text_ids = torch.tensor(np.array(text_ids).astype('int64'))
        text_attention_mask = torch.tensor(np.array(text_attention_mask).astype('int64'))
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
        return video,text_ids,text_attention_mask,dense_label_ids
    
    def collate_fn(self,batch):
        # 自定义dataloader 对一个batch的处理方式
        # 需要完成的任务有：
        # 1. 对video和audio的序列进行padding
        # 2. 对text，label_ids同样padding
        video_stacks = []
        text_stacks = []
        label_stacks = []
        text_attention_stacks = []
        for i in batch:
            video_stacks.append(i[0])
            text_stacks.append(i[1])
            text_attention_stacks.append(i[2])
            label_stacks.append(i[3])
        
        video_stacks = pad_sequence(video_stacks,batch_first=True,padding_value=0)
        text_stacks = pad_sequence(text_stacks,batch_first=True,padding_value=0) # 实际上没有pad
        # 实际上并没有padding，因为label变成multi-hot向量，长度都是82
        label_stacks = pad_sequence(label_stacks,batch_first=True,padding_value=0) 
        text_attention_stacks = pad_sequence(text_attention_stacks,batch_first=True,padding_value=0) # 实际上也没有pad
        return video_stacks,text_stacks,text_attention_stacks,label_stacks
    
    def frame_iterator_list(self,filename, every_ms=1000, max_num_frames=300):
        video_capture = cv2.VideoCapture()
        if not video_capture.open(filename):
            print(sys.stderr, 'Error: Cannot open video file ' + filename)
            return
        last_ts = -99999  # The timestamp of last retrieved frame.
        num_retrieved = 0

        frame_all = []
        while num_retrieved < max_num_frames:
            # Skip frames
            while video_capture.get(cv2.CAP_PROP_POS_MSEC) < every_ms + last_ts:
                if not video_capture.read()[0]:
                    return frame_all

            last_ts = video_capture.get(cv2.CAP_PROP_POS_MSEC)
            has_frames, frame = video_capture.read()
            if not has_frames:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_all.append(frame[:, :, ::-1])
            num_retrieved += 1

        return frame_all
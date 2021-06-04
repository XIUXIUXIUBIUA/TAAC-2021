# ensemble 模型在验证集上测试
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import yaml
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import utils.train_util as train_util
from dataloader.dataloader import TestingDataset
from src.loss.loss_compute import SimpleLossCompute
from src.model.baseline_model import Baseline
from src.loop.run_epoch import training_loop,validating_loop
from dataloader.dataloader import MultimodaFeaturesDataset,Datasetfortextcnn
from torch.utils.data import DataLoader
batch_size = 16
modal_name_list = ['video','audio','text']
config_path = './config/config.yaml'
config = yaml.load(open(config_path))
dataset = MultimodaFeaturesDataset(config['DatasetConfig'],job='valdation')
loader = DataLoader(dataset,num_workers=8,
                    batch_size=batch_size,
                    pin_memory=False,
                    collate_fn=dataset.collate_fn)

model_path_1 = "../checkpoint/0604/resnet50/epoch_22 0.7814.pt" # 已保存模型的路径
model_path_2 = '../checkpoint/0604/resnet50/epoch_34 0.7822.pt'
model_path_3 = '../checkpoint/0604/resnet50/epoch_20 0.7802.pt'
model_path_4 = '../checkpoint/0604/resnet50/epoch_30 0.7816.pt'
#model_path_3 = '../checkpoint/0604/resnet50_with_selfsupervised/epoch_20 0.7812.pt'
#model_path_4 = '../checkpoint/0604/lr/epoch_22 0.7797.pt'
#model_path_5 = '../checkpoint/0604/lr/epoch_42 0.7810.pt'

models_path = [model_path_1,model_path_2,model_path_3,model_path_4]
device = 'cuda'
top_k=20
# output_json = './0604_resnet_ensemble.json'
models = []
for path in models_path:
    model = Baseline(config['ModelConfig'])
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    models.append(model)

tagging_class_num = 82
evl_metrics = [train_util.EvaluationMetrics(tagging_class_num, top_k=20)
                           for i in range(len(modal_name_list)+1)] #+1 for fusion
for i in range(len(evl_metrics)):
    evl_metrics[i].clear()
metric_dict = {}
gap_dict = {}
for i,batch in enumerate(loader):
    if(len(batch)==5):
        video,audio,text,text_mask,label = batch
        video = video.to(device)
        audio = audio.to(device)
        text = text.to(device)
        text_mask = text_mask.to(device)
        label = label.to(device)
    else:
        video,audio,text,label = batch
        video = video.to(device)
        audio = audio.to(device)
        text = text.to(device)
        label = label.to(device)

    inputs_dict={}
    inputs_dict['video'] = video
    inputs_dict['audio'] = audio
    inputs_dict['text'] = text 
    if(len(batch)==5):
        inputs_dict['attention_mask'] = text_mask
    else:
        inputs_dict['attention_mask'] = None
        
    B = video.shape[0]
    pred_dict_ensemble = {}
    for modal_name in (modal_name_list+['fusion']):
        pred_dict_ensemble['tagging_output_'+modal_name] = {}
        pred_dict_ensemble['tagging_output_'+modal_name]['predictions'] = torch.zeros(B,82).cuda()
    
    for model in models:
        pred_dict = model(inputs_dict)
        for modal_name in (modal_name_list+['fusion']):
            pred_dict_ensemble['tagging_output_'+modal_name]['predictions'] += pred_dict['tagging_output_'+modal_name]['predictions']
            
    for modal_name in (modal_name_list+['fusion']):
        pred_dict_ensemble['tagging_output_'+modal_name]['predictions'] = pred_dict_ensemble['tagging_output_'+modal_name]['predictions']/len(models)
        
    for index,modal_name in enumerate(modal_name_list+['fusion']):
        pred = pred_dict_ensemble['tagging_output_'+modal_name]
        pred = pred['predictions'].detach().cpu().numpy()
        val_label = label.cpu().numpy()
        gap = train_util.calculate_gap(pred, val_label)
        evl_metrics[index].accumulate(pred, val_label, loss=0)
for index,modal_name in enumerate(modal_name_list+['fusion']):
    metric_dict[modal_name] = evl_metrics[index].get()
    gap_dict[modal_name] = metric_dict[modal_name]['gap']
print(gap_dict)


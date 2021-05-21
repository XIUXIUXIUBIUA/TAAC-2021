import torch
import torch.nn as nn
from tqdm import tqdm

import numpy as np
import utils.train_util as train_util

def training_loop(model, loader, loss_compute,modal_name_list, epoch):
    model.train()
    losses = []
    loss_weight = {'video':0.8,'audio':0.5,'fusion':0.8}
    for i, batch in enumerate(tqdm(loader, desc=f'train ({epoch})')):
        
        loss_compute.optimizer.zero_grad()
        
        video,audio,text,label = batch
        inputs_dict={}
        inputs_dict['video'] = video
        inputs_dict['audio'] = audio
        inputs_dict['text'] = text
        pred = model(inputs_dict)
        loss_dict = {}
        loss = 0
        for modal in (modal_name_list + ['fusion']):
            if(modal=='fusion'):
                loss_dict[modal] = loss_compute(pred['tagging_output_'+modal]['predictions'],label)
            else:
                loss_dict[modal] = loss_compute(pred['tagging_output_'+modal]['predictions'],label,pred[modal+'_loss_weight'])
        for key in loss_dict:
            loss += loss_dict[key]
        losses.append(loss.item())
        # 反向传播计算梯度
        loss.backward()
        # 更新网络参数
        loss_compute.optimizer.step()
        # print(loss_compute.optimizer.param_groups[0]['lr'])
        loss_compute.lr_scheduler.step()
        
    return losses
def validating_loop(model, loader, loss_compute,modal_name_list,epoch):
    model.eval()
    tagging_class_num = 82
    evl_metrics = [train_util.EvaluationMetrics(tagging_class_num, top_k=20)
                               for i in range(len(modal_name_list)+1)] #+1 for fusion
    for i in range(len(evl_metrics)):
        evl_metrics[i].clear()
    metric_dict = {}
    gap_dict = {}
    for i,batch in enumerate(loader):
        video,audio,text,label = batch
        inputs_dict={}
        inputs_dict['video'] = video
        inputs_dict['audio'] = audio
        inputs_dict['text'] = text
        pred_dict = model(inputs_dict)
        for index,modal_name in enumerate(modal_name_list+['fusion']):
            pred = pred_dict['tagging_output_'+modal_name]
            val_label = label
            loss = loss_compute(pred['predictions'],val_label)
            pred = pred['predictions'].detach().cpu().numpy()
            val_label = label.cpu().numpy()
            #print(np.array(pred_gap))
            #print(val_label_gap)
            gap = train_util.calculate_gap(pred, val_label)
            # print('gap: ',gap)
            evl_metrics[index].accumulate(pred, val_label, loss=0)
    for index,modal_name in enumerate(modal_name_list+['fusion']):
        metric_dict[modal_name] = evl_metrics[index].get()
        gap_dict[modal_name] = metric_dict[modal_name]['gap']
    return gap_dict

def testing_loop():
    pass
    
    
        
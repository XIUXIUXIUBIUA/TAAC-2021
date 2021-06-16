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


if __name__ == '__main__':
    config_path = './config/config.yaml'
    config = yaml.load(open(config_path))
    dataset = TestingDataset(config['DatasetConfig'])
    '''
    model_path_1 = "../checkpoint/0604/resnet50/epoch_22 0.7814.pt" # 已保存模型的路径
    model_path_2 = '../checkpoint/0604/resnet50/epoch_34 0.7822.pt'
    model_path_3 = '../checkpoint/0604/resnet50_with_selfsupervised/epoch_20 0.7812.pt'
    '''
    
    '''
    # val 7845 test 7894
    model_path_1 = "../checkpoint/0604/resnet50/epoch_22 0.7814.pt" # 已保存模型的路径
    model_path_2 = '../checkpoint/0604/resnet50/epoch_34 0.7822.pt'
    model_path_3 = '../checkpoint/0604/resnet50/epoch_20 0.7802.pt'
    model_path_4 = '../checkpoint/0604/lr/epoch_24 0.7801.pt'
    model_path_5 = '../checkpoint/0605/01/epoch_24 0.7799.pt'
    model_path_6 = '../checkpoint/0604/lr/epoch_42 0.7810.pt'
    model_path_7 = '../checkpoint/0604/lr/epoch_22 0.7797.pt'
    model_path_8 = '../checkpoint/0604/resnet50_with_selfsupervised/epoch_16 0.7744.pt'
    model_path_9 = '../checkpoint/0604/resnet50_with_selfsupervised/epoch_20 0.7812.pt'
    model_path_10 = '../checkpoint/0604/lr/epoch_32 0.7805.pt'
    '''
    
    '''
    # val 7867 test ?
    model_path_1 = '../checkpoint/0607/01/epoch_56 0.7846.pt'
    model_path_2 = '../checkpoint/0607/01/epoch_42 0.7843.pt'
    model_path_3 = '../checkpoint/0607/02/epoch_28 0.7822.pt'
    model_path_4 = '../checkpoint/0607/02/epoch_50 0.7835.pt'
    model_path_5 = '../checkpoint/0607/01/epoch_28 0.7831.pt'
    model_path_6 = '../checkpoint/0607/02/epoch_48 0.7830.pt'
    '''
    
    '''
    # val 7894 test 7914
    model_path_1 = '../checkpoint/0607/01/epoch_56 0.7846.pt'
    model_path_2 = '../checkpoint/0607/01/epoch_42 0.7843.pt'
    model_path_3 = '../checkpoint/0607/02/epoch_28 0.7822.pt'
    model_path_4 = '../checkpoint/0607/02/epoch_50 0.7835.pt'
    model_path_5 = '../checkpoint/0607/01/epoch_28 0.7831.pt'
    model_path_6 = '../checkpoint/0607/02/epoch_48 0.7830.pt'
    model_path_7 = '../checkpoint/0607/03/epoch_88 0.7873.pt'
    model_path_8 = '../checkpoint/0607/03/epoch_46 0.7862.pt'
    model_path_9 = '../checkpoint/0607/03/epoch_28 0.7855.pt'
    '''
    
    '''
    # val 7914 test 7923
    model_path_1 = '../checkpoint/0607/01/epoch_56 0.7846.pt'
    model_path_2 = '../checkpoint/0607/01/epoch_42 0.7843.pt'
    model_path_3 = '../checkpoint/0607/02/epoch_28 0.7822.pt'
    model_path_4 = '../checkpoint/0607/02/epoch_50 0.7835.pt'
    model_path_5 = '../checkpoint/0607/01/epoch_28 0.7831.pt'
    model_path_6 = '../checkpoint/0607/02/epoch_48 0.7830.pt'
    model_path_7 = '../checkpoint/0607/03/epoch_88 0.7873.pt'
    model_path_8 = '../checkpoint/0607/03/epoch_46 0.7862.pt'
    model_path_9 = '../checkpoint/0607/03/epoch_28 0.7855.pt'
    model_path_10 = '../checkpoint/0608/01/epoch_86 0.7877.pt'
    model_path_11 = '../checkpoint/0608/01/epoch_50 0.7873.pt'
    model_path_12 = '../checkpoint/0608/01/epoch_28 0.7869.pt'
    '''
    
    # val 0792 test 07938
    model_path_1 = '../checkpoint/0610/03/epoch_50 0.9010.pt'
    model_path_2 = '../checkpoint/0610/03/epoch_44 0.9004.pt'
    
    model_path_3 = '../checkpoint/0610/03/epoch_52 0.9011.pt'
    model_path_4 = '../checkpoint/0610/03/epoch_40 0.8996.pt'
    model_path_5 = '../checkpoint/0610/03/epoch_32 0.8901.pt'
    model_path_6 = '../checkpoint/0610/03/epoch_24 0.8793.pt'
    
    model_path_7 = '../checkpoint/0607/03/epoch_88 0.7873.pt'
    model_path_8 = '../checkpoint/0607/03/epoch_46 0.7862.pt'
    model_path_9 = '../checkpoint/0607/03/epoch_28 0.7855.pt'
    model_path_10 = '../checkpoint/0608/01/epoch_86 0.7877.pt'
    model_path_11 = '../checkpoint/0608/01/epoch_50 0.7873.pt'
    model_path_12 = '../checkpoint/0608/01/epoch_28 0.7869.pt'

    model_path_13 = '../checkpoint/0608/03/epoch_28 0.7877.pt'
    model_path_14 = '../checkpoint/0608/03/epoch_52 0.7890.pt'
    model_path_15 = '../checkpoint/0608/03/epoch_74 0.7896.pt'
    models_path = [model_path_1,model_path_3,model_path_4,model_path_5,model_path_6,
                   model_path_7,model_path_8,model_path_9,
                   model_path_10,model_path_11,
                   model_path_14,model_path_15]
    

    device = 'cuda'
    top_k=20
    output_json = './0610_03.json'
    models = []
    
    '''
    for path in models_path:
        model = Baseline(config['ModelConfig'])
        model.load_state_dict(torch.load(path))
        model.to(device)
        model.eval()
        models.append(model)
    '''
    for path in models_path:
        if((path.split('/')[2]+path.split('/')[3]=='060801')or(path.split('/')[2]=='0607')):
            config['ModelConfig']['fusion_head_params']['concat_feat_dim']['fusion'] = 30720
            config['ModelConfig']['audio_head_params']['max_frames'] = 300
        else:
            config['ModelConfig']['fusion_head_params']['concat_feat_dim']['fusion'] = 29696
            config['ModelConfig']['audio_head_params']['max_frames'] = 200
        model = Baseline(config['ModelConfig'])
        model.load_state_dict(torch.load(path))
        model.to(device)
        model.eval()
        models.append(model)
    tagging_class_num = 82
    output_result = {}
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            feat_dict = dataset[i]

            video = feat_dict['video']
            audio = feat_dict['audio']
            text = feat_dict['text_ids']
            text_mask = feat_dict['text_attention_mask']
            file_name = feat_dict['file_name']

            video = video.to(device)
            audio = audio.to(device)
            text = text.to(device)
            text_mask = text_mask.to(device)

            inputs_dict={}
            inputs_dict['video'] = video.unsqueeze(0)
            inputs_dict['audio'] = audio.unsqueeze(0)
            inputs_dict['text'] = text.unsqueeze(0)
            inputs_dict['attention_mask'] = text_mask.unsqueeze(0)

            scores = torch.zeros(1,82)
            for i,model in enumerate(models):
                pred_dict = model(inputs_dict)
                scores += pred_dict['tagging_output_fusion']['predictions'].cpu()
            scores = scores/len(models)
            
            scores,indices = scores[0].sort(descending=True)
            scores = scores.detach().numpy()
            indices = indices.detach().numpy()
            labels = [dataset.id2label[idx] for idx in indices]
            cur_output = {}
            output_result[file_name] = cur_output
            cur_output['result'] = [{"labels": labels[:top_k], "scores": ["%.2f" % scores[i] for i in range(top_k)]}]
        
    with open(output_json, 'w', encoding="utf-8") as f:
        json.dump(output_result, f, ensure_ascii=False, indent = 4)
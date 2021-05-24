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
    model_path = "../checkpoint/0524/epoch_28_0.7606.pt" # 已保存模型的路径
    device = 'cuda'
    top_k=20
    output_json = './h.json'
    model = Baseline(config['ModelConfig'])
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    tagging_class_num = 82
    output_result = {}
    for i in tqdm(range(len(dataset))):
        feat_dict = dataset[i]

        video = feat_dict['video']
        # audio = feat_dict['audio']
        text = feat_dict['text_ids']
        text_mask = feat_dict['text_attention_mask']
        file_name = feat_dict['file_name']

        video = video.to(device)
        # audio = audio.to(device)
        text = text.to(device)
        text_mask = text_mask.to(device)

        inputs_dict={}
        inputs_dict['video'] = video.unsqueeze(0)
        # inputs_dict['audio'] = audio.unsqueeze(0)
        inputs_dict['text'] = text.unsqueeze(0)
        inputs_dict['attention_mask'] = text_mask.unsqueeze(0)

        pred_dict = model(inputs_dict)
        scores = pred_dict['tagging_output_fusion']['predictions'].cpu()
        scores,indices = scores[0].sort(descending=True)
        scores = scores.detach().numpy()
        indices = indices.detach().numpy()
        labels = [dataset.id2label[idx] for idx in indices]
        cur_output = {}
        output_result[file_name] = cur_output
        cur_output['result'] = [{"labels": labels[:top_k], "scores": ["%.2f" % scores[i] for i in range(top_k)]}]
        
    with open(output_json, 'w', encoding="utf-8") as f:
        json.dump(output_result, f, ensure_ascii=False, indent = 4)
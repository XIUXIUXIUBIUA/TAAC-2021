import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from munch import Munch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

from dataloader.dataloader import MultimodaFeaturesDataset
from src.loss.loss_compute import SimpleLossCompute
from src.model.baseline_model import Baseline
from src.loop.run_epoch import training_loop,validating_loop
if __name__ == '__main__':
    # 定义配置文件路径并读入文件
    torch.multiprocessing.set_start_method('spawn')
    config_path = './config/config.yaml'
    config = yaml.load(open(config_path))
    device_ids = [0,1]
    # 定义数据集并封装dataloader
    train_dataset = MultimodaFeaturesDataset(config['DatasetConfig'],job='training')
    val_dataset = MultimodaFeaturesDataset(config['DatasetConfig'],job='valdation')
    train_loader = DataLoader(train_dataset,num_workers=8,
                              batch_size=config['DatasetConfig']['batch_size']*len(device_ids),
                              shuffle=True,
                              pin_memory=False,
                              collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset,num_workers=8,
                            batch_size=config['DatasetConfig']['batch_size']*len(device_ids),
                            pin_memory=False,
                            collate_fn=val_dataset.collate_fn)
    # 定义模型
    model = Baseline(config['ModelConfig'])
    model.to(train_dataset.device)
    modal_name_list = model.modal_name_list
    # 定义loss函数和优化器
    criterion = nn.BCELoss(reduction='none')# sum应该没问题吧
    # 不同部件采用不同的学习率
    classifier_params = list(map(id, model.classifier_dict.parameters()))
    base_params = filter(lambda p: id(p) not in classifier_params,
                         model.parameters())
    optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': model.classifier_dict.parameters(), 'lr': 1e-2}],lr=1e-4)
    
    model = torch.nn.DataParallel(model,device_ids)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3000,last_epoch=-1,gamma=0.1)
    loss_compute = SimpleLossCompute(criterion,optimizer,lr_scheduler)
    
    # 开始训练epoch
    epoch_num=100
    loss_epoch = []
    for epoch in range(epoch_num):
        loss = training_loop(model, train_loader, loss_compute, modal_name_list, epoch)
        loss_epoch.append(loss)
        if(epoch%5==0): # 每20个epoch 验证一次
            gap_dict = validating_loop(model,val_loader, loss_compute,modal_name_list, epoch)
            print(f'epoch({epoch}): ',gap_dict['fusion'])
        # break
    # 保存模型
    
    


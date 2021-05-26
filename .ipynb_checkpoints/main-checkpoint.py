import os
import yaml
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from munch import Munch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import glob
from dataloader.dataloader import MultimodaFeaturesDataset
from src.loss.loss_compute import SimpleLossCompute
from src.model.baseline_model import Baseline
from src.loop.run_epoch import training_loop,validating_loop
from torch.utils import tensorboard as tensorboard
from datetime import datetime

if __name__ == '__main__':
    # 定义配置文件路径并读入文件
    torch.multiprocessing.set_start_method('spawn',force=True)
    log_dir = os.path.join('./results/logs/',datetime.now().strftime("%Y%m%d%H%M%S"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    TBoard = tensorboard.SummaryWriter(log_dir=log_dir)
    config_path = './config/config.yaml'
    config = yaml.load(open(config_path))
    device_ids = [0]
    # 定义数据集并封装dataloader
    train_dataset = MultimodaFeaturesDataset(config['DatasetConfig'],job='training')
    val_dataset = MultimodaFeaturesDataset(config['DatasetConfig'],job='valdation')
    train_loader = DataLoader(train_dataset,num_workers=8,
                              batch_size=config['DatasetConfig']['batch_size']*len(device_ids),
                              shuffle=True,
                              pin_memory=True,
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
    '''
    # video+bert
    classifier_params = list(map(id, model.classifier_dict.parameters()))
    bert_params = list(map(id,model.head_dict['text'].parameters()))
    
    base_params = filter(lambda p: id(p) not in (classifier_params+bert_params),
                         model.parameters())
    optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': model.classifier_dict.parameters(), 'lr': 1e-2},
                {'params': model.head_dict['text'].parameters(),'lr': 1e-5}],lr=1e-4)
    '''
    
    # video only
    classifier_params = list(map(id, model.classifier_dict.parameters()))
    base_params = filter(lambda p: id(p) not in (classifier_params),
                         model.parameters())
    optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': model.classifier_dict.parameters(), 'lr': 1e-2}],lr=1e-4)
    
    # model = torch.nn.DataParallel(model,device_ids)
    warm_up_epochs = 5
    max_num_epochs = 50
    lr_milestones = [20,40,60]
    warm_up_with_multistep_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs else 0.1**len([m for m in lr_milestones if m <= epoch])
    warm_up_with_cosine_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
    else 0.5 * ( math.cos((epoch - warm_up_epochs) /(max_num_epochs - warm_up_epochs) * math.pi) + 1)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=warm_up_with_cosine_lr)
    loss_compute = SimpleLossCompute(criterion,optimizer,lr_scheduler)
    best_gap = 0
    # 开始训练epoch
    epoch_num=50
    loss_epoch = []
    for epoch in range(epoch_num):
        loss = training_loop(model, train_loader, loss_compute, modal_name_list,train_dataset.device, epoch,TBoard)
        loss_epoch.append(loss)
        if(epoch%2==0): # 每2个epoch 验证一次
            gap_dict = validating_loop(model,val_loader, loss_compute,modal_name_list,train_dataset.device, epoch,TBoard)
            
            for modal in modal_name_list+['fusion']:
                print(f'epoch({epoch})({modal}): ',gap_dict[modal])
                
            if(gap_dict['fusion']>best_gap):
                best_gap = gap_dict['fusion']
                model_name = f'../checkpoint/0524/01/epoch_{epoch}_{best_gap}.pt'
                # torch.save(model.state_dict(),model_name)
        # break
    # 保存模型
    
    

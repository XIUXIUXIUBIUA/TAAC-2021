import os
import yaml
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from munch import Munch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import glob
from dataloader.dataloader import MultimodaFeaturesDataset,Datasetfortextcnn
from src.loss.loss_compute import SimpleLossCompute
from src.model.baseline_model import Baseline
from src.loop.run_epoch import training_loop,validating_loop
from torch.utils import tensorboard as tensorboard
from datetime import datetime
import numpy as np
import random
# from apex import amp
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # 定义配置文件路径并读入文件
    torch.multiprocessing.set_start_method('spawn',force=True)
    setup_seed(seed=2022)
    log_dir = os.path.join('./results/logs/',datetime.now().strftime("%Y%m%d%H%M%S"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    TBoard = tensorboard.SummaryWriter(log_dir=log_dir)
    config_path = './config/config.yaml'
    config = yaml.load(open(config_path))
    device_ids = [0]
    # 定义数据集并封装dataloader
    if(config['ModelConfig']['text_head_type']=='BERT'):
        train_dataset = MultimodaFeaturesDataset(config['DatasetConfig'],job='training')
        val_dataset = MultimodaFeaturesDataset(config['DatasetConfig'],job='valdation')
    else:
        train_dataset = Datasetfortextcnn(config['DatasetConfig'],job='training')
        val_dataset = Datasetfortextcnn(config['DatasetConfig'],job='valdation')
    
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
    
    # 加载自监督模型
    model_path = '../checkpoint/0608/enhance_VT_lr/50_wp.pt'
    model_dict = model.state_dict() # 定义模型的参数字典
    extractor = torch.load(model_path) # 加载预训练模型
    # state_dict = {k:v for k,v in extractor.items() if k in model_dict.keys()}
    state_dict = {k:v for k,v in extractor.items() if ((k.split('.')[0]!='classifier_dict')and(k.split('.')[0]!='fusion_head_dict')and(k.split('.')[1]!='fusion')and(k in model_dict.keys()))} # 不加载classifier的参数
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    
    
    model.to(train_dataset.device)
    modal_name_list = model.modal_name_list
    # 定义loss函数和优化器
    criterion = nn.BCELoss(reduction='none')# sum应该没问题吧
    # 不同部件采用不同的学习率
    
    # video+bert
    classifier_params = list(map(id, model.classifier_dict.parameters()))
    bert_params = list(map(id,model.head_dict['text'].parameters()))
    # bert_params = []
    audio_params = list(map(id,model.head_dict['audio'].parameters()))
    base_params = filter(lambda p: id(p) not in (classifier_params+bert_params+audio_params),
                         model.parameters())
    params_group_list = [
                {'params': base_params},
                {'params': model.classifier_dict['video'].parameters(), 'lr': 1e-2},
                {'params': model.classifier_dict['audio'].parameters(), 'lr': 1e-3},
                {'params': model.classifier_dict['text'].parameters(), 'lr': 1e-3},
                {'params': model.classifier_dict['fusion'].parameters(), 'lr': 1e-3},
                {'params': model.head_dict['audio'].parameters(),'lr': 1e-3},
                {'params': model.head_dict['text'].parameters(),'lr': 1e-5}
                ]
    
    optimizer = torch.optim.Adam(params_group_list,lr=1e-4)
    
    '''
    # video only
    classifier_params = list(map(id, model.classifier_dict.parameters()))
    base_params = filter(lambda p: id(p) not in (classifier_params),
                         model.parameters())
    optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': model.classifier_dict.parameters(), 'lr': 1e-2}],lr=1e-4)
    '''
    # model = torch.nn.DataParallel(model,device_ids)
    #model.load_state_dict(torch.load(model_path))
    # model,optimizer = amp.initialize(model, optimizer, opt_level="O1")
    warm_up_epochs = 5
    max_num_epochs = 100
    lr_milestones = [20,40,50,60]
    warm_up_with_multistep_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs else 0.1**len([m for m in lr_milestones if m <= epoch])
    warm_up_with_cosine_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
    else 0.5 * ( math.cos((epoch - warm_up_epochs) /(max_num_epochs - warm_up_epochs) * math.pi) + 1)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=warm_up_with_multistep_lr)
    
    base_lr = [1e-6,1e-4,1e-5,1e-5,1e-5,1e-5,1e-7]
    max_lr = [1e-4,1e-2,1e-3,1e-3,1e-3,1e-3,1e-5]
    # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,step_size_up=700,mode='triangular2',cycle_momentum=False)
    loss_compute = SimpleLossCompute(criterion,optimizer,lr_scheduler)
    best_gap = 0
    # 开始训练epoch
    epoch_num=100
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
                save_path = '../checkpoint/0609/02/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                model_name = f'epoch_{epoch} '+str(best_gap)[:6]+'.pt'
                torch.save(model.state_dict(),save_path+model_name)
        # break
    # 保存模型
    
    


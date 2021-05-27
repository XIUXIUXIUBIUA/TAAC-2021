import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLossCompute(object):
    # 包括loss计算和反向传播计算梯度，以及用optimizer更新
    def __init__(self, criterion, optimizer, lr_scheduler):
        self.criterion = criterion # 用来计算损失
        self.lr_scheduler = lr_scheduler # 用来更新梯度
        self.optimizer = optimizer
        
    def __call__(self, pred, target,alpha=None):
        loss = self.criterion(pred, target)
        if(alpha==None):
            loss_mean = torch.mean(torch.sum(loss,1))
        else:
            loss_mean = torch.mean(torch.sum(loss,1)*alpha)
        return loss_mean

class ContrastiveLossCompute(object):
    # 包括loss计算和反向传播计算梯度，以及用optimizer更新
    def __init__(self, optimizer, lr_scheduler, margin):
        self.lr_scheduler = lr_scheduler # 用来更新梯度
        self.optimizer = optimizer
        self.margin = margin
    def __call__(self, pred):
        video_rep = pred['video']
        text_rep = pred['text']
        B = video_rep.shape[0]
        
        video_pos = video_rep[:(B//2)]
        text_pos = text_rep[:(B//2)]
        video_neg = video_rep[(B//2):]
        text_neg = text_rep[(B//2):]
        
        pos_pos_dist = F.pairwise_distance(video_pos,text_pos,p=2)
        pos_neg_dist = F.pairwise_distance(video_pos,text_neg,p=2)
        neg_pos_dist = F.pairwise_distance(video_neg,text_pos,p=2)
        neg_neg_dist = F.pairwise_distance(video_neg,text_neg,p=2)
        
        loss1 = self.hinge_loss(pos_pos_dist, pos_neg_dist)
        loss2 = self.hinge_loss(pos_pos_dist, neg_pos_dist)
        loss3 = self.hinge_loss(neg_neg_dist, pos_neg_dist)
        loss4 = self.hinge_loss(neg_neg_dist, neg_pos_dist)
        
        loss = loss1 + loss2 + loss3 + loss4
        
        return loss
    def hinge_loss(self,low, high):
        return torch.mean(F.relu(low + self.margin - high))
    
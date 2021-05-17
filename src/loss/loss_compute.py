import torch
import torch.nn as nn

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
        '''
        loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
        '''
        return loss_mean
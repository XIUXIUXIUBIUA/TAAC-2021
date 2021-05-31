import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
class SE(nn.Module):
    
    def __init__(self,drop_rate, hidden1_size, gating_reduction,concat_feat_dim ,gating_last_bn=False):
        super(SE,self).__init__()
        self.drop_rate = drop_rate
        self.hidden1_size = hidden1_size # mafp: 1024
        self.gating_reduction = gating_reduction # mafp: 8
        self.gating_last_bn = gating_last_bn # 
        
        self.dropout_1 = nn.Dropout(p=drop_rate)
        self.concat_feat_dim = concat_feat_dim
        self.hidden1_weights = Parameter(torch.randn((self.concat_feat_dim, self.hidden1_size)))
        self.bn_1 = nn.BatchNorm1d(self.hidden1_size)
        self.gating_weights_1 = Parameter(torch.randn((self.hidden1_size, self.hidden1_size // self.gating_reduction)))
        self.bn_2 = nn.BatchNorm1d(self.hidden1_size//self.gating_reduction)
        self.gating_weights_2 = Parameter(torch.randn((self.hidden1_size // self.gating_reduction, self.hidden1_size)))

        if(self.gating_last_bn):
            self.bn_3 = nn.BatchNorm1d(self.hidden1_size)
        for name,parameter in self.named_parameters():
            if(name in ['hidden1_weights','gating_weights_1','gating_weights_2']):
                nn.init.kaiming_normal_(parameter)
            
            #if(name in ['text_factor','video_factor','fusion_weights']):
            #    nn.init.xavier_uniform_(parameter)
    def forward(self,input_list):
        #if(len(input_list)==1):
        concat_feat = torch.cat(input_list,dim=1) 
        if(self.drop_rate>0.):
            # mafp: care 一下
            concat_feat = self.dropout_1(concat_feat)
        activation = torch.matmul(concat_feat, self.hidden1_weights)
        activation = self.bn_1(activation)

        gates = torch.matmul(activation,self.gating_weights_1)
        gates = F.relu(self.bn_2(gates))
        gates = torch.matmul(gates, self.gating_weights_2)

        if(self.gating_last_bn):
            gates = self.bn_3(gates)
        gates = torch.sigmoid(gates)
        activation = torch.mul(activation, gates)
        return activation
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models as models
class NeXtVLAD(nn.Module):
    def __init__(self,feature_size, max_frames, nextvlad_cluster_size, expansion, groups):
        super(NeXtVLAD,self).__init__()
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.nextvlad_cluster_size = nextvlad_cluster_size
        self.expansion = expansion
        self.groups = groups
        
        self.linear_1 = nn.Linear(self.feature_size,self.expansion*self.feature_size)
        
        # 在forward的时候要在后面接上sigmoid
        self.attention_1 = nn.Linear(self.expansion*self.feature_size,self.groups)

        self.cluster_weights = Parameter(torch.randn([self.expansion*self.feature_size,self.groups*self.nextvlad_cluster_size]))
        self.cluster_weights_2 = Parameter(torch.randn([1, (self.expansion * self.feature_size // self.groups), self.nextvlad_cluster_size]))
        # self.cluster_weights = nn.Linear()
        self.bn_1 = nn.BatchNorm1d(self.groups * self.nextvlad_cluster_size)
        self.bn_2 = nn.BatchNorm1d(self.nextvlad_cluster_size * (self.expansion * self.feature_size // self.groups))
        for name,parameter in self.named_parameters():
            if(name in ['linear_1','attention_1','cluster_weights','cluster_weights2']):
                nn.init.kaiming_normal_(parameter)
    def forward(self,input,mask=None):
        # input shape (B,M,N)
        _,seq_len,_ = input.shape
        input = self.linear_1(input) # shape (B,M,lambda*N)
        attention = self.attention_1(input)
        attention  = torch.sigmoid(attention) # shape (B,M,G)
        if mask is not None:
            attention = torch.mul(attention, mask.unsqueeze(-1))
        attention = torch.reshape(attention,[-1,seq_len*self.groups,1])
        # 分组后（也就是降维）的特征维度
        feature_size_ = self.expansion * self.feature_size // self.groups 

        reshape_input = torch.reshape(input,[-1,self.expansion*self.feature_size])
        activation = torch.matmul(reshape_input,self.cluster_weights)
        activation = self.bn_1(activation)
        activation = torch.reshape(activation, [-1, seq_len * self.groups, self.nextvlad_cluster_size])
        activation = F.softmax(activation, dim=-1)
        activation = torch.mul(activation, attention)

        a_sum = torch.sum(activation,-2,keepdim=True)
        a = torch.mul(a_sum, self.cluster_weights_2)

        activation = torch.transpose(activation, 1, 2)
        reshape_input = torch.reshape(input, [-1, seq_len * self.groups, feature_size_])

        vlad = torch.matmul(activation, reshape_input)
        vlad = torch.transpose(vlad, 1, 2)
        vlad = torch.sub(vlad,a)
        
        vlad = F.normalize(vlad,p=2,dim=1)
        vlad = torch.reshape(vlad,[-1, self.nextvlad_cluster_size * feature_size_])
        vlad = self.bn_2(vlad)
        return vlad

    
class RawNeXtVLAD(nn.Module):
    def __init__(self,feature_size, max_frames, nextvlad_cluster_size, expansion, groups):
        super(RawNeXtVLAD,self).__init__()
        self.nextvlad = NeXtVLAD(feature_size, max_frames, nextvlad_cluster_size, expansion, groups)
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(2048,1024)
        
    def forward(self,input,mask=None):
        # 输入图像shape (batch,len,channel,H,W)
        B,S,C,H,W = input.shape
        input = input.contiguous().view(B*S,C,H,W)
        output = self.resnet50(input)
        output = output.contiguous().view(B,S,-1)
        if(mask!=None):
            output = self.nextvlad(output,mask)
        else:
            output = self.nextvlad(output)
        return output

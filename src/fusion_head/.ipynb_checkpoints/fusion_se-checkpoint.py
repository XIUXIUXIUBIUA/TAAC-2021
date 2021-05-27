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
        '''
        rank = 50
        d_model_text = 1024
        d_model_video = 16384
        self.post_fusion_dropout = nn.Dropout(p=0.1)
        self.text_factor = Parameter(torch.Tensor(rank, d_model_text + 1, d_model_text))
        self.video_factor = Parameter(torch.Tensor(rank, d_model_video + 1, d_model_video))
        
        self.fusion_weights = Parameter(torch.Tensor(1, rank))
        self.fusion_bias = Parameter(torch.Tensor(1, d_model_video))
        '''
        for name,parameter in self.named_parameters():
            if(name in ['hidden1_weights','gating_weights_1','gating_weights_2']):
                nn.init.kaiming_normal_(parameter)
            if(name in ['bn_1','bn_2']):
                nn.init.zeros_(parameter)
            #if(name in ['text_factor','video_factor','fusion_weights']):
            #    nn.init.xavier_uniform_(parameter)
    def forward(self,input_list):
        #if(len(input_list)==1):
        concat_feat = torch.cat(input_list,dim=1)
        ''' 
        else:
            batch_size,d_model_video = input_list[0].shape
            _video_h = torch.cat((Variable(torch.ones(batch_size,1), requires_grad=False), input_list[0]), dim=-1)
            _text_h = torch.cat((Variable(torch.ones(batch_size,1), requires_grad=False), input_list[1]), dim=-1)
            
            # _audio_h 是(B,S,d_audio+1), audio_factor 是 (rank,d+1,d_out)
            # 可以考虑 (B,S,d+1) --> (B*S,d+1)
            fusion_zy = torch.matmul(_video_h, self.video_factor) * torch.matmul(_text_h, self.text_factor)
            del _audio_h,_video_h
            fusion_memory = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
            concat_feat = fusion_memory.contiguous().view(batch_size,d_model_video)
        '''
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
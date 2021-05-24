import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class LogisticModel(nn.Module):
    def __init__(self, num_classes, input_dim, l2_penalty=None):
        super(LogisticModel,self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = 1024
        self.l2_penalty =  0.0 if l2_penalty==None else l2_penalty
        self.linear_1 = nn.Linear(input_dim, self.num_classes)
        #self.dropout = nn.Dropout(p=0.6)
        #self.linear_2 = nn.Linear(self.hidden_dim, num_classes)
        for name,parameter in self.named_parameters():
            if(name in ['linear_1','linear_2']):
                nn.init.kaiming_normal_(parameter)
    def forward(self,model_input):
        logits = self.linear_1(model_input)
        #logits = self.dropout(logits)
        #logits = self.linear_2(F.relu(logits))
        output = torch.sigmoid(logits)
        return {"predictions": output, "logits": logits}
        
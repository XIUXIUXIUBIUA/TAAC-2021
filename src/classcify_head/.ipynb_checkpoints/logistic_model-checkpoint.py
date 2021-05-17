import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class LogisticModel(nn.Module):
    def __init__(self, num_classes, input_dim, l2_penalty=None):
        super(LogisticModel,self).__init__()
        self.num_classes = num_classes
        self.l2_penalty =  0.0 if l2_penalty==None else l2_penalty
        self.linear_1 = nn.Linear(input_dim, num_classes)
    def forward(self,model_input):
        logits = self.linear_1(model_input)
        output = torch.sigmoid(logits)
        return {"predictions": output, "logits": logits}
        
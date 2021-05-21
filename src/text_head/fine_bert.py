import torch
import torch.nn as nn
from transformers import BertTokenizer,BertModel

class BERT(nn.Module):
    def __init__(self,bert_path):
        super(BERT,self).__init__()
        self.model = BertModel.from_pretrained(bert_path)
        self.linear = nn.Linear(768,1024)
        self.bn = nn.BatchNorm1d(1024)
    
    def forward(self,input_ids,attention_mask):
        output_1 = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooler = output_1.pooler_output
        pooler = self.linear(pooler)
        pooler = self.bn(pooler)
        return pooler
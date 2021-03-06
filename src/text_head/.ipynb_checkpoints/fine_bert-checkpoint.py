import torch
import torch.nn as nn
from transformers import BertTokenizer,BertModel
import torch.nn.functional as F
import torchtext
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

class TextCnn(nn.Module):
    def __init__(self, embed_num,embed_dim,feature_dim,kernel_num,kernel_sizes,embedding_path,embedding_name,dropout=0.5):
        super(TextCnn, self).__init__()

        Ci = 1
        Co = kernel_num

        self.embed = nn.Embedding(embed_num, embed_dim)
        vectors = torchtext.vocab.Vectors(name=embedding_name, cache=embedding_path)
        self.embed = self.embed.from_pretrained(vectors.vectors)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (f, embed_dim), padding = (2, 0)) for f in kernel_sizes])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(Co * len(kernel_sizes), feature_dim)
        self.bn = nn.BatchNorm1d(1024)
    def forward(self, x):
        x = self.embed(x)  # (N, token_num, embed_dim)
        x = x.unsqueeze(1)  # (N, Ci, token_num, embed_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, token_num) * len(kernel_sizes)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co) * len(kernel_sizes)]
        x = torch.cat(x, 1) # (N, Co * len(kernel_sizes))
        x = self.dropout(x)  # (N, Co * len(kernel_sizes))
        x = self.fc(x)  # (N, class_num)
        x = self.bn(x)
        return x
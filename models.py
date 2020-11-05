from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
from transformers import BertModel
from torch.autograd import Variable
import torch.nn.functional as F


class MABert(nn.Module):
    def __init__(self, bert, num_classes, bert_trainable=True):
        super(MABert, self).__init__()

        self.add_module('bert', bert)
        if not bert_trainable:
            for m in self.bert.parameters():
                m.requires_grad = False

        self.num_classes = num_classes

        self.class_weight = Parameter(torch.Tensor(num_classes, 768).uniform_(0, 1), requires_grad=False).cuda(0) #
        self.class_weight.requires_grad = True

    def forward(self, ids, token_type_ids, attention_mask, encoded_tag, tag_mask):
        token_feat = self.bert(ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask)[0]

        embed = self.bert.get_input_embeddings()
        tag_embedding = embed(encoded_tag)
        tag_embedding = torch.sum(tag_embedding * tag_mask.unsqueeze(-1), dim=1) \
                        / torch.sum(tag_mask, dim=1, keepdim=True)

        masks = torch.unsqueeze(attention_mask, 1)  # N, 1, L
        attention = (torch.matmul(token_feat, tag_embedding.transpose(0, 1))).transpose(1, 2).masked_fill(1 - masks.byte(), torch.tensor(-np.inf))
        attention = F.softmax(attention, -1)
        attention_out = attention @ token_feat   # N, labels_num, hidden_size
        attention_out = attention_out * self.class_weight
        pred = torch.sum(attention_out, -1)

        return pred

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.bert.parameters(), 'lr': lr * lrp},
            {'params': self.class_weight, 'lr': lr},
        ]

class Bert_Encoder(nn.Module):
    def __init__(self, bert, bert_trainable=True):
        super(Bert_Encoder, self).__init__()

        self.add_module('bert', bert)
        if not bert_trainable:
            for m in self.bert.parameters():
                m.requires_grad = False

    def forward(self, ids, token_type_ids, attention_mask):
        token_feat = self.bert(ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask)[0]
        sentence_feat = torch.sum(token_feat * attention_mask.unsqueeze(-1), dim=1) \
                        / torch.sum(attention_mask, dim=1, keepdim=True)

        return sentence_feat

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.bert.parameters(), 'lr': lrp},
        ]

class Discriminator(nn.Module):
    def __init__(self, num_classes, input_dim=768, num_hidden_discriminator=1, hidden_dim_discriminator=400):
        super(Discriminator, self).__init__()

        self.dropout = nn.Dropout(p=0.5)
        self.act = ReLU()#nn.LeakyReLU(0.2)#

        self.num_hidden_discriminator = num_hidden_discriminator
        self.hidden_list_discriminator = nn.ModuleList()
        for i in range(num_hidden_discriminator):
            dim = input_dim if i == 0 else hidden_dim_discriminator
            self.hidden_list_discriminator.append(nn.Linear(dim, hidden_dim_discriminator))

        self.Linear = nn.Linear(hidden_dim_discriminator, (num_classes + 1))
        self.output = nn.Softmax(dim=-1)

    def forward(self, feat):
        # x = self.dropout(feat)
        x = feat
        for i in range(self.num_hidden_discriminator):
            x = self.hidden_list_discriminator[i](x)
            x = self.act(x)
            # x = self.dropout(x)

        flatten = x
        logit = self.Linear(x)
        prob = self.output(logit)
        return flatten, logit, prob

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.hidden_list_discriminator.parameters(), 'lr': lr},
            {'params': self.Linear.parameters(), 'lr': lr},
            {'params': self.output.parameters(), 'lr': lr},
        ]

class Generator(nn.Module):
    def __init__(self, hidden_dim=768, input_dim=768, num_hidden_generator=3, hidden_dim_generator=2000):
        super(Generator, self).__init__()

        self.dropout = nn.Dropout(p=0.5)
        self.act = nn.ReLU()

        self.num_hidden_generator = num_hidden_generator
        self.hidden_list_generator = nn.ModuleList()
        for i in range(num_hidden_generator):
            dim = input_dim if i == 0 else hidden_dim_generator
            self.hidden_list_generator.append(nn.Linear(dim, hidden_dim_generator))
        self.output = nn.Linear(hidden_dim_generator, hidden_dim)

    def forward(self, feat):
        x = feat
        for i in range(self.num_hidden_generator):
            x = self.hidden_list_generator[i](x)
            x = self.act(x)
            # x = self.dropout(x)
        y = self.output(x)
        return y

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.hidden_list_generator.parameters(), 'lr': lr},
            {'params': self.output.parameters(), 'lr': lr},
        ]
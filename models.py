from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
from transformers import BertModel

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNBert(nn.Module):
    def __init__(self, bert, num_classes, t=0, co_occur_mat=None):
        super(GCNResnet, self).__init__()
        
        self.bert = bert
        
        self.num_classes = num_classes

        self.gc1 = GraphConvolution(768, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, co_occur_mat)
        self.adj = gen_adj(_adj)

    def forward(self, ids, token_type_ids, attention_mask, inp):
        token_feat = self.bert(ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)[0]
        sentence_feat = torch.sum(token_feat * attention_mask.unsqueeze(-1), dim=1) \
            / torch.sum(attention_mask, dim=1, )
        
        inp = inp[0]
        x = self.gc1(inp, self.adj)
        x = self.relu(x)
        x = self.gc2(x, self.adj)

        x = x.transpose(0, 1)
        x = torch.matmul(sentence_feat, x)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.bert.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]


def gcn_bert(num_classes, t, co_occur_mat=None):
    bert = BertModel.from_pretrained('bert-base-uncased')
    return GCNBert(bert, num_classes, t=t, co_occur_mat=co_occur_mat)

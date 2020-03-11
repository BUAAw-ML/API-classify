from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

from torch.autograd import Variable

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
        # print(support.shape)
        # print(adj.shape)
        # print(output.shape)
        # torch.Size([108, 1500])
        # torch.Size([108, 108])
        # torch.Size([108, 1500])

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
        super(GCNBert, self).__init__()
        
        self.add_module('bert', bert)
        for m in self.bert.parameters():
            m.requires_grad = True
        
        self.num_classes = num_classes

        # self.tanh1 = nn.Tanh()
        # self.linear0 = nn.Linear(768, 768)
        # self.w = nn.Parameter(torch.Tensor(768))

        #self.dropout = nn.Dropout(p=0.5)
        self.gc1 = GraphConvolution(768, 1500)
        self.relu1 = nn.LeakyReLU(0.2)
        self.gc2 = GraphConvolution(1500, 5000)
        self.relu2 = nn.LeakyReLU(0.2)
        self.gc3 = GraphConvolution(5000, 768)

        _adj = gen_A(num_classes, t, co_occur_mat)
        _adj = torch.FloatTensor(_adj)
        self.adj = nn.Parameter(gen_adj(_adj), requires_grad=True)
        #
        # self.linear1 = nn.Linear(768, 768)
        # #
        # self.relu = nn.LeakyReLU()
        # self.linear2 = nn.Linear(768, num_classes)

    def forward(self, ids, token_type_ids, attention_mask, inputs_tfidf, encoded_tag, tag_mask, tfidf_result):

        token_feat = self.bert(ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)[0]

        #print(token_feat.shape)
        # alpha = F.softmax(torch.matmul(self.tanh1(self.linear0(token_feat)), self.w), dim=-1).unsqueeze(-1)  # [16, seq_len, 1]
        # token_feat = token_feat * alpha  # [16, seq_len, 768]

        #torch.set_printoptions(threshold=np.inf)

        sentence_feat = torch.sum(token_feat * attention_mask.unsqueeze(-1), dim=1) \
            / torch.sum(attention_mask, dim=1, keepdim=True)  # [batch_size, seq_len, embeding] [16, seq_len, 768]
        #sentence_feat = self.dropout(sentence_feat)
        # print(alpha.shape)
        # print(token_feat.shape)
        # print(sentence_feat.shape)

        #sentence_feat = token_feat[:,0,:]

        embed = self.bert.get_input_embeddings()
        tag_embedding = embed(encoded_tag)
        tag_embedding = torch.sum(tag_embedding * tag_mask.unsqueeze(-1), dim=1) \
            / torch.sum(tag_mask, dim=1, keepdim=True)

        x = self.gc1(tag_embedding, self.adj)
        x = self.relu1(x)
        x = self.gc2(x, self.adj)
        x = self.relu2(x)
        x = self.gc3(x, self.adj)

        x = x.transpose(0, 1)
        x = torch.matmul(sentence_feat, x)

        # x = self.linear1(sentence_feat)
        # #x = self.dropout(x)
        # x = self.relu(x)
        # x = self.linear2(x)

        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.bert.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                {'params': self.gc3.parameters(), 'lr': lr},
                ]
    # def get_config_optim(self, lr, lrp):
    #     return [
    #             {'params': self.bert.parameters(), 'lr': lr * lrp},
    #             {'params': self.linear1.parameters(), 'lr': lr},
    #             {'params': self.linear2.parameters(), 'lr': lr},
    #             ]


def gcn_bert(num_classes, t, co_occur_mat=None):
    bert = BertModel.from_pretrained('bert-base-uncased')

    return GCNBert(bert, num_classes, t=t, co_occur_mat=co_occur_mat)

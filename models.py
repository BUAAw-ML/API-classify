from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
from torch.autograd import Variable
import pickle as pkl
from CosNormClassifier import CosNorm_Classifier
import numpy as np
from capsule_module import *

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
        self.linear1 = nn.Linear(in_features, out_features)


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #support = torch.matmul(input, self.weight)
        # support = self.linear1(input)
        #
        # # #output = support
        # support = torch.matmul(input, self.weight)
        # output = torch.matmul(support.transpose(1, 2), adj)
        # output = output.transpose(1, 2)

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
        super(GCNBert, self).__init__()

        self.aa = torch.FloatTensor(co_occur_mat.numpy()).cuda(0)

        self.add_module('bert', bert)
        for m in self.bert.parameters():
            m.requires_grad = True

        # for i in range(9, 11+1):#l in self.bert.encoder.layer:
        #     m = self.bert.encoder.layer[i]
        #     m.trainable = True
        #     for p in m.parameters():
        #         p.requires_grad = True

        # for i in range(8, 11+1):#l in self.bert.encoder.layer:
        #     self.bert.encoder.layer[i].requires_grad = True

        self.num_classes = num_classes

        # self.tanh1 = nn.Tanh()
        # self.linear0 = nn.Linear(768, 768)
        # self.w = nn.Parameter(torch.Tensor(768))

        self.attention = nn.Linear(768, num_classes, bias=False) #num_classes
        nn.init.xavier_uniform_(self.attention.weight)

        # self.dropout = nn.Dropout(p=0.5)
        self.gc1 = GraphConvolution(768, 2000)
        self.relu1 = nn.LeakyReLU(0.2)
        self.gc2 = GraphConvolution(2000, 768)

        _adj, origin_adj = gen_A(num_classes, t, co_occur_mat)

        # exist = (_adj > 0) * 1.0
        # factor = np.ones(_adj.shape[1])
        # self.res = torch.FloatTensor(np.dot(exist, factor)).cuda(1)

        _adj = torch.FloatTensor(_adj)
        # _adj = _adj.transpose(0, 1)
        # self.adj = nn.Parameter(gen_adj(_adj), requires_grad=False)  #gen_adj(_adj)
        self.adj = nn.Parameter(_adj, requires_grad=True)

        _nums = co_occur_mat.numpy().diagonal()
        self.class_weight = torch.FloatTensor(np.round(1 - _nums / _nums.max(),3)).cuda(0).unsqueeze(-1)
        # print(self.class_weight)

        _nums = _nums[:, np.newaxis]

        weight_adj = np.hstack([_nums, origin_adj])
        # print(weight_adj)
        self.weight_adj = torch.FloatTensor(origin_adj).cuda(0)

        self.linear0 = nn.Linear(num_classes, num_classes)

        # self.fc_hallucinator = nn.Linear(768, num_classes)
        # self.fc_selector = nn.Linear(768, num_classes)

        self.linear1 = nn.Linear(768, 400)
        self.relu2 = nn.LeakyReLU()
        self.linear2 = nn.Linear(400, num_classes)
        self.output_layer = nn.Linear(768, num_classes)

        #self.cosnorm_classifier = CosNorm_Classifier(768, num_classes)
        self.weight1 = torch.nn.Linear(num_classes, 1)
        self.weight2 = torch.nn.Linear(768, 1)
        # self.lstm_hid_dim = num_classes / 2
        # self.lstm = torch.nn.LSTM(num_classes, hidden_size=self.lstm_hid_dim, num_layers=2,
        #                     batch_first=True, bidirectional=True)
        self.weight0 = torch.nn.Linear(768, 1)

        self.weight3 = Parameter(torch.Tensor(13, 1))
        self.weight3.data.uniform_(0, 1)

        # self.memory = Parameter(torch.Tensor(num_classes, 768), requires_grad=False).cuda(0)

        self.memory = torch.zeros(108, 768).cuda(0)
        self.relu = nn.ReLU()

        self.class_weight = Parameter(torch.Tensor(num_classes, 768).uniform_(0, 1), requires_grad=False).cuda(0) #
        self.class_weight.requires_grad = True


    def init_hidden(self, batch_size):
        return (torch.randn(4, batch_size, self.lstm_hid_dim).cuda(0),
                torch.randn(4, batch_size, self.lstm_hid_dim).cuda(0))

    def forward(self, ids, token_type_ids, attention_mask, inputs_tfidf, encoded_tag, tag_mask, tag_embedding_file,
                tfidf_result, title_ids, title_token_type_ids, title_attention_mask):


        token_feat = self.bert(ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)[2]
        token_feat = torch.stack(token_feat, dim=3) #[batch_size, seq_len, 768, layer_num]
        token_feat = torch.matmul(token_feat,  self.weight3).squeeze(-1)

        # token_feat = self.bert(ids,
        #     token_type_ids=token_type_ids,
        #     attention_mask=attention_mask)[0]  # [batch_size, seq_len, embeding] [16, seq_len, 768]


        # hidden_state = self.init_hidden(token_feat.shape[0])
        # token_feat, _ = self.lstm(token_feat, hidden_state)
        # token_feat = self.linear2(token_feat)

        #print(token_feat.shape)

        # alpha = (self.weight0(token_feat).squeeze(-1)).masked_fill(1 - attention_mask.byte(), torch.tensor(-np.inf))
        #
        # alpha = F.softmax(alpha, -1).unsqueeze(1)  #16, seq_len
        #
        # sentence_feat = alpha @ token_feat  #16,1,768

        # [16, seq_len, 768]

        # torch.set_printoptions(threshold=np.inf)

        # print(inputs_tfidf)
        # exit()
        # * inputs_tfidf.unsqueeze(-1)

        # sentence_feat = torch.sum(token_feat * attention_mask.unsqueeze(-1), dim=1) \
        #     / torch.sum(attention_mask, dim=1, keepdim=True)  # [batch_size, seq_len, embeding] [16, seq_len, 768]
        # # sentence_feat = sentence_feat.unsqueeze(1)


        # sentence_feat = token_feat[:,0,:]

        tag_embedding = self.bert(encoded_tag,
            attention_mask=tag_mask)[0]

        # embed = self.bert.get_input_embeddings()
        # tag_embedding = embed(encoded_tag)  #num_classes, 7, 768

        tag_embedding = torch.sum(tag_embedding * tag_mask.unsqueeze(-1), dim=1) \
            / torch.sum(tag_mask, dim=1, keepdim=True)

        # title_token_feat = self.bert(title_ids,
        #     token_type_ids=title_token_type_ids,
        #     attention_mask=title_attention_mask)[0]
        # title_feat = title_token_feat[:, 0, :]

        # with open(tag_embedding_file, 'rb') as fp:
        #     feats = pkl.load(fp)#, encoding='utf-8')
        # tag_embedding2 = feats.tolist()
        # tag_embedding2 = torch.tensor(tag_embedding2).cuda(1)
        #
        # tag_embedding2 = self.linear1(tag_embedding2)
        #
        # values_memory = torch.sigmoid(self.fc_hallucinator(self.weight_adj)).squeeze(-1).unsqueeze(0)
        # values_memory = values_memory.softmax(dim=1)
        #
        # concept_selector = self.fc_selector(sentence_feat)
        # concept_selector = concept_selector.tanh()
        #
        # weight2 = torch.sigmoid(self.weight2(attention_out)).squeeze(-1)

        # attention_out = torch.sum(attention_out, -1)

        # attention_out = torch.sum(attention_out, dim=2)
        # attention_out = torch.sum(attention_out, 1) / self.num_classes
        #
        # x = self.gc1(tag_embedding, self.adj)
        # x = self.relu1(x)
        # x = self.gc2(x, self.adj)
        # # #
        # x = x.transpose(0, 1)
        # x = torch.matmul(sentence_feat, x)
        #
        # x = torch.mul(sentence_feat.unsqueeze(1), x)
        # x = torch.sum(x, -1)

        # tag_embedding = t orch.matmul(self.adj, tag_embedding)

        masks = torch.unsqueeze(attention_mask, 1)#.clone()  # N, 1, L
        # masks[:,:, 0] = 0

        # attention = self.attention(token_feat).transpose(1, 2).masked_fill(1 - masks.byte(), torch.tensor(-np.inf))  # N, labels_num, L
        attention = (torch.matmul(token_feat, tag_embedding.transpose(0, 1))).transpose(1, 2).masked_fill(1 - masks.byte(), torch.tensor(-np.inf))

        attention = F.softmax(attention, -1)

        # attention_out = attention * confidence

        attention_out = attention @ token_feat   # N, labels_num, hidden_size

        # x = self.gc1(attention_out, self.adj)
        # x = self.relu1(x)
        # attention_out = self.gc2(x, self.adj)

        # attention_out = torch.sum(attention_out, -1)
        # attention_out = torch.matmul(self.adj, attention_out)

        attention_out = attention_out * self.class_weight

        # self.memory = torch.mean(attention_out, 0).clone()

        # pred = attention_out + x

        # x = torch.cat((x, attention_out), 2)

        # pred = self.output_layer(attention_out)  # + x

        pred = torch.sum(attention_out, -1)

        # pred *= torch.sigmoid(self.weight3)

        # pred *= torch.sigmoid(self.weight3)


        #x = x.unsqueeze(0)
        #print(x.shape)
        #x = sentence_feat * x
        # x = self.linear1(sentence_feat)
        # x = self.relu1(x)

        # attention_out = attention_out.squeeze(-1)

        # m1 = torch.matmul(tag_embedding, token_feat.transpose(1, 2))
        # label_att = torch.bmm(m1, token_feat)

        # weight1 = torch.sigmoid(self.weight1(x))
        # weight2 = torch.sigmoid(self.weight2(attention_out))
        # weight1 = weight1 / (weight1 + weight2)
        # weight2 = 1 - weight1

        # doc = weight1 * label_att + weight2 * attention_out
        # # doc = attention_out + values_memory.unsqueeze(-1) * label_att
        #
        # values_memory = torch.sigmoid(self.fc_hallucinator(self.weight_adj)).squeeze(-1).unsqueeze(0)

        # w1 = torch.sigmoid(self.weight1(self.weight_adj)).squeeze(-1).unsqueeze(0)

        # pred = attention_out #+ x

        # pred = 0.5 * torch.sigmoid(attention_out) + 0.5 * torch.sigmoid(x)
        # pred = torch.cat((attention_out, x), -1)
        # pred = attention_out
        # pred = torch.sum(x, -1)
        # pred = torch.sigmoid(pred)

        # avg_sentence_embeddings = torch.sum(pred, 1) / self.num_classes
        # pred = torch.matmul(avg_sentence_embeddings, x)

        # pred = self.linear0(attention_out.transpose(1, 2))
        # pred = pred.transpose(1, 2).squeeze(1)
        # pred = torch.matmul(pred, x)
        # pred = self.output_layer(pred)

        # pred = torch.matmul(pred, self.adj.transpose(0, 1))

        # pred = self.linear1(attention_out).squeeze(-1)

        # x = self.linear1(sentence_feat)  #sentence_feat + concept_selector *
        # x = self.relu2(x)
        # pred = self.linear2(x)
        # pred = x
        # print(pred.shape)

        return pred

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.bert.parameters(), 'lr': lr*lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                {'params': self.linear0.parameters(), 'lr': lr},
                {'params': self.linear1.parameters(), 'lr': lr},
                {'params': self.linear2.parameters(), 'lr': lr},
                {'params': self.weight1.parameters(), 'lr': lr},
                {'params': self.weight2.parameters(), 'lr': lr},
                {'params': self.weight3, 'lr': lr},
                {'params': self.adj, 'lr': lr},
                {'params': self.class_weight, 'lr': lr},
                {'params': self.attention.parameters(), 'lr': lr},
                {'params': self.output_layer.parameters(), 'lr': lr},
                ]
    # def get_config_optim(self, lr, lrp):
    #     return [
    #             {'params': self.bert.parameters(), 'lr': lr * lrp},
    #             {'params': self.linear1.parameters(), 'lr': lr},
    #             {'params': self.linear2.parameters(), 'lr': lr},
    #             ]


def gcn_bert(num_classes, t, co_occur_mat=None):
    bert = BertModel.from_pretrained('bert-base-uncased',  output_hidden_states=True)  #
    return GCNBert(bert, num_classes, t=t, co_occur_mat=co_occur_mat)


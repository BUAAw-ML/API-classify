from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
from transformers import BertModel
from torch.autograd import Variable
import torch.nn.functional as F


class MABert(nn.Module):
    def __init__(self, bert, num_classes, bert_trainable=True, device=0):
        super(MABert, self).__init__()

        self.add_module('bert', bert)
        if not bert_trainable:
            for m in self.bert.parameters():
                m.requires_grad = False

        self.num_classes = num_classes

        self.class_weight = Parameter(torch.Tensor(num_classes, 768).uniform_(0, 1), requires_grad=False).cuda(device)
        self.class_weight.requires_grad = True

        self.discriminator = Parameter(torch.Tensor(1, 768).uniform_(0, 1), requires_grad=False).cuda(device)
        self.discriminator.requires_grad = True

        self.Linear1 = nn.Linear(768, 500)
        self.Linear2 = nn.Linear(500, 1)
        self.act = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.output = nn.Softmax(dim=-1)

    # def forward(self, ids, token_type_ids, attention_mask, encoded_tag, tag_mask, feat):
    #     token_feat = self.bert(ids,
    #                            token_type_ids=token_type_ids,
    #                            attention_mask=attention_mask)[0] #N, L, hidden_size
    #
    #     # print(token_feat.shape)
    #
    #     sentence_feat = torch.sum(token_feat * attention_mask.unsqueeze(-1), dim=1) \
    #                     / torch.sum(attention_mask, dim=1, keepdim=True)#N, hidden_size
    #
    #     embed = self.bert.get_input_embeddings()
    #     tag_embedding = embed(encoded_tag)
    #     tag_embedding = torch.sum(tag_embedding * tag_mask.unsqueeze(-1), dim=1) \
    #                     / torch.sum(tag_mask, dim=1, keepdim=True)  #labels_num, hidden_size
    #
    #     masks = torch.unsqueeze(attention_mask, 1)  # N, 1, L  .bool()
    #     attention = (torch.matmul(token_feat, tag_embedding.transpose(0, 1))).transpose(1, 2).masked_fill((1 - masks.byte()), torch.tensor(-np.inf))
    #
    #     # similarity = (torch.matmul(token_feat, tag_embedding.transpose(0, 1))).transpose(1, 2).masked_fill(
    #     #     (1 - masks.byte()), torch.tensor(0))
    #     # similarity = torch.sum(similarity, -1)#.unsqueeze(-1)
    #     # similarity = torch.sum(similarity, -1)
    #
    #     attention = F.softmax(attention, -1)
    #     attention_out = attention @ token_feat   # N, labels_num, hidden_size
    #
    #     attention_out = attention_out * self.class_weight
    #     attention_out = torch.sum(attention_out, -1)
    #     logit = torch.sigmoid(attention_out)
    #     #################fake sample process#######
    #     feat = feat[:,:token_feat.shape[1],:] # N, L, hidden_size
    #     # feat += token_feat.detach()
    #     # feat = torch.mean(feat, 1)
    #     attention_fake = (torch.matmul(feat, tag_embedding.transpose(0, 1))).transpose(1, 2).masked_fill(
    #         (1 - masks.byte()), torch.tensor(-np.inf))
    #
    #     # similarity_fake = (torch.matmul(feat, tag_embedding.transpose(0, 1))).transpose(1, 2).masked_fill(
    #     #     (1 - masks.byte()), torch.tensor(0))
    #     # similarity_fake = torch.sum(similarity_fake, -1)#.unsqueeze(-1)
    #     # similarity_fake = torch.sum(similarity_fake, -1)
    #
    #     attention_fake = F.softmax(attention_fake, -1)
    #     attention_out_fake = attention_fake @ feat  # N, labels_num, hidden_size
    #     # discrimate = torch.matmul(feat, tag_embedding.transpose(0, 1))
    #     attention_out_fake = attention_out_fake * self.class_weight
    #     attention_out_fake = torch.sum(attention_out_fake, -1)
    #
    #     # discrimate = torch.mean(attention_out_fake, -2, keepdim=True)
    #     # discrimate = torch.sum(discrimate, -1, keepdim=True)
    #     #################
    #     # pred = torch.cat((discrimate, logit), -1)
    #
    #     # discrimate = torch.sum(torch.matmul(feat, self.class_weight.transpose(0, 1)), -1, keepdim=True)
    #     # attention_out = attention_out * self.class_weight
    #
    #     # pred = torch.cat((attention_out_fake, attention_out), -2)
    #     # logit = self.Linear1(logit)
    #     # logit = self.act(logit)
    #     # logit = self.Linear2(logit).squeeze(-1)
    #     # logit = torch.sigmoid(logit)
    #
    #     # logit = pred[:,self.num_classes:]
    #
    #     # prob =  torch.sigmoid(attention_out_fake)
    #
    #     attention_out = torch.sum(attention_out, -1, keepdim=True)
    #     attention_out_fake = torch.sum(attention_out_fake, -1, keepdim=True)
    #
    #     prob = torch.cat((attention_out_fake,attention_out),-1)
    #     # print(prob)
    #     # print(self.output(prob))
    #     prob = self.output(prob)[:,0]
    #
    #
    #     #################
    #     # masks = torch.unsqueeze(attention_mask, 1)  # N, 1, L  .bool()
    #     # attention = (torch.matmul(token_feat, self.discriminator.transpose(0, 1))).transpose(1, 2).masked_fill(
    #     #     (1 - masks.byte()), torch.tensor(-np.inf))
    #     # attention = F.softmax(attention, -1)
    #     # attention_out = attention @ token_feat  # N, 1, hidden_size
    #     #
    #     # feat = feat[:, :token_feat.shape[1], :]  # N, L, hidden_size
    #     # attention_fake = (torch.matmul(feat, self.discriminator.transpose(0, 1))).transpose(1, 2).masked_fill(
    #     #     (1 - masks.byte()), torch.tensor(-np.inf))
    #     # attention_fake = F.softmax(attention_fake, -1)
    #     # attention_out_fake = attention_fake @ feat  # N, 1, hidden_size
    #     #
    #     # flatten = torch.cat((attention_out, attention_out_fake),-2)
    #     # # flatten = flatten * self.class_weight
    #     # # flatten = torch.sum(flatten, -1)
    #     #
    #     # flatten = self.Linear1(flatten)
    #     # flatten = self.act(flatten)
    #     # flatten = self.Linear2(flatten).squeeze(-1)
    #     # flatten = torch.sigmoid(flatten)
    #
    #
    #     #################
    #
    #     flatten = token_feat
    #     # prob = torch.cat((similarity_fake, similarity), -1)
    #     #
    #     # prob = self.output(prob)
    #
    #
    #     # prob = torch.max(pred[:,:self.num_classes],-1)[0] - 0.5
    #     # prob = self.relu(prob)
    #
    #     return flatten, logit, prob

    def forward(self, ids, token_type_ids, attention_mask, encoded_tag, tag_mask, feat):
        token_feat = self.bert(ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask)[0] #N, L, hidden_size
        sentence_feat = torch.sum(token_feat * attention_mask.unsqueeze(-1), dim=1) \
                        / torch.sum(attention_mask, dim=1, keepdim=True)#N, hidden_size

        embed = self.bert.get_input_embeddings()
        tag_embedding = embed(encoded_tag)
        tag_embedding = torch.sum(tag_embedding * tag_mask.unsqueeze(-1), dim=1) \
                        / torch.sum(tag_mask, dim=1, keepdim=True)
        # print("similarity {}".format(torch.mean(torch.mean(tag_embedding, -1))))

        masks = torch.unsqueeze(attention_mask, 1)  # N, 1, L  .bool()
        attention = (torch.matmul(token_feat, tag_embedding.transpose(0, 1))).transpose(1, 2).masked_fill((1 - masks.byte()), torch.tensor(-np.inf))

        similarity = (torch.matmul(token_feat, tag_embedding.transpose(0, 1))).transpose(1, 2).masked_fill(
                    (1 - masks.byte()), torch.tensor(0))
        # print("similarity {}".format(torch.mean(torch.sum(similarity, -1))))

        attention = F.softmax(attention, -1)

        # print("1 {}".format(torch.max(torch.max(attention,-1)[0],-1)[1]))
        # print("2 {}".format(torch.max(torch.max(attention, -1)[0], -1)[0]))
        attention_out = attention @ token_feat   # N, labels_num, hidden_size
        # print("0 {}".format(torch.sum(token_feat, -1)))
        attention_out = attention_out * self.class_weight
        attention_out = torch.sum(attention_out, -1)
        # print("3 {}".format(torch.max(prob,-1)[1]))
        # print("4 {}".format(torch.max(prob, -1)[0]))

        logit = torch.sigmoid(attention_out)
        # logit = prob

        flatten = torch.sum(attention_out, -1, keepdim=True)

        feat = feat * self.class_weight
        prob = torch.sum(feat, -1)
        # prob = torch.sigmoid(prob)
        prob = torch.sum(prob, -1, keepdim=True)

        prob = torch.cat((prob,flatten),-1)

        prob = self.output(prob)[:,0]


        # prob = attention

        # discrimate = torch.sum(torch.matmul(feat, self.class_weight.transpose(0, 1)), -1, keepdim=True)
        # discrimate = torch.matmul(feat, tag_embedding.transpose(0, 1))
        # pred = torch.sum(logit, -1, keepdim=True)

        # pred = torch.cat((discrimate, logit), -1)

        # attention_out = torch.cat((feat.unsqueeze(1), attention_out), 1)
        # pred = self.Linear1(attention_out)#.squeeze(-1)
        # pred = self.act(pred)
        # pred = self.Linear2(pred).squeeze(-1)
        # pred = torch.sigmoid(pred)
        # logit = pred[:,1:]


        #
        # flatten = torch.mean(attention_out,-2)

        # prob = pred[:,0]

        return flatten, logit, prob

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.class_weight, 'lr': lr},
            {'params': self.bert.parameters(), 'lr': lrp},
            {'params': self.Linear1.parameters(), 'lr': lr},
            {'params': self.Linear2.parameters(), 'lr': lr},
        ]


class Generator(nn.Module):
    def __init__(self, bert, hidden_dim=768, input_dim=768 + 71, num_hidden_generator=2, hidden_dim_generator=2000):
        super(Generator, self).__init__()

        self.dropout = nn.Dropout(p=0.5)
        self.act = nn.LeakyReLU(0.2) #nn.Sigmoid()#

        self.num_hidden_generator = num_hidden_generator
        self.hidden_list_generator = nn.ModuleList()
        for i in range(num_hidden_generator):
            dim = input_dim if i == 0 else hidden_dim_generator
            self.hidden_list_generator.append(nn.Linear(dim, hidden_dim_generator))

        # self.Linear1 = nn.Linear(input_dim, 1500)
        # self.Linear2 = nn.Linear(1500, 3000)
        # self.Linear3 = nn.Linear(3000, 2000)
        self.output = nn.Linear(hidden_dim_generator, hidden_dim)

        self.m1 = nn.BatchNorm1d(2000)

        self.add_module('bert', bert)

        for m in self.bert.parameters():
            m.requires_grad = True

    def forward(self, feat, encoded_tag, tag_mask):

        feat = feat.expand(feat.shape[0], 71,feat.shape[2])
        #
        # embed = self.bert.get_input_embeddings()
        # tag_embedding = embed(encoded_tag)
        # tag_embedding = torch.sum(tag_embedding * tag_mask.unsqueeze(-1), dim=1) \
        #                 / torch.sum(tag_mask, dim=1, keepdim=True)
        # tag_embedding = tag_embedding.detach().unsqueeze(0).expand_as(feat)
        #
        tag_embedding = torch.eye(71).cuda(0).unsqueeze(0).expand(feat.shape[0],71,71)
        x = torch.cat((feat,tag_embedding),-1)
        # x = feat

        for i in range(self.num_hidden_generator):
            x = self.hidden_list_generator[i](x)
            # x = self.m1(x)

            x = self.act(x)
            # x = self.dropout(x)
        # x = self.Linear1(x)
        # x = self.act(x)
        # x = self.Linear2(x)
        # x = self.act(x)
        # x = self.Linear3(x)
        # x = self.act(x)
        y = self.output(x)
        return y

    def get_config_optim(self, lr):
        return [
            {'params': self.hidden_list_generator.parameters(), 'lr': lr},
            {'params': self.output.parameters(), 'lr': lr},
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

        return sentence_feat, token_feat

    def get_config_optim(self, lrp):
        return [
            {'params': self.bert.parameters(), 'lr': lrp},
        ]


class Discriminator(nn.Module):
    def __init__(self, num_classes, input_dim=768, num_hidden_discriminator=1, hidden_dim_discriminator=500):
        super(Discriminator, self).__init__()

        self.dropout = nn.Dropout(p=0.5)
        self.act = nn.LeakyReLU(0.2)#nn.ReLU()

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

    def get_config_optim(self, lr):
        return [
            {'params': self.hidden_list_discriminator.parameters(), 'lr': lr},
            {'params': self.Linear.parameters(), 'lr': lr},
            {'params': self.output.parameters(), 'lr': lr},
        ]
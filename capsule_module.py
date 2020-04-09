import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

USING_LEAKY_ROUTING = True

class CapsuleModule(nn.Module):

    ## 初始化需要配置的参数
    # args.dim_capsule = 16 # 胶囊网络以 向量 取代 标量 , 设定向量的维数(长度)
    # args.in_channels = ? # 输入数据通道数
    # args.num_primary_capsule = ? #初始胶囊层输出的所有胶囊数 , 与输入数据形状相关, 应为 x*32(即PrimaryCaps的out_channels)
    # args.num_compressed_capsule = 128 # 缩减后的胶囊数量
    # args.num_classes = ? # 最终的分类数量
    # args.is_AKDE = True # 选择动态路由策略

    # 使用时
    # __init__:
    #     ...
    #     self.cap = CapsuleModule(args)
    # forward:
    #     data = ...
    #     _,activations = self.cap(data)

    # 我去掉了预测时需要提供 label 的 限制
    # 默认使用所有label去训练
    # 这样的话在训练时需要更多内存

    def __init__(self,args):
        super(CapsuleModule, self).__init__()

        self.primary_capsules_doc = PrimaryCaps(num_capsules=args.dim_capsule, in_channels=args.in_channels, out_channels=32,
                                                kernel_size=1, stride=1)

        self.flatten_capsules = FlattenCaps()
        self.W_doc = nn.Parameter(torch.FloatTensor(args.num_primary_capsule, args.num_compressed_capsule))
        torch.nn.init.xavier_uniform_(self.W_doc)

        self.fc_capsules_doc_child = FCCaps(args, output_capsule_num=args.num_classes,
                                            input_capsule_num=args.num_compressed_capsule,
                                            in_channels=args.dim_capsule, out_channels=args.dim_capsule)

    def compression(self, poses, W):
        poses = torch.matmul(poses.permute(0,2,1), W).permute(0,2,1)
        activations = torch.sqrt((poses ** 2).sum(2))
        return poses, activations

    def forward(self, x):
        # 我不清楚应该加在哪里比较好
        # 这里原来是接在卷积层后的，使用primasy_capsules_doc 代替池化层
        # 把输入的数据转换成胶囊形式
        poses_doc, activations_doc = self.primary_capsules_doc(x) # 把数据转换成胶囊形式

        poses, activations = self.flatten_capsules(poses_doc, activations_doc) # 展开成一层胶囊

        poses, activations = self.compression(poses, self.W_doc)

        poses, activations = self.fc_capsules_doc_child(poses, activations) # 相当于output_layer,输出预测结果

        return poses, activations # poses 是向量, activations 是 poses 的长度, 相当于概率


def squash_v1(x, axis):
    s_squared_norm = (x ** 2).sum(axis, keepdim=True)
    # keepdim 为了保持 输出的 二维特性 而设置为 TRUE
    scale = torch.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # 论文的公式里写的是1 ，为什么这里用了0.5呢？
    # 不过数学意义上看，应是个超参，可以调整
    return scale * x

def dynamic_routing(batch_size, b_ij, u_hat, input_capsule_num):
    num_iterations = 3

    for i in range(num_iterations):
        if USING_LEAKY_ROUTING:
            leak = torch.zeros_like(b_ij).sum(dim=2, keepdim=True)
            leaky_logits = torch.cat((leak, b_ij),2)
            leaky_routing = F.softmax(leaky_logits, dim=2)
            c_ij = leaky_routing[:,:,1:,:].unsqueeze(4)
        else:
            c_ij = F.softmax(b_ij, dim=2).unsqueeze(4)
        v_j = squash_v1((c_ij * u_hat).sum(dim=1, keepdim=True), axis=3)
        if i < num_iterations - 1:
            b_ij = b_ij + (torch.cat([v_j] * input_capsule_num, dim=1) * u_hat).sum(3)

    poses = v_j.squeeze(1)
    activations = torch.sqrt((poses ** 2).sum(2))
    return poses, activations


def Adaptive_KDE_routing(batch_size, b_ij, u_hat):
    last_loss = 0.0
    while True:
        if USING_LEAKY_ROUTING:
            leak = torch.zeros_like(b_ij).sum(dim=2, keepdim=True)
            leaky_logits = torch.cat((leak, b_ij),2)
            leaky_routing = F.softmax(leaky_logits, dim=2)
            c_ij = leaky_routing[:,:,1:,:].unsqueeze(4)
        else:
            c_ij = F.softmax(b_ij, dim=2).unsqueeze(4)
        c_ij = c_ij/c_ij.sum(dim=1, keepdim=True)
        v_j = squash_v1((c_ij * u_hat).sum(dim=1, keepdim=True), axis=3)
        dd = 1 - ((squash_v1(u_hat, axis=3)-v_j)** 2).sum(3)
        b_ij = b_ij + dd

        c_ij = c_ij.view(batch_size, c_ij.size(1), c_ij.size(2))
        dd = dd.view(batch_size, dd.size(1), dd.size(2))

        kde_loss = torch.mul(c_ij, dd).sum()/batch_size
        kde_loss = np.log(kde_loss.item())

        if abs(kde_loss - last_loss) < 0.05:
            break
        else:
            last_loss = kde_loss
    poses = v_j.squeeze(1)
    activations = torch.sqrt((poses ** 2).sum(2))
    return poses, activations


def KDE_routing(batch_size, b_ij, u_hat):
    num_iterations = 3
    for i in range(num_iterations):
        if USING_LEAKY_ROUTING:
            leak = torch.zeros_like(b_ij).sum(dim=2, keepdim=True)
            leaky_logits = torch.cat((leak, b_ij),2)
            leaky_routing = F.softmax(leaky_logits, dim=2)
            c_ij = leaky_routing[:,:,1:,:].unsqueeze(4)
        else:
            c_ij = F.softmax(b_ij, dim=2).unsqueeze(4)

        c_ij = c_ij/c_ij.sum(dim=1, keepdim=True)
        v_j = squash_v1((c_ij * u_hat).sum(dim=1, keepdim=True), axis=3)

        if i < num_iterations - 1:
            dd = 1 - ((squash_v1(u_hat, axis=3)-v_j)** 2).sum(3)
            b_ij = b_ij + dd
    poses = v_j.squeeze(1)
    activations = torch.sqrt((poses ** 2).sum(2))
    return poses, activations

class FlattenCaps(nn.Module):
    def __init__(self):
        super(FlattenCaps, self).__init__()
    def forward(self, p, a):
        poses = p.view(p.size(0), p.size(2) * p.size(3) * p.size(4), -1)
        activations = a.view(a.size(0), a.size(1) * a.size(2) * a.size(3), -1)
        return poses, activations

class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.Conv1d(in_channels, out_channels * num_capsules, kernel_size, stride)

        torch.nn.init.xavier_uniform_(self.capsules.weight)

        self.out_channels = out_channels
        self.num_capsules = num_capsules

    def forward(self, x):
        batch_size = x.size(0)
        u = self.capsules(x).view(batch_size, self.num_capsules, self.out_channels, -1, 1)
        poses = squash_v1(u, axis=1)
        activations = torch.sqrt((poses ** 2).sum(1))
        return poses, activations

class FCCaps(nn.Module):
    def __init__(self, args, output_capsule_num, input_capsule_num, in_channels, out_channels):
        super(FCCaps, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_capsule_num = input_capsule_num
        self.output_capsule_num = output_capsule_num

        self.W1 = nn.Parameter(torch.FloatTensor(1, input_capsule_num, output_capsule_num, out_channels, in_channels))
        torch.nn.init.xavier_uniform_(self.W1)

        self.is_AKDE = args.is_AKDE
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        batch_size = x.size(0)
        variable_output_capsule_num = self.output_capsule_num
        W1 = self.W1

        x = torch.stack([x] * variable_output_capsule_num, dim=2).unsqueeze(4)

        W1 = W1.repeat(batch_size, 1, 1, 1, 1)
        u_hat = torch.matmul(W1, x)

        b_ij = Variable(torch.zeros(batch_size, self.input_capsule_num, variable_output_capsule_num, 1)).cuda()

        if self.is_AKDE == True:
            poses, activations = Adaptive_KDE_routing(batch_size, b_ij, u_hat)
        else:
            #poses, activations = dynamic_routing(batch_size, b_ij, u_hat, self.input_capsule_num)
            poses, activations = KDE_routing(batch_size, b_ij, u_hat)
        return poses, activations


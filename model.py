import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoreFusion(nn.Module):
    def __init__(self, opt):
        super(CoreFusion, self).__init__()
        self.opt = opt
        self.class_num = 2
        self.linear_v = nn.Linear(self.opt['fusion']['dim_v'], self.opt['fusion']['dim_hv'])#linear_v线性层4096，1024
        self.linear_g = nn.Linear(self.opt['fusion']['dim_v'], self.opt['fusion']['dim_hv'])  # linear_v线性层4096，1024
        self.linear_r = nn.Linear(self.opt['fusion']['dim_v'], self.opt['fusion']['dim_hv'])  # linear_v线性层4096，1024

        self.linear_t = nn.Linear(self.opt['fusion']['dim_t'], self.opt['fusion']['dim_ht'])#linear_t线性层0，1024
        self.bn_layer_v = nn.BatchNorm1d(self.opt['fusion']['dim_hv'], affine=True, track_running_stats=True)#bn_layer_v，对1024维进行归一化
        self.bn_layer_t = nn.BatchNorm1d(self.opt['fusion']['dim_ht'], affine=True, track_running_stats=True)#bn_layer_t，对1024维进行归一化

        self.bn_layer_g = nn.BatchNorm1d(self.opt['fusion']['dim_hv'], affine=True, track_running_stats=True)  # bn_layer_v，对1024维进行归一化
        self.bn_layer_r = nn.BatchNorm1d(self.opt['fusion']['dim_hv'], affine=True, track_running_stats=True)  # bn_layer_v，对1024维进行归一化


        self.list_linear_hv = nn.ModuleList([
            nn.Linear(self.opt['fusion']['dim_hv'], self.opt['fusion']['dim_mm'])
            for i in range(self.opt['fusion']['R'])])#list_linear_hv，堆叠两层，两层都是1024，1024线性层
        self.list_linear_hg = nn.ModuleList([
            nn.Linear(self.opt['fusion']['dim_hv'], self.opt['fusion']['dim_mm'])
            for i in range(self.opt['fusion']['R'])])  # list_linear_hv，堆叠两层，两层都是1024，1024线性层
        self.list_linear_hr = nn.ModuleList([
            nn.Linear(self.opt['fusion']['dim_hv'], self.opt['fusion']['dim_mm'])
            for i in range(self.opt['fusion']['R'])])  # list_linear_hv，堆叠两层，两层都是1024，1024线性层

        self.list_linear_ht = nn.ModuleList([
            nn.Linear(self.opt['fusion']['dim_ht'], self.opt['fusion']['dim_mm'])
            for i in range(self.opt['fusion']['R'])])#list_linear_ht，堆叠两层，两层都是1024，1024线性层

        self.dist_learning_v = nn.Linear(self.opt['fusion']['dim_hv'], self.opt['fusion']['dim_hv'])#dist_learning_v线性层，1024，1024
        self.dist_learning_t = nn.Linear(self.opt['fusion']['dim_ht'], self.opt['fusion']['dim_ht'])#dist_learning_t线性层，1024，1024
        self.dist_learning_g = nn.Linear(self.opt['fusion']['dim_hv'], self.opt['fusion']['dim_hv'])  # dist_learning_v线性层，1024，1024
        self.dist_learning_r = nn.Linear(self.opt['fusion']['dim_hv'], self.opt['fusion']['dim_hv'])  # dist_learning_v线性层，1024，1024


        self.fake_ln1 = nn.Linear(self.opt['fusion']['dim_mm'], self.opt['fake_dec']['hidden1'])#fake_ln1线性层，1024，64
        self.fake_last = nn.Linear(self.opt['fake_dec']['hidden1'], self.class_num)#fake_last线性层，64，2

        self.bn_layer1 = nn.BatchNorm1d(self.opt['fake_dec']['hidden1'], affine=True, track_running_stats=True)#bn_layer1，对64维进行归一化
        self.bn_layer4 = nn.BatchNorm1d(self.class_num, affine=True, track_running_stats=True)#bn_layer4，对2维进行归一化

        self.linear_calculsim = nn.Linear(self.opt['fusion']['dim_mm'], 1)#linear_calculsim,1024,1维线性层

#x它的形状应该是 (batch_size_v, batch_size_t, embedding_dim)，其中 batch_size_v 和
#batch_size_t 分别代表两个不同数据集的批次大小，embedding_dim 是嵌入向量的维度'''

    def _calculsim(self, x):#是 CoreFusion 类的一个辅助函数，用于计算输入数据 x=torch.Size([256, 256, 1024]的相似度分数
        batch_size_v = x.size(0)#获取输入张量 x 的第一个维度的大小bs
        batch_size_t = x.size(1)#获取输入张量 x 的第二个维度的大小bs

        x = F.dropout(x,
                      p=self.opt['classif']['dropout'],
                      training=self.training)#p=0
        x = self.linear_calculsim(x)#1024,1线性层
        x = torch.sigmoid(x)#将值压缩到 0 和 1 之间，使得输出可以被解释为概率值
        x = x.view(batch_size_v, batch_size_t)#将输出张量 x 重塑为 (batch_size_v, batch_size_t) 的形状，这样每个元素代表对应的相似度分数
        return x

    def forward(self, input_v, input_t, input_g, input_r):#输入图像文本
        batch_size_v = input_v.size(0)
        batch_size_t = input_t.size(0)# 分别获取图片和文本输入数据的批次大小

        input_v = torch.relu(input_v)
        x_v = self.linear_v(input_v)#linear_v线性层4096，1024，图片成bs，1024
        x_v = self.bn_layer_v(x_v)#归一化
        x_v = F.relu(x_v)

        input_g = torch.relu(input_g)
        x_g = self.linear_g(input_g)  # linear_v线性层4096，1024，图片成bs，1024
        x_g = self.bn_layer_g(x_g)  # 归一化
        x_g = F.relu(x_g)

        input_r = torch.relu(input_r)
        x_r = self.linear_r(input_r)  # linear_v线性层4096，1024，图片成bs，1024
        x_r = self.bn_layer_r(x_r)  # 归一化
        x_r = F.relu(x_r)

        x_t = self.linear_t(input_t)#bs,38400映射到bs,1024
        x_t = self.bn_layer_t(x_t)
        x_t = F.relu(x_t)#归一化和激活函数

        x_dl_v = self.dist_learning_v(x_v)#图片线性层1024，1024
        x_dl_t = self.dist_learning_t(x_t)#文本线性层1024，1024
        x_dl_g = self.dist_learning_g(x_g)  # 文本线性层1024，1024
        x_dl_r = self.dist_learning_r(x_r)  # 文本线性层1024，1024

        x_mm = []#创建空列表x_mm
        x_gg = []
        x_rr = []
        for i in range(self.opt['fusion']['R']):#R: 2

            x_hv = F.dropout(x_v, p=self.opt['fusion']['dropout_hv'], training=self.training)#training=self.training 指示 Dropout 是否处于训练模式，如果是训练模式，则会随机置零；如果是评估模式，则不会应用 Dropout
            x_hv = self.list_linear_hv[i](x_hv)#堆叠两层的1024，1024线性层，输出还是bs,1024

            x_hg = F.dropout(x_g, p=self.opt['fusion']['dropout_hv'], training=self.training)  # training=self.training 指示 Dropout 是否处于训练模式，如果是训练模式，则会随机置零；如果是评估模式，则不会应用 Dropout
            x_hg = self.list_linear_hg[i](x_hg)  # 堆叠两层的1024，1024线性层，输出还是bs,1024
            x_hr = F.dropout(x_r, p=self.opt['fusion']['dropout_hv'], training=self.training)  # training=self.training 指示 Dropout 是否处于训练模式，如果是训练模式，则会随机置零；如果是评估模式，则不会应用 Dropout
            x_hr = self.list_linear_hr[i](x_hr)  # 堆叠两层的1024，1024线性层，输出还是bs,1024

            x_ht = F.dropout(x_t, p=self.opt['fusion']['dropout_ht'], training=self.training)
            x_ht = self.list_linear_ht[i](x_ht)#输出还是bs,1024

            x_mm.append(torch.mul(x_hv[:, None, :], x_ht[None, :, :]))#特征融合,计算 x_hv 和 x_ht 的逐元素乘积
            x_gg.append(torch.mul(x_hv[:, None, :], x_hg[None, :, :]))
            x_rr.append(torch.mul(x_hv[:, None, :], x_hr[None, :, :]))



        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size_v, batch_size_t, self.opt['fusion']['dim_mm'])#x_mm shape: torch.Size([256, 256, 1024])
        x_gg = torch.stack(x_gg, dim=1)
        x_gg = x_gg.sum(1).view(batch_size_v, batch_size_t, self.opt['fusion']['dim_mm'])  # x_mm shape: torch.Size([256, 256, 1024])
        x_rr = torch.stack(x_rr, dim=1)
        x_rr = x_rr.sum(1).view(batch_size_v, batch_size_t, self.opt['fusion']['dim_mm'])  # x_mm shape: torch.Size([256, 256, 1024])


        pairs_num = min(batch_size_v, batch_size_t)#bs
        x_mm_diagonal = x_mm[torch.arange(pairs_num), torch.arange(pairs_num), :]#从融合特征张量 x_mm 中提取对角线元素,x_mm_diagonal: torch.Size([256, 1024])
        #x_mm_diagonal 提取了 x_mm 中所有满足第一个和第二个索引相同的元素，即 x_mm[0, 0, :], x_mm[1, 1, :]...
        x_gg_diagonal = x_gg[torch.arange(pairs_num), torch.arange(pairs_num), :]  # 从融合特征张量 x_mm 中提取对角线元素,x_mm_diagonal: torch.Size([256, 1024])
        x_rr_diagonal = x_rr[torch.arange(pairs_num), torch.arange(pairs_num), :]  # 从融合特征张量 x_mm 中提取对角线元素,x_mm_diagonal: torch.Size([256, 1024])

        fake_res = self.fake_ln1(x_mm_diagonal)##fake_ln1线性层，1024，64,输出bs,64
        fake_res = self.bn_layer1(fake_res)#归一化
        fake_res = F.relu(fake_res)

        fake_res = self.fake_last(fake_res)#线性层64，2，输出bs,2
        fake_res = self.bn_layer4(fake_res)
        fake_res = F.softmax(fake_res, dim=1)#输出融合特征经过softmax的结果，即预测值，用于交叉熵损失计算

        sim_m = self._calculsim(x_mm)#计算相似度，输入bs,bs,1024，输出bs,bs，即成对图片和文本之间的相似度值
        sim_g = self._calculsim(x_gg)  # 计算相似度，输入bs,bs,1024，输出bs,bs，即成对图片和文本之间的相似度值
        sim_r = self._calculsim(x_rr)  # 计算相似度，输入bs,bs,1024，输出bs,bs，即成对图片和文本之间的相似度值


        return sim_m, sim_g, sim_r, x_v, x_t, x_g, x_r, x_dl_v, x_dl_t, x_dl_g, x_dl_r, x_mm_diagonal, x_gg_diagonal, x_rr_diagonal, fake_res
    #分别返回成对图文间相似度（bs,bs），图片表示（bs,1024）,文本表示（bs,1024），再经线性层的图片表示（bs,1024），再经线性层的文本表示（bs,1024），图文融合特征（bs,1024），预测值（bs,2）

def factory(opt, cuda=True):
    opt = copy.copy(opt)
    model = CoreFusion(opt)#我感觉欠拟合，让模型再复杂一些
    if cuda:
        model = model.cuda()
    return model#总的来说，模型就是一个融合，模型的输入是图片和文本嵌入

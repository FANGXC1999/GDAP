from random import random
import torch
import torch.nn as nn
import random
import torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

manualseed = 64
random.seed(manualseed)
np.random.seed(manualseed)
torch.manual_seed(manualseed)
torch.cuda.manual_seed(manualseed)
cudnn.deterministic = True


class UnimodalDetection(nn.Module):
    def __init__(self, shared_dim=256, prime_dim=64):
        super(UnimodalDetection, self).__init__()

        self.text_uni = nn.Sequential(
            nn.Linear(1280, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU())

        self.image_uni = nn.Sequential(
            nn.Linear(1512, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU())

    def forward(self, text_encoding, image_encoding):
        text_prime = self.text_uni(text_encoding)
        image_prime = self.image_uni(image_encoding)
        return text_prime, image_prime


class CrossModule(nn.Module):  # 输入是clip编码的图文，维度都是bs,512
    def __init__(
            self,
            corre_out_dim=64):
        super(CrossModule, self).__init__()
        self.corre_dim = 1024
        self.c_specific = nn.Sequential(
            nn.Linear(self.corre_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        correlation = torch.cat((text, image), 1)  # correlation是bs,1024

        correlation_out = self.c_specific(correlation.float())
        return correlation_out

class CrossShallowModule(nn.Module):  # 输入是clip编码的图文，维度都是bs,512
    def __init__(
            self,
            corre_out_dim=64):
        super(CrossShallowModule, self).__init__()
        self.corre_dim = 1768
        self.c_specific = nn.Sequential(
            nn.Linear(self.corre_dim, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        correlation = torch.cat((text, image), 1)  # correlation是bs,1024,文本是bert，图片是resnet

        correlation_out = self.c_specific(correlation.float())
        return correlation_out

class MultiModal(nn.Module):
    def __init__(
            self,
            feature_dim=64,
            h_dim=64
    ):
        super(MultiModal, self).__init__()
        self.weights = nn.Parameter(torch.rand(13, 1))
        # SENET
        self.senet = nn.Sequential(
            nn.Linear(3, 3),
            nn.GELU(),
            nn.Linear(3, 3),
        )
        self.sigmoid = nn.Sigmoid()

        self.w = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))

        self.avepooling = nn.AvgPool1d(64, stride=1)
        self.maxpooling = nn.MaxPool1d(64, stride=1)

        self.resnet101 = torchvision.models.resnet101(pretrained=True).cuda()

        self.uni_repre = UnimodalDetection()
        self.cross_module = CrossModule()
        self.CrossShallowModule = CrossShallowModule()
        self.classifier_corre = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 2)
        )
        self.deep_corre = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU()
        )
        self.list_linear_hv = nn.ModuleList([
            nn.Linear(64, 64)
            for i in range(2)])#list_linear_hv
        self.list_linear_hg = nn.ModuleList([
            nn.Linear(64, 64)
            for i in range(2)])  # list_linear_hv
        self.list_linear_hr = nn.ModuleList([
            nn.Linear(64, 64)
            for i in range(2)])  # list_linear_hv
        self.list_linear_ht = nn.ModuleList([
            nn.Linear(64, 64)
            for i in range(2)])  # list_linear_ht

        self.image_uni = nn.Sequential(
            nn.Linear(1512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU())
        self.linear_calculsim = nn.Linear(64, 1)  # linear_calculsim,64,1维线性层

    def _calculsim(self, x):  # 是 CoreFusion 类的一个辅助函数，用于计算输入数据 x=torch.Size([256, 256, 1024]的相似度分数
        batch_size_v = x.size(0)  # 获取输入张量 x 的第一个维度的大小bs
        batch_size_t = x.size(1)  # 获取输入张量 x 的第二个维度的大小bs

        x = F.dropout(x, p=0, training=self.training)  # p=0
        x = self.linear_calculsim(x)  # 1024,1线性层
        x = torch.sigmoid(x)  # 将值压缩到 0 和 1 之间，使得输出可以被解释为概率值
        x = x.view(batch_size_v, batch_size_t)  # 将输出张量 x 重塑为 (batch_size_v, batch_size_t) 的形状，这样每个元素代表对应的相似度分数
        return x

    def forward(self, input_ids, all_hidden_states, image_raw, text, image, image_generation, image_relative):  # 模型的输入是Bert的最后的隐藏层和所有隐藏层，原图，clip的图文表征
        # 输入bs,300,768  bert的13层的元组 原图bs,3,224,224 clip编码的图文表示为bs,512
        # Process image
        #image_raw = self.resnet101(image_raw)  # 编码为bs，1000
        batch_size_v = image.size(0)
        batch_size_t = text.size(0)


        # Process text
        ht_cls = all_hidden_states#torch.Size([13, 32, 768])
        ht_cls = torch.unsqueeze(ht_cls, dim=2)
        #ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(13, input_ids.shape[0], 1, 768)  # 维度变化为13,64,1,768
        atten = torch.sum(ht_cls * self.weights.view(13, 1, 1, 1), dim=[1, 3])#torch.Size([13, 1])
        atten = F.softmax(atten.view(-1), dim=0)#torch.Size([13])
        text_raw = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])  # 变为bs,768

        # Unimodal processing
        text_prime, image_prime = self.uni_repre(torch.cat([text_raw, text], 1),
                                                 torch.cat([image_raw, image], 1))  # 图片和文本都编码为bs,64
        image_g_prime = self.image_uni(torch.cat([image_generation, image], 1))
        image_r_prime = self.image_uni(torch.cat([image_relative, image], 1))

        x_mm = []#创建空列表x_mm
        x_gg = []
        x_rr = []
        for i in range(2):#R: 2
            image_prime = F.dropout(image_prime, p=0.4, training=self.training)#training=self.training 指示 Dropout 是否处于训练模式，如果是训练模式，则会随机置零；如果是评估模式，则不会应用 Dropout
            image_prime = self.list_linear_hv[i](image_prime)#堆叠两层的1024，1024线性层，输出还是bs,1024
            image_g_prime = F.dropout(image_g_prime, p=0.4, training=self.training)  # training=self.training 指示 Dropout 是否处于训练模式，如果是训练模式，则会随机置零；如果是评估模式，则不会应用 Dropout
            image_g_prime = self.list_linear_hg[i](image_g_prime)  # 堆叠两层的1024，1024线性层，输出还是bs,1024
            image_r_prime = F.dropout(image_r_prime, p=0.4, training=self.training)  # training=self.training 指示 Dropout 是否处于训练模式，如果是训练模式，则会随机置零；如果是评估模式，则不会应用 Dropout
            image_r_prime = self.list_linear_hr[i](image_r_prime)  # 堆叠两层的1024，1024线性层，输出还是bs,1024
            text_prime = F.dropout(text_prime, p=0.4, training=self.training)
            text_prime = self.list_linear_ht[i](text_prime)#输出还是bs,1024
            x_mm.append(torch.mul(image_prime[:, None, :], text_prime[None, :, :]))#特征融合,计算图文的逐元素乘积
            x_gg.append(torch.mul(image_prime[:, None, :], image_g_prime[None, :, :]))#原图和生成图之间的相关性
            x_rr.append(torch.mul(image_prime[:, None, :], image_r_prime[None, :, :]))#原图和相对图之间的相关性
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size_v, batch_size_t, 64)#x_mm shape: torch.Size([bs, bs, 64])
        x_gg = torch.stack(x_gg, dim=1)
        x_gg = x_gg.sum(1).view(batch_size_v, batch_size_t, 64)  # x_mm shape: torch.Size([256, 256, 1024])
        x_rr = torch.stack(x_rr, dim=1)
        x_rr = x_rr.sum(1).view(batch_size_v, batch_size_t, 64)  # x_mm shape: torch.Size([256, 256, 1024])
        sim_m = self._calculsim(x_mm)  # 计算相似度，输入bs,bs,1024，输出bs,bs，即成对图片和文本之间的相似度值
        sim_g = self._calculsim(x_gg)  # 计算相似度，输入bs,bs,1024，输出bs,bs，原图和生成图之间的相关性
        sim_r = self._calculsim(x_rr)  # 计算相似度，输入bs,bs,1024，输出bs,bs，原图和相对图之间的相关性

        # Cross-modal processing $ Visualize shallow features
        correlation = self.cross_module(text, image)  # 输出为bs,64
        correlation_shallow1 = self.CrossShallowModule(text_raw, image_generation)
        correlation_shallow2 = self.CrossShallowModule(text_raw, image_relative)

        # Calculate similarity weights
        sim = torch.div(torch.sum(text * image, 1),
                        torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(torch.sum(torch.pow(image, 2), 1)))
        # 计算 text 和 image 张量逐元素相乘的结果，然后沿着维度 1（通常是特征维度）求和。 整个公式相当于计算余弦相似度
        sim = sim * self.w + self.b  # 输出bs,
        mweight = sim.unsqueeze(
            1)  # 将其变为形状为 (batch_size, 1) 的二维张量。 mweight 是调整后的相似度分数，现在具有额外的维度，可以直接用作广播（broadcasting）操作的权重

        # Apply correlation weights
        correlation = correlation * mweight  # bs，64

        # Combine features
        final_feature = torch.cat([text_prime.unsqueeze(1), image_prime.unsqueeze(1), correlation.unsqueeze(1)],
                                  1)  # 输出bs,3,64

        # Pooling and transformation
        s1 = self.avepooling(final_feature)  # 输出torch.Size([bs, 3, 1])
        s2 = self.maxpooling(final_feature)  # 输出torch.Size([bs, 3, 1])
        s1 = s1.view(s1.size(0), -1)#torch.Size([32, 3])
        s2 = s2.view(s2.size(0), -1)#torch.Size([32, 3])
        s1 = self.senet(s1)
        s2 = self.senet(s2)
        s = self.sigmoid(s1 + s2)
        s = s.view(s.size(0), s.size(1), 1)  # 这个操作将张量bs,3 重塑为一个三维张量bs,3,1

        # Apply pooling weights
        final_feature = s * final_feature  # 输出bs,3,64
        feature_deep1 = final_feature[:, 0, :]#text_prime
        feature_deep2 = final_feature[:, 1, :]#image_prime
        feature_deep3 = final_feature[:, 2, :]#correlation
        feature_deep = self.deep_corre(
            final_feature[:, 0, :] + final_feature[:, 1, :] + final_feature[:, 2, :])  # 经过线性层、bn、relu得到bs，64.用于深层特征的可视化展示


        # Classification
        pre_label = self.classifier_corre(
            final_feature[:, 0, :] + final_feature[:, 1, :] + final_feature[:, 2, :])  # 经过线性层、bn、relu得到bs，2

        return pre_label, sim_m, sim_g, sim_r, correlation, correlation_shallow1, correlation_shallow2, feature_deep1, feature_deep2, feature_deep3, feature_deep
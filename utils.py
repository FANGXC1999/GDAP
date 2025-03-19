import torch
import numpy as np


def pairwise_loss_with_label(score, labels):
    size_v = score.size(0)
    size_t = score.size(1)
    diagonal = score.diag().view(size_v, 1)#.diag()函数用于从一个方阵中提取对角线元素,bs,1
    #print(diagonal.shape)
    d1 = diagonal.expand_as(score)#expand_as() 方法用于将 diagonal 张量扩展到与另一个张量 score 相同的形状。
    d2 = diagonal.t().expand_as(score)

    # compare every diagonal score to scores in its column   将每个对角线分数与其列中的分数进行比较
    cost_s = (0.5 + score - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    cost_im = (0.5 + score - d2).clamp(min=0)

    mask = torch.eye(score.size(0)) > .5#torch.eye(score.size(0)) > .5这个表达式将单位矩阵中的每个元素与0.5进行比较。因此，这个比较操作将返回一个布尔型张量，其中对角线元素为 True，其余元素为 False
    if torch.cuda.is_available():
        mask = mask.cuda()
    cost_s = cost_s.masked_fill_(mask, 0)#这个方法将 cost_s 中所有对应于 mask 中 True 的位置的元素替换为0。其他位置的元素保持不变
    cost_im = cost_im.masked_fill_(mask, 0)

    #labels_expand = labels[:, 0].expand(size_v, size_v)
    labels_expand = labels.expand(size_v, size_v)
    #labels[:, 0]这个操作选择了 labels 张量的所有行（:）和第一列（0），结果是一个一维张量，包含了每个样本的第一个标签值。.expand() 方法用于创建一个新的张量视图，其形状由给定的维度大小确定
    label_mask = labels_expand ^ labels_expand.T#label_mask 是一个整数型张量，其中元素值为0或1，0表示对应位置的标签相同，1表示标签不同。
    label_mask = label_mask < 1#将 label_mask 中的每个元素与1进行比较。 label_mask < 1 的结果是将0转换为 True，将1转换为 False。即将标签相同位置变为true

    cost_s = cost_s.masked_fill_(label_mask, 0)#每一行只剩下不同标签的值，值表示为相似性的差值
    cost_im = cost_im.masked_fill_(label_mask, 0)

    cost_s = cost_s.max(1)[0]#.max(1) 表示沿着 cost_s 的第二个维度（即每一行）计算最大值。这将返回一个包含最大值和对应索引的元组。因此，cost_s.max(1)[0] 将返回一个一维张量，其中包含了 cost_s 每一行的最大值。
    cost_im = cost_im.max(0)[0]#.max(0) 表示沿着 cost_im 的第一个维度（即每一列）计算最大值。这将返回一个包含最大值和对应索引的元组。因此，cost_im.max(0)[0] 将返回一个一维张量，其中包含了 cost_im 每一列的最大值。

    return (cost_s.sum() + cost_im.sum()) * 0.5

def pairwise_loss1_with_label(score, labels):
    size_v = score.size(0)
    size_t = score.size(1)
    diagonal = score.diag().view(size_v, 1)
    cost = (0.5 + diagonal).clamp(min=0)
    return cost.sum() * 0.5



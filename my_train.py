import os  # 用于操作系统功能，如设置环境变量
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor

import torch
import numpy as np  # 用于数学运算
from sklearn.metrics import accuracy_score, confusion_matrix  # 用于计算准确率和混淆矩阵
from transformers import logging, BertConfig  # 用于加载预训练的BERT模型
from transformers import BertModel, BertTokenizer
from torch.autograd import Variable
from tqdm import tqdm  # 用于显示进度条
import torch.nn.functional as F
from weibo_dataset import *
import argparse
import yaml
import torch.nn as nn
from pytorchtools import EarlyStopping
import model
import my_models
import data_prase
import engine
import my_engine
import random
from my_network import MultiModal
#from network import MultiModal
from sklearn.metrics import precision_score, recall_score, f1_score
import utils

# Set logging verbosity for transformers library
# logging.set_verbosity_warning()#设置transformers库的日志级别
# logging.set_verbosity_error()


# Set CUDA device if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BERT model
model_name = 'bert-base-chinese'
# 对于 BertModel，如果你没有设置 output_hidden_states=True，那么模型默认只返回最后一层的隐藏状态（last_hidden_state）。如果设置了 output_hidden_states=True，则会返回一个包含多个元素的元组，其中第一个元素是 last_hidden_state。
tokenizer = BertTokenizer.from_pretrained(model_name)
model_config = BertConfig.from_pretrained(model_name)
model_config.output_hidden_states = True
BERT_model = BertModel.from_pretrained(model_name, config=model_config).cuda()


# Freeze the parameters of the BERT model
for param in BERT_model.parameters():
    param.requires_grad = False  # BERT模型的参数在训练过程中不会被更新


def train():
    #batch_size = 64  # 设置批量大小、学习率和L2正则化系数
    #lr = 1e-3#1e-3为0.001
    lr = 0.0005
    l2 = 0
    print('1.load data..............')
    train_loader, test_loader = data_prase.get_loaders('./my_data', 64)

    # Initialize the MultiModal network
    print('2.build the model........')
    rumor_module = MultiModal()  # 初始化MultiModal网络并将其移动到GPU
    rumor_module.to(device)

    # Define the loss function for rumor classification
    loss_f_rumor = torch.nn.CrossEntropyLoss()  # 定义损失函数，同时声明自己设计的损失函数

    # Define the optimizer
    optim_task = torch.optim.Adam(
        rumor_module.parameters(), lr=lr, weight_decay=l2)  # 定义优化器,weight_decay是超参数，可以改

    # Training loop
    print('3.start training.........')
    for epoch in range(200):

        rumor_module.train()
        corrects_pre_rumor = 0
        loss_total = 0
        rumor_count = 0

        for step, train_data in tqdm(enumerate(train_loader)):#加载bs数据
            input_image, image_clip_embs, image_generated_embs, image_comparison_embs, image_diff_absolute_embs, image_diff_relative_embs, input_token, text_clip_embs, labels = train_data
            #input_image, image_clip_embs, image_generated_embs, image_comparison_embs, image_diff_absolute_embs, image_diff_relative_embs, text_bert_token, text_clip_embs, labels = input_image.to(device), image_clip_embs.to(device), image_generated_embs.to(device), image_comparison_embs.to(device), image_diff_absolute_embs.to(device), image_diff_relative_embs.to(device), text_bert_token.to(device), text_clip_embs.to(device), labels.to(device)
            batch_last_hidden_states = []#将bert token转出来，再用于后面提取充足的bert特征
            batch_all_hidden_states = []
            #batch_attention_mask = []
            for i in range(input_token.size(0)):#因为input_token是bs,1,300维度，所以需要处理
                sample_token_ids = input_token[i].cpu().numpy()
                sample_token_ids = sample_token_ids.astype(int)

                sample_tokens = [tokenizer.convert_ids_to_tokens(token_id) for token_id in sample_token_ids]
                input_ids = [tokenizer.convert_tokens_to_ids(i) for i in sample_tokens]
                input_ids = torch.tensor(input_ids)
                input_ids = torch.LongTensor(input_ids).to(device)
                outputs = BERT_model(input_ids)
                batch_last_hidden_states.append(outputs[0])
                batch_all_hidden_states.append(outputs[2])
            #print('batch_features', batch_features)#batch_features分为3部分，分别是1，300，768。1，768。13的tuple，里面是13个1，300，768的特征
            #print('batch_features shape',batch_features.shape)

            batch_last_hidden_states = torch.stack(batch_last_hidden_states)
            batch_last_hidden_states = torch.squeeze(batch_last_hidden_states, dim=1)

            batch_all_hidden_states = torch.stack([torch.cat(tuple(item), dim=0) for item in batch_all_hidden_states])
            batch_all_hidden_states = batch_all_hidden_states.permute(1, 0, 2, 3)  # 将维度从bs,13,300,768到13,bs,300,768
            batch_all_hidden_states = torch.mean(batch_all_hidden_states, dim=2)#第3维度求最平均，得到13,bs,1,768
            #batch_all_hidden_states, _ = torch.max(tensor, dim=2)#求最大，得到13,bs,1,768

            # again tokenize text data using BERT tokenizer
            #print('batch_tokens', batch_tokens.shape)#batch_tokens是list
            # Forward pass through the MultiModal network.

            #batch_all_hidden_states = torch.squeeze(batch_all_hidden_states, dim=1)
            input_image = torch.squeeze(input_image, dim=1)
            image_generated_embs = torch.squeeze(image_generated_embs, dim=1)
            image_diff_relative_embs = torch.squeeze(image_diff_relative_embs, dim=1)
            text_clip_embs = torch.squeeze(text_clip_embs, dim=1)
            image_clip_embs = torch.squeeze(image_clip_embs, dim=1)
            label = labels.long()
            #label = labels[:, 0]  # 选择所有行（:）和第一列（0）
            pre_rumor, scores, scores_g, scores_r, _, _, _, _, _, _, _ = rumor_module(batch_last_hidden_states, batch_all_hidden_states, input_image, text_clip_embs,
                                     image_clip_embs, image_generated_embs, image_diff_relative_embs)  # 模型的输入是Bert的最后的隐藏层和所有隐藏层，原图，clip的图文表征。输出bs，2
            loss_rumor = loss_f_rumor(pre_rumor, label)  # 损失就一个交叉熵损失
            cross_loss = utils.pairwise_loss_with_label(scores, label) / input_image.size(0)  # 文章提到的成对损失，自定义的损失
            cross_loss2 = utils.pairwise_loss1_with_label(scores_g, label) / input_image.size(0)
            cross_loss3 = utils.pairwise_loss1_with_label(scores_r, label) / input_image.size(0)
            loss_all = 0.2 * cross_loss - 0.5 * cross_loss2 + 0.5 * cross_loss3 + 2 * loss_rumor

            optim_task.zero_grad()
            loss_all.backward()
            optim_task.step()  # 优化器的参数是模型的所有参数

            pre_label_rumor = pre_rumor.argmax(1)  # 从模型的输出中获取预测的标签
            corrects_pre_rumor += pre_label_rumor.eq(label.view_as(pre_label_rumor)).sum().item()#计算在当前批次中模型预测正确的样本数量，并将其累加到 corrects_pre_rumor 变量中
            # 这里，eq函数用于比较预测标签pre_label_rumor和真实标签label是否相等，返回一个布尔类型的张量，形状与pre_label_rumor相同，其中的每个元素都是True或False，表示对应的预测是否正确
            # .sum()：对上述布尔张量中的True值进行求和，因为True在数值上等价于1，False等价于0，所以这个操作实际上是计算正确预测的数量。.item()：将求和结果从张量转换为Python的标量数值，以便可以累加到corrects_pre_rumor变量中
            loss_total += loss_rumor.item() * input_image.shape[0]  # 乘以 bs 即64。 将本轮批量的损失乘以批量大小，然后累加到loss_total变量中。这样做是为了在计算平均损失时考虑到每个样本，而不是仅仅计算总损失。
            rumor_count += input_image.shape[0]  # 所有数据数

        loss_rumor_train = loss_total / rumor_count  # 平均的总损失
        acc_rumor_train = corrects_pre_rumor / rumor_count  # 训练集的准确率

        acc_rumor_test, loss_rumor_test, conf_rumor, precision_rumor_test, recall_rumor_test, f1_rumor_test, precision_rumor_test_neg, recall_rumor_test_neg, f1_rumor_test_neg = test(rumor_module, test_loader)
        print('-----------rumor detection----------------')
        print(
            "EPOCH = %d || acc_rumor_train = %.3f || acc_rumor_test = %.3f ||  loss_rumor_train = %.3f || loss_rumor_test = %.3f || precision_rumor_test = %.3f || recall_rumor_test = %.3f || f1_rumor_test = %.3f || precision_rumor_test_neg = %.3f || recall_rumor_test_neg = %.3f || f1_rumor_test_neg = %.3f" %
            (epoch + 1, acc_rumor_train, acc_rumor_test, loss_rumor_train, loss_rumor_test, precision_rumor_test, recall_rumor_test, f1_rumor_test, precision_rumor_test_neg, recall_rumor_test_neg, f1_rumor_test_neg))

        print('-----------rumor_confusion_matrix---------')
        print(conf_rumor)


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def test(rumor_module, test_loader):
    rumor_module.eval()


    loss_f_rumor = torch.nn.CrossEntropyLoss()

    rumor_count = 0
    loss_total = 0
    precision_test = 0
    recall_test = 0
    f1_test = 0
    rumor_label_all = []
    rumor_pre_label_all = []
    feature_shallow1_matrix = []#存生成图res和文bert相关浅层特征
    feature_shallow2_matrix = []#存差值图res和文bert相关浅层特征
    feature_shallow3_matrix = []#存原图文相关浅层特征
    feature_deep1_matrix = []  #存文本深层特征
    feature_deep2_matrix = []  #存图深层特征
    feature_deep3_matrix = []  #存多模态深层特征
    feature_deep4_matrix = []  #存总体深层特征
    label_matrix = []
    for step, test_data in tqdm(enumerate(test_loader)):  # 加载bs数据
        input_image, image_clip_embs, image_generated_embs, image_comparison_embs, image_diff_absolute_embs, image_diff_relative_embs, input_token, text_clip_embs, labels = test_data
        batch_last_hidden_states = []  # 将bert token转出来，再用于后面提取充足的bert特征
        batch_all_hidden_states = []
        # batch_attention_mask = []
        for i in range(input_token.size(0)):  # 因为input_token是bs,1,300维度，所以需要处理
            sample_token_ids = input_token[i].cpu().numpy()
            sample_token_ids = sample_token_ids.astype(int)

            sample_tokens = [tokenizer.convert_ids_to_tokens(token_id) for token_id in sample_token_ids]
            input_ids = [tokenizer.convert_tokens_to_ids(i) for i in sample_tokens]
            input_ids = torch.tensor(input_ids)
            input_ids = torch.LongTensor(input_ids).to(device)
            outputs = BERT_model(input_ids)
            batch_last_hidden_states.append(outputs[0])
            batch_all_hidden_states.append(outputs[2])
        batch_last_hidden_states = torch.stack(batch_last_hidden_states)
        batch_last_hidden_states = torch.squeeze(batch_last_hidden_states, dim=1)
        batch_all_hidden_states = torch.stack([torch.cat(tuple(item), dim=0) for item in batch_all_hidden_states])
        batch_all_hidden_states = batch_all_hidden_states.permute(1, 0, 2, 3)  # 将维度从bs,13,300,768到13,bs,300,768
        batch_all_hidden_states = torch.mean(batch_all_hidden_states, dim=2)  # 第3维度求最平均，得到13,bs,1,768
        input_image = torch.squeeze(input_image, dim=1)
        image_generated_embs = torch.squeeze(image_generated_embs, dim=1)
        image_diff_relative_embs = torch.squeeze(image_diff_relative_embs, dim=1)
        text_clip_embs = torch.squeeze(text_clip_embs, dim=1)
        image_clip_embs = torch.squeeze(image_clip_embs, dim=1)
        label = labels.long()
        #label = labels[:, 0]  # 选择所有行（:）和第一列（0）
        pre_rumor, _, _, _, feature_shallow1, feature_shallow2, feature_shallow3, feature_deep1, feature_deep2, feature_deep3, feature_deep4 = rumor_module(batch_last_hidden_states, batch_all_hidden_states, input_image, text_clip_embs,
                                 image_clip_embs, image_generated_embs, image_diff_relative_embs)  # 模型的输入是Bert的最后的隐藏层和所有隐藏层，原图，clip的图文表征。输出bs，2
        loss_rumor = loss_f_rumor(pre_rumor, label)
        pre_label_rumor = pre_rumor.argmax(1)
        loss_total += loss_rumor.item() * input_image.shape[0]
        rumor_count += input_image.shape[0]

        # Store predicted and true labels for evaluation
        rumor_pre_label_all.append(pre_label_rumor.detach().cpu().numpy())
        rumor_label_all.append(label.detach().cpu().numpy())

        #shallow, deep feature model
        label_matrix.append(label.cpu().detach().numpy())
        feature_shallow1_matrix.append(feature_shallow1.cpu().detach().numpy())
        feature_shallow2_matrix.append(feature_shallow2.cpu().detach().numpy())
        feature_shallow3_matrix.append(feature_shallow3.cpu().detach().numpy())
        feature_deep1_matrix.append(feature_deep1.cpu().detach().numpy())
        feature_deep2_matrix.append(feature_deep2.cpu().detach().numpy())
        feature_deep3_matrix.append(feature_deep3.cpu().detach().numpy())
        feature_deep4_matrix.append(feature_deep4.cpu().detach().numpy())
    #save shallow, deep feature
    all_label_np = np.concatenate(label_matrix, axis=0)
    all_feature_shallow1_np = np.concatenate(feature_shallow1_matrix, axis=0)
    all_feature_shallow2_np = np.concatenate(feature_shallow2_matrix, axis=0)
    all_feature_shallow3_np = np.concatenate(feature_shallow3_matrix, axis=0)
    all_feature_deep1_np = np.concatenate(feature_deep1_matrix, axis=0)
    all_feature_deep2_np = np.concatenate(feature_deep2_matrix, axis=0)
    all_feature_deep3_np = np.concatenate(feature_deep3_matrix, axis=0)
    all_feature_deep4_np = np.concatenate(feature_deep4_matrix, axis=0)
    matrix_save_dir = './weibo_dataset'
    np.save('{}/label_matrix'.format(matrix_save_dir), all_label_np)
    np.save('{}/feature_shallow1_matrix'.format(matrix_save_dir), all_feature_shallow1_np)
    np.save('{}/feature_shallow2_matrix'.format(matrix_save_dir), all_feature_shallow2_np)
    np.save('{}/feature_shallow3_matrix'.format(matrix_save_dir), all_feature_shallow3_np)
    np.save('{}/feature_deep1_matrix'.format(matrix_save_dir), all_feature_deep1_np)
    np.save('{}/feature_deep2_matrix'.format(matrix_save_dir), all_feature_deep2_np)
    np.save('{}/feature_deep3_matrix'.format(matrix_save_dir), all_feature_deep3_np)
    np.save('{}/feature_deep4_matrix'.format(matrix_save_dir), all_feature_deep4_np)


    # Calculate accuracy, p, r, F1 and confusion matrix
    loss_rumor_test = loss_total / rumor_count
    rumor_pre_label_all = np.concatenate(rumor_pre_label_all, 0)
    rumor_label_all = np.concatenate(rumor_label_all, 0)
    acc_rumor_test = accuracy_score(rumor_pre_label_all, rumor_label_all)  # 直接调用sklearn.metrics计算的准确率
    conf_rumor = confusion_matrix(rumor_pre_label_all, rumor_label_all)  # 有用吗？我感觉没用
    #计算p，r，f1
    precision_rumor_test = precision_score(rumor_pre_label_all, rumor_label_all)
    recall_rumor_test = precision_score(rumor_pre_label_all, rumor_label_all)
    f1_rumor_test = precision_score(rumor_pre_label_all, rumor_label_all)
    precision_rumor_test_neg = precision_score(rumor_pre_label_all, rumor_label_all, pos_label=0, average='binary')
    recall_rumor_test_neg = recall_score(rumor_pre_label_all, rumor_label_all, pos_label=0, average='binary')
    f1_rumor_test_neg = f1_score(rumor_pre_label_all, rumor_label_all, pos_label=0, average='binary')


    return acc_rumor_test, loss_rumor_test, conf_rumor, precision_rumor_test, recall_rumor_test, f1_rumor_test, precision_rumor_test_neg, recall_rumor_test_neg, f1_rumor_test_neg


if __name__ == "__main__":
    train()

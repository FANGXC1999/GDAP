import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import logging, BertConfig
from transformers import BertModel, BertTokenizer
from torch.autograd import Variable
from tqdm import tqdm
import data_prase
from network import MultiModal
from sklearn.metrics import precision_score, recall_score, f1_score
import utils



# Set CUDA device if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BERT model
model_name = './bert-base-chinese'
# 对于 BertModel，如果你没有设置 output_hidden_states=True，那么模型默认只返回最后一层的隐藏状态（last_hidden_state）。如果设置了 output_hidden_states=True，则会返回一个包含多个元素的元组，其中第一个元素是 last_hidden_state。
tokenizer = BertTokenizer.from_pretrained(model_name)
model_config = BertConfig.from_pretrained(model_name)
model_config.output_hidden_states = True
BERT_model = BertModel.from_pretrained(model_name, config=model_config).cuda()


# Freeze the parameters of the BERT model
for param in BERT_model.parameters():
    param.requires_grad = False  # BERT模型的参数在训练过程中不会被更新


def train():
    #设置学习率和L2正则化系数
    lr = 0.001
    l2 = 0
    print('1.load data..............')
    train_loader, test_loader = data_prase.get_loaders('./data', 64)

    # Initialize the MultiModal network
    print('2.build the model........')
    rumor_module = MultiModal()  # 初始化MultiModal网络并将其移动到GPU
    rumor_module.cuda()

    # Define the loss function for fake news classification
    loss_f_rumor = torch.nn.CrossEntropyLoss()
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

        for step, train_data in tqdm(enumerate(train_loader)):
            '''The loaded data refers to the original image res representation, original image clip representation, generated image (i.e. marginal image) representation, 
            comparison image representation, image absolute difference representation, image relative difference representation, input text token (used to obtain all hidden states of BERT), 
            text clip representation, and label. Note: In the implementation process of GDAP, the comparison image representation and the image absolute difference representation of the image are not used.'''
            input_image, image_clip_embs, image_generated_embs, image_comparison_embs, image_diff_absolute_embs, image_diff_relative_embs, input_token, text_clip_embs, labels = train_data#分别是图像、文本、标签表示
            batch_last_hidden_states = []
            batch_all_hidden_states = []
            for i in range(input_token.size(0)):#因为input_token是bs,1,300维度，所以需要处理
                sample_token_ids = input_token[i].cpu().numpy()
                sample_token_ids = sample_token_ids.astype(int)
                sample_tokens = [tokenizer.convert_ids_to_tokens(token_id) for token_id in sample_token_ids]
                input_ids = [tokenizer.convert_tokens_to_ids(i) for i in sample_tokens]
                input_ids = torch.tensor(input_ids)
                input_ids = torch.LongTensor(input_ids).cuda()
                outputs = BERT_model(input_ids)
                batch_last_hidden_states.append(outputs[0])
                batch_all_hidden_states.append(outputs[2])
            batch_last_hidden_states = torch.stack(batch_last_hidden_states)
            batch_last_hidden_states = torch.squeeze(batch_last_hidden_states, dim=1)

            '''Calculate all hidden state outputs of BERT'''
            batch_all_hidden_states = torch.stack([torch.cat(tuple(item), dim=0) for item in batch_all_hidden_states])
            batch_all_hidden_states = batch_all_hidden_states.permute(1, 0, 2, 3)  #Dimension from bs, 13, 300, 768 to 13, bs,300,768
            batch_all_hidden_states = torch.mean(batch_all_hidden_states, dim=2)#The third dimension is averaged to obtain 13, bs, 1, 768

            input_image = torch.squeeze(input_image, dim=1)
            image_generated_embs = torch.squeeze(image_generated_embs, dim=1)
            image_diff_relative_embs = torch.squeeze(image_diff_relative_embs, dim=1)
            text_clip_embs = torch.squeeze(text_clip_embs, dim=1)
            image_clip_embs = torch.squeeze(image_clip_embs, dim=1)
            label = labels.long()

            pre_rumor, scores, scores_g, scores_r, _, _, _, _, _, _, _ = rumor_module(batch_last_hidden_states, batch_all_hidden_states, input_image, text_clip_embs,
                                     image_clip_embs, image_generated_embs, image_diff_relative_embs)
            '''Three loss functions. They are respectively the loss for classification (loss_rumor), the contrastive loss for deep feature modeling (cross_loss), and the contrastive loss for shallow feature modeling (cross_loss2 and cross_loss3).'''
            loss_rumor = loss_f_rumor(pre_rumor, label)
            cross_loss = utils.pairwise_loss_with_label(scores, label) / input_image.size(0)  # 文章提到的成对损失，自定义的损失
            cross_loss2 = utils.pairwise_loss1_with_label(scores_g, label) / input_image.size(0)
            cross_loss3 = utils.pairwise_loss1_with_label(scores_r, label) / input_image.size(0)
            loss_all = 0.2 * cross_loss - 0.5 * cross_loss2 + 0.5 * cross_loss3 + 2 * loss_rumor
            optim_task.zero_grad()
            loss_all.backward()
            optim_task.step()  # 优化器的参数是模型的所有参数
            pre_label_rumor = pre_rumor.argmax(1)#Prediction label
            corrects_pre_rumor += pre_label_rumor.eq(label.view_as(pre_label_rumor)).sum().item()#Calculate the number of samples correctly predicted by the model in the current batch and add it to the corrects_pre_rumor variable
            loss_total += loss_rumor.item() * input_image.shape[0]#Multiply the loss of this batch by the batch size and add it to the loss_total variable. This is done to consider each sample when calculating the average loss, rather than just calculating the total loss.
            rumor_count += input_image.shape[0]  # 所有数据数
        loss_rumor_train = loss_total / rumor_count#Average total loss
        acc_rumor_train = corrects_pre_rumor / rumor_count#Accuracy of the training set

        '''Conduct testing'''
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
    rumor_label_all = []
    rumor_pre_label_all = []
    # Used for visualization below
    #feature_shallow1_matrix = []
    #feature_shallow2_matrix = []
    #feature_shallow3_matrix = []
    #feature_deep1_matrix = []
    #feature_deep2_matrix = []
    #feature_deep3_matrix = []
    #feature_deep4_matrix = []
    #label_matrix = []
    for step, test_data in tqdm(enumerate(test_loader)):
        input_image, image_clip_embs, image_generated_embs, image_comparison_embs, image_diff_absolute_embs, image_diff_relative_embs, input_token, text_clip_embs, labels = test_data
        batch_last_hidden_states = []  # 将bert token转出来，再用于后面提取充足的bert特征
        batch_all_hidden_states = []
        for i in range(input_token.size(0)):
            sample_token_ids = input_token[i].cpu().numpy()
            sample_token_ids = sample_token_ids.astype(int)
            sample_tokens = [tokenizer.convert_ids_to_tokens(token_id) for token_id in sample_token_ids]
            input_ids = [tokenizer.convert_tokens_to_ids(i) for i in sample_tokens]
            input_ids = torch.tensor(input_ids)
            input_ids = torch.LongTensor(input_ids).cuda()
            outputs = BERT_model(input_ids)
            batch_last_hidden_states.append(outputs[0])
            batch_all_hidden_states.append(outputs[2])
        batch_last_hidden_states = torch.stack(batch_last_hidden_states)
        batch_last_hidden_states = torch.squeeze(batch_last_hidden_states, dim=1)
        batch_all_hidden_states = torch.stack([torch.cat(tuple(item), dim=0) for item in batch_all_hidden_states])
        batch_all_hidden_states = batch_all_hidden_states.permute(1, 0, 2, 3)
        batch_all_hidden_states = torch.mean(batch_all_hidden_states, dim=2)
        input_image = torch.squeeze(input_image, dim=1)
        image_generated_embs = torch.squeeze(image_generated_embs, dim=1)
        image_diff_relative_embs = torch.squeeze(image_diff_relative_embs, dim=1)
        text_clip_embs = torch.squeeze(text_clip_embs, dim=1)
        image_clip_embs = torch.squeeze(image_clip_embs, dim=1)
        label = labels.long()

        pre_rumor, _, _, _, feature_shallow1, feature_shallow2, feature_shallow3, feature_deep1, feature_deep2, feature_deep3, feature_deep4 = rumor_module(batch_last_hidden_states, batch_all_hidden_states, input_image, text_clip_embs,
                                 image_clip_embs, image_generated_embs, image_diff_relative_embs)  # 模型的输入是Bert的最后的隐藏层和所有隐藏层，原图，clip的图文表征。输出bs，2
        loss_rumor = loss_f_rumor(pre_rumor, label)
        pre_label_rumor = pre_rumor.argmax(1)
        loss_total += loss_rumor.item() * input_image.shape[0]
        rumor_count += input_image.shape[0]

        rumor_pre_label_all.append(pre_label_rumor.detach().cpu().numpy())
        rumor_label_all.append(label.detach().cpu().numpy())
        # Store shallow, deep feature for visualization
        #label_matrix.append(label.cpu().detach().numpy())
        #feature_shallow1_matrix.append(feature_shallow1.cpu().detach().numpy())
        #feature_shallow2_matrix.append(feature_shallow2.cpu().detach().numpy())
        #feature_shallow3_matrix.append(feature_shallow3.cpu().detach().numpy())
        #feature_deep1_matrix.append(feature_deep1.cpu().detach().numpy())
        #feature_deep2_matrix.append(feature_deep2.cpu().detach().numpy())
        #feature_deep3_matrix.append(feature_deep3.cpu().detach().numpy())
        #feature_deep4_matrix.append(feature_deep4.cpu().detach().numpy())
    #save shallow, deep feature
    #all_label_np = np.concatenate(label_matrix, axis=0)
    #all_feature_shallow1_np = np.concatenate(feature_shallow1_matrix, axis=0)
    #all_feature_shallow2_np = np.concatenate(feature_shallow2_matrix, axis=0)
    #all_feature_shallow3_np = np.concatenate(feature_shallow3_matrix, axis=0)
    #all_feature_deep1_np = np.concatenate(feature_deep1_matrix, axis=0)
    #all_feature_deep2_np = np.concatenate(feature_deep2_matrix, axis=0)
    #all_feature_deep3_np = np.concatenate(feature_deep3_matrix, axis=0)
    #all_feature_deep4_np = np.concatenate(feature_deep4_matrix, axis=0)
    #matrix_save_dir = './weibo_dataset'
    #np.save('{}/label_matrix'.format(matrix_save_dir), all_label_np)
    #np.save('{}/feature_shallow1_matrix'.format(matrix_save_dir), all_feature_shallow1_np)
    #np.save('{}/feature_shallow2_matrix'.format(matrix_save_dir), all_feature_shallow2_np)
    #np.save('{}/feature_shallow3_matrix'.format(matrix_save_dir), all_feature_shallow3_np)
    #np.save('{}/feature_deep1_matrix'.format(matrix_save_dir), all_feature_deep1_np)
    #np.save('{}/feature_deep2_matrix'.format(matrix_save_dir), all_feature_deep2_np)
    #np.save('{}/feature_deep3_matrix'.format(matrix_save_dir), all_feature_deep3_np)
    #np.save('{}/feature_deep4_matrix'.format(matrix_save_dir), all_feature_deep4_np)


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

import torch.utils.data as data
import numpy as np
import torch
import torch.utils


class PreDataset(data.Dataset):
    """
    Load Prepare datasets, including images and texts
    Possible datasets: Weibo, Twitter
    """
    def __init__(self, data_path, data_split):#输入是#./my_data  data_split：train/test
        self.images = np.load('{}/{}_image_embed.npy'.format(data_path, data_split))#原始图像的resnet嵌入，1000维特征
        #self.images = self.images.astype(np.float)

        self.images_clip = np.load('{}/{}_image_clip_embed.npy'.format(data_path, data_split))  #原图的clip特征512维度
        #self.texts_clip = self.texts_clip.astype(np.float)

        label = np.load('{}/{}_label.npy'.format(data_path, data_split))#标签特征，(6147, 2)(1699, 2)
        #labels = labels.long()
        self.labels = label[:, 0]  # 选择所有行（:）和第一列（0）
        #self.ids = np.load('{}/{}_post_ids.npy'.format(data_path, data_split))

        self.generated_images = np.load('{}/{}_generated_image_embed.npy'.format(data_path, data_split))#生成图特征1000维的res特征，下面一样
        #self.generated_images = self.generated_images.astype(np.float)

        self.comparison_images = np.load('{}/{}_comparison_image_embed.npy'.format(data_path, data_split))#对比图特征1000维的res特征
        #self.comparison_images = self.comparison_images.astype(np.float)

        self.diff_absolute_images = np.load('{}/{}_diff_absolute_image_embed.npy'.format(data_path, data_split))#绝对差图1000维的res特征
        #self.diff_absolute_images = self.diff_absolute_images.astype(np.float)

        self.diff_relative_images = np.load('{}/{}_diff_relative_image_embed.npy'.format(data_path, data_split))#相对差图1000维的res特征
        #self.diff_relative_images = self.diff_relative_images.astype(np.float)

        self.texts_clip = np.load('{}/{}_clip_embed.npy'.format(data_path, data_split))#文本的clip特征512维度
        #self.texts_clip = self.texts_clip.astype(np.float)

        self.texts_bert_token = np.load('{}/{}_input_ids.npy'.format(data_path, data_split))  # 文本的bert的token，300维度

        self.length = len(self.labels)#训练集或测试集的长度

        #texts_dim = self.texts_i.shape[1] * self.texts_i.shape[2]#50*768=38400#因为我不是和原论文一样，原论文直接seq_length=50*dim768。
        #self.texts = np.reshape(self.texts_i, (self.length, texts_dim))


        print('  {} image resnet emb shape: {}'.format(data_split, self.images.shape))#原图resnet1000维度
        print('  {} image clip emb shape: {}'.format(data_split, self.images_clip.shape))#原图clip512维度
        print('  {} image generated_emb shape: {}'.format(data_split, self.generated_images.shape))#生成图resnet1000维
        print('  {} image comparison_emb shape: {}'.format(data_split, self.comparison_images.shape))
        print('  {} image diff_absolute_emb shape: {}'.format(data_split, self.diff_absolute_images.shape))
        print('  {} image diff_relative_emb shape: {}'.format(data_split, self.diff_relative_images.shape))
        print('  {} text clip emb shape: {}'.format(data_split, self.texts_clip.shape))#文本clip 512维度
        print('  {} text bert token shape: {}'.format(data_split, self.texts_bert_token.shape))  #最大长度300，所以维度300


    def __getitem__(self, index):
        image_res_embs = torch.tensor(self.images[index]).float()#返回原图resnet特征
        image_clip_embs = torch.tensor(self.images_clip[index]).float()#返回原图clip特征
        image_generated_embs = torch.tensor(self.generated_images[index]).float()#返回生成图特征
        image_comparison_embs = torch.tensor(self.comparison_images[index]).float()#返回比较图特征
        image_diff_absolute_embs = torch.tensor(self.diff_absolute_images[index]).float()#返回绝对差图特征
        image_diff_relative_embs = torch.tensor(self.diff_relative_images[index]).float()#返回相对差图特征
        text_bert_token = torch.tensor(self.texts_bert_token[index]).float()
        text_clip_embs = torch.tensor(self.texts_clip[index]).float()
        labels = torch.tensor(self.labels[index])
        #ids = self.ids[index]

        if torch.cuda.is_available():
            image_res_embs = image_res_embs.cuda()
            image_clip_embs = image_clip_embs.cuda()
            image_generated_embs = image_generated_embs.cuda()
            image_comparison_embs = image_comparison_embs.cuda()
            image_diff_absolute_embs = image_diff_absolute_embs.cuda()
            image_diff_relative_embs = image_diff_relative_embs.cuda()
            text_bert_token = text_bert_token.cuda()
            text_clip_embs = text_clip_embs.cuda()
            labels = labels.cuda()
        #return image_embs, text_embs, labels, labels, ids
        return image_res_embs, image_clip_embs, image_generated_embs, image_comparison_embs, image_diff_absolute_embs, image_diff_relative_embs, text_bert_token, text_clip_embs, labels
    #返回值：图像的res，clip，生成图，对比图，绝对差图，相对差图，bert的token，文本clip，标签

    def __len__(self):
        return self.length


def get_loaders(data_path, batch_size):#./my_data  bs

    train_data = PreDataset(data_path, 'train')
    test_data = PreDataset(data_path, 'test')

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = get_loaders('./my_data', 128)
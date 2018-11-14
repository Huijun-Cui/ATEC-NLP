# this is the first main file
# implement unique word
import pandas as pd
import re
import gensim
import torch
import os
import numpy as np
import torch

import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.autograd import Variable
import torch.nn.functional as F
import jieba


class Config(object):
    def __init__(self):
        self.embedding_dims_word = 300
        self.hidden_dims_word = 200
        self.embedding_dims_char = 300
        self.hidden_dims_char = 200
        self.learning_rate = 0.001
        self.alpha = 0.5


opt = Config()


# 1.compute label weight after each epoch using validation data.
# def compute_labels_weights(weights_label,logits,labels):
#     """
#     compute weights for labels in current batch, and update weights_label(a dict)
#     :param weights_label:a dict
#     :param logit: [None,Vocabulary_size]
#     :param label: [None,]
#     :return:
#     """
#     # labels_predict=np.argmax(logits,axis=1) # logits:(256,108,754)
#     labels_predict = logits.topk(1,dim = 1)[1]
#     labels = labels.cpu().detach().numpy().tolist()
#     for i in range(len(labels)):
#         label=labels[i]
#         label_predict=labels_predict[i].item()
#         weight=weights_label.get(label,None)
#         if weight==None:
#             if label_predict == label:
#                 weights_label[label]=(1,1)
#             else:
#                 weights_label[label]=(1,0)
#         else:
#             number=weight[0]
#             correct=weight[1]
#             number=number+1.7
#             if label_predict==label:
#                 correct=correct+1.7
#             weights_label[label]=(number,correct)
#     return weights_label










class DecomseAtten(nn.Module):

    def __init__(self, num_embeddings_c, embedding_size_c, num_embeddings_w, embedding_size_w, hidden_size_c, \
                 hidden_size_w, label_size, feat_size=30, weight_char=None, weight_word=None):
        super(DecomseAtten, self).__init__()
        self.num_embeddings_c = num_embeddings_c
        self.embedding_size_c = embedding_size_c
        self.num_embeddings_w = num_embeddings_w
        self.embedding_size_w = embedding_size_w
        self.label_size = label_size
        # self.feat_liear = nn.Linear(feat_size, 10)
        self.sigmoid = nn.Sigmoid()

        if weight_char is not None:
            print('char pre train is used...........................')
            self.embedding_c = nn.Embedding.from_pretrained(weight_char, freeze=False)
        else:
            self.embedding_c = nn.Embedding(self.num_embeddings_c, self.embedding_size_c)

        if weight_word is not None:
            print('word pre train is used...........................')
            self.embedding_w = nn.Embedding.from_pretrained(weight_word, freeze=False)
        else:
            self.embedding_w = nn.Embedding(self.num_embeddings_w, self.embedding_size_w)

        self.hidden_size_c = hidden_size_c
        self.hidden_size_w = hidden_size_w

        self.mlp_f_c = self._mlp_layers(self.embedding_size_c, self.hidden_size_c)
        self.mlp_g_c = self._mlp_layers(2 * self.embedding_size_c, self.hidden_size_c)
        self.mlp_h_c = self._mlp_layers(4 * self.hidden_size_c, self.hidden_size_c)
        self.mlp_t_c = self._mlp_layers(2 * self.embedding_size_c, self.hidden_size_c)

        self.mlp_f_w = self._mlp_layers(self.embedding_size_w, self.hidden_size_w)
        self.mlp_g_w = self._mlp_layers(2 * self.embedding_size_w, self.hidden_size_w)
        self.mlp_h_w = self._mlp_layers(4 * self.hidden_size_w, self.hidden_size_w)
        self.mlp_t_w = self._mlp_layers(2 * self.embedding_size_w, self.hidden_size_w)

        self.c_iter_linear = self._mlp_layers(self.embedding_size_c * 12,30)
        self.w_iter_linear = self._mlp_layers(self.embedding_size_w * 12,30)
        self.unit_iter_linear = self._mlp_layers(self.embedding_size_c * 12,30)
        self.unit_iter_linear_w = self._mlp_layers(self.embedding_size_w * 12, 30)

        self.conv_c = nn.Conv1d(in_channels=2 * self.embedding_size_c, out_channels=50, kernel_size=3)
        self.conv_w = nn.Conv1d(in_channels=2 * self.embedding_size_w, out_channels=50, kernel_size=3)

        # self.embc_feat = nn.Linear(self.hidden_size_c,self.hidden_size_c)
        # self.embc_feat = self._mlp_layers_dropout_c(self.hidden_size_c,self.hidden_size_c)
        # self.embw_feat = nn.Linear(self.hidden_size_w,self.hidden_size_w)
        # self.embw_feat = self._mlp_layers_dropout_w(self.hidden_size_w,self.hidden_size_w)

        self.final_linear = self._mlp_final_layers(4 * self.hidden_size_c + 4 * self.hidden_size_w + 30*4 + 200, 2)
        self.label_size = label_size
        self.log_prob_c = nn.LogSoftmax()
        self.log_prob_w = nn.LogSoftmax()

        for m in self.modules():
            # print m
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.normal_(0, 0.01)

    def _mlp_layers(self, input_dim, output_dim):
        mlp_layers = []
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(
            input_dim, output_dim, bias=True))
        mlp_layers = []
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(
            input_dim, output_dim, bias=True))
        # mlp_layers.append(nn.BatchNorm1d(output_dim))
        mlp_layers.append(nn.ReLU())
        # mlp_layers.append(nn.Linear(
        #     output_dim, output_dim, bias=True))
        # mlp_layers.append(nn.BatchNorm1d(output_dim))
        # mlp_layers.append(nn.Tanh())
        return nn.Sequential(*mlp_layers)  # * used to unpack list

    def _mlp_final_layers(self, input_dim, output_dim):
        mlp_layers = []
        mlp_layers.append(nn.BatchNorm1d(input_dim))
        mlp_layers.append(nn.Linear(
            input_dim, output_dim, bias=True))
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.BatchNorm1d(output_dim))
        mlp_layers.append(nn.Linear(
            output_dim, output_dim, bias=True))
        # mlp_layers.append(nn.BatchNorm1d(output_dim))
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(output_dim, 2))
        return nn.Sequential(*mlp_layers)  # * used to unpack list

    def kmax_pooling(self,x,dim,k):
        index = x.topk(k,dim = dim)[1].sort(dim = dim)[0]
        return x.gather(dim,index)


    def forward(self, sent1_c, sent2_c, sent1_w, sent2_w, uni_1,uni_2,uni_1_w,uni_2_w):

        # criterion = ContrastiveLoss()

        '''
            sent_linear: batch_size x length x hidden_size
        '''
        # ---------------------------------------------------------------------------------



        sent1_linear_c = self.embedding_c(sent1_c)
        sent2_linear_c = self.embedding_c(sent2_c)
        sent1_linear_w = self.embedding_w(sent1_w)
        sent2_linear_w = self.embedding_w(sent2_w)

        sent1_c_iter = torch.cat([sent1_linear_c.mean(dim = 1),\
                                  sent1_linear_c.max(dim = 1)[0],sent1_linear_c.min(dim = 1)[0]],dim = 1)
        sent2_c_iter = torch.cat([sent2_linear_c.mean(dim=1),\
                                  sent2_linear_c.max(dim=1)[0],sent2_linear_c.min(dim = 1)[0]],dim = 1)

        sent1_w_iter = torch.cat([sent1_linear_w.mean(dim = 1),\
                                  sent1_linear_w.max(dim = 1)[0],sent1_linear_w.min(dim = 1)[0]],dim = 1)
        sent2_w_iter = torch.cat([sent2_linear_w.mean(dim=1),\
                                  sent2_linear_w.max(dim=1)[0],sent2_linear_w.min(dim = 1)[0]],dim =1)

        c_iter = torch.cat([sent1_c_iter,sent2_c_iter,sent1_c_iter - sent2_c_iter,\
                            torch.max(sent1_c_iter,sent2_c_iter)],dim = 1)
        c_iter = self.c_iter_linear(c_iter)
        w_iter = torch.cat([sent1_w_iter,sent2_w_iter,sent1_w_iter - sent2_w_iter,\
                            torch.max(sent1_w_iter,sent2_w_iter)],dim = 1)
        w_iter = self.w_iter_linear(w_iter)
        uni_1_emb = self.embedding_c(uni_1)
        uni_2_emb = self.embedding_c(uni_2)
        uni_1_vec = torch.cat([uni_1_emb.mean(dim = 1),\
                               uni_1_emb.max(dim = 1)[0],uni_1_emb.min(dim=1)[0]],dim = 1)
        uni_2_vec = torch.cat([uni_2_emb.mean(dim=1),\
                               uni_2_emb.max(dim=1)[0],uni_2_emb.min(dim=1)[0]],dim = 1)
        uni_iter = torch.cat([uni_1_vec,uni_2_vec,torch.abs(uni_1_vec - uni_2_vec),\
                              torch.max(uni_1_vec,uni_2_vec)],dim = 1)
        uni_iter = self.unit_iter_linear(uni_iter)

        uni_1_emb_w = self.embedding_w(uni_1_w)
        uni_2_emb_w = self.embedding_w(uni_2_w)
        uni_1_vec_w = torch.cat([uni_1_emb_w.mean(dim=1), \
                               uni_1_emb_w.max(dim=1)[0], uni_1_emb_w.min(dim=1)[0]], dim=1)
        uni_2_vec_w = torch.cat([uni_2_emb_w.mean(dim=1), \
                               uni_2_emb_w.max(dim=1)[0], uni_2_emb_w.min(dim=1)[0]], dim=1)
        uni_iter_w = torch.cat([uni_1_vec_w, uni_2_vec_w, torch.abs(uni_1_vec_w - uni_2_vec_w), \
                              torch.max(uni_1_vec_w, uni_2_vec_w)], dim=1)
        uni_iter_w = self.unit_iter_linear_w(uni_iter_w)






        # inter_c = (sent1_linear_c - sent2_linear_c).topk(1,dim = 1)[0].view(ba_size,-1)
        # inter_w = (sent1_linear_w - sent2_linear_w).topk(1, dim=1)[0].view(ba_size, -1)
        # inter_c_hidden = self.embc_feat(inter_c)
        # inter_w_hidden = self.embw_feat(inter_w)

        len1_c = sent1_linear_c.size(1)
        len2_c = sent2_linear_c.size(1)

        '''attend'''

        f1_c = self.mlp_f_c(sent1_linear_c.view(-1, self.embedding_size_c))
        # f1_c = sent1_linear_c
        f2_c = self.mlp_f_c(sent2_linear_c.view(-1, self.embedding_size_c))
        # f2_c = sent2_linear_c

        f1_c = f1_c.view(-1, len1_c, self.hidden_size_c)
        # batch_size x len1 x hidden_size

        f2_c = f2_c.view(-1, len2_c, self.hidden_size_c)
        # batch_size x len2 x hidden_size

        score1_c = torch.bmm(f1_c, torch.transpose(f2_c, 1, 2))
        # e_{ij} batch_size x len1 x len2
        prob1_c = F.softmax(score1_c.view(-1, len2_c), dim=1).view(-1, len1_c, len2_c)
        # batch_size x len1 x len2

        score2_c = torch.transpose(score1_c.contiguous(), 1, 2)
        score2_c = score2_c.contiguous()
        # e_{ji} batch_size x len2 x len1
        prob2_c = F.softmax(score2_c.view(-1, len1_c), dim=1).view(-1, len2_c, len1_c)
        # batch_size x len2 x len1

        sent1_combine_c = torch.cat(
            (sent1_linear_c, torch.bmm(prob1_c, sent2_linear_c)), 2)
        # batch_size x len1 x (embed_size x 2)
        sent2_combine_c = torch.cat(
            (sent2_linear_c, torch.bmm(prob2_c, sent1_linear_c)), 2)

        sen1c_cnn = sent1_combine_c.permute(0, 2, 1)
        sen2c_cnn = sent2_combine_c.permute(0, 2, 1)
        sen1c_cnn = self.conv_c(sen1c_cnn)
        sen2c_cnn = self.conv_c(sen2c_cnn)
        sen1c_cnn = self.kmax_pooling(sen1c_cnn, 2, 1).squeeze(2)
        sen2c_cnn = self.kmax_pooling(sen2c_cnn, 2, 1).squeeze(2)




        # batch_size x len2 x (embed_size x 2)

        '''sum'''
        # g1_c = sent1_combine_c
        # g2_c = sent2_combine_c
        g1_c = self.mlp_g_c(sent1_combine_c.view(-1, 2 * self.embedding_size_c))
        g2_c = self.mlp_g_c(sent2_combine_c.view(-1, 2 * self.embedding_size_c))
        g1_c = g1_c.view(-1, len1_c, self.hidden_size_c)
        # # batch_size x len1 x hidden_size
        g2_c = g2_c.view(-1, len2_c, self.hidden_size_c)

        # g1_t_c = self.mlp_t_c_2(sent1_combine_c.view(-1, 2 * self.embedding_size_c))
        # g2_t_c = self.mlp_t_c_2(sent2_combine_c.view(-1, 2 * self.embedding_size_c))
        # g1_t_c = g1_t_c.view(-1, len1_c, self.hidden_size_c)
        # g2_t_c = g2_t_c.view(-1, len2_c, self.hidden_size_c)
        # batch_size x len2 x hidden_size

        # g1_t_c_2 = self.mlp_t_c_2(sent1_combine_c.view(-1, 2 * self.embedding_size_c))
        # g2_t_c_2 = self.mlp_t_c_2(sent2_combine_c.view(-1, 2 * self.embedding_size_c))
        # g1_t_c_2 = g1_t_c_2.view(-1, len1_c, self.hidden_size_c)
        # g2_t_c_2 = g2_t_c_2.view(-1, len2_c, self.hidden_size_c)
        # batch_size x len2 x hidden_size

        # sent1_output_c = torch.sum(g1_c, 1)  # batch_size x 1 x hidden_size
        # sent1_output_c = torch.squeeze(sent1_output_c, 1)
        #
        # sent2_output_c = torch.sum(g2_c, 1)  # batch_size x 1 x hidden_size
        # sent2_output_c = torch.squeeze(sent2_output_c, 1)

        sent1_output_c = torch.cat([g1_c.mean(dim=1), g1_c.max(dim=1)[0]], dim=1)
        sent2_output_c = torch.cat([g2_c.mean(dim=1), g2_c.max(dim=1)[0]], dim=1)

        # input_combine_c = torch.cat((sent1_output_c, sent2_output_c, \
        #                              torch.max(sent1_output_c, sent2_output_c), \
        #                              torch.abs(sent1_output_c - sent2_output_c)), 1)

        # sent1_output_t_c = torch.sum(g1_t_c, 1)  # batch_size x 1 x hidden_size
        # sent1_output_t_c = torch.squeeze(sent1_output_t_c, 1)
        # sent2_output_t_c = torch.sum(g2_t_c, 1)  # batch_size x 1 x hidden_size
        # sent2_output_t_c = torch.squeeze(sent2_output_t_c, 1)
        #
        # input_combine_t_c = torch.cat((sent1_output_t_c, sent2_output_t_c, \
        #                              torch.max(sent1_output_t_c, sent2_output_t_c), \
        #                              torch.abs(sent1_output_t_c - sent2_output_t_c)), 1)
        #
        # sent1_output_t_c_2 = torch.sum(g1_t_c_2, 1)  # batch_size x 1 x hidden_size
        # sent1_output_t_c_2 = torch.squeeze(sent1_output_t_c_2, 1)
        # sent2_output_t_c_2 = torch.sum(g2_t_c_2, 1)  # batch_size x 1 x hidden_size
        # sent2_output_t_c_2 = torch.squeeze(sent2_output_t_c_2, 1)
        #
        # input_combine_t_c_2 = torch.cat((sent1_output_t_c_2, sent2_output_t_c_2, \
        #                                torch.max(sent1_output_t_c_2, sent2_output_t_c_2), \
        #                                torch.abs(sent1_output_t_c_2 - sent2_output_t_c_2)), 1)

        # batch_size x (2 * hidden_size)
        # h_c = self.mlp_h_c(input_combine_c)
        # h_c_t = self.mlp_h_c_t(input_combine_t_c)
        # h_c_t_2 = self.mlp_h_c_t_2(input_combine_t_c_2)
        # batch_size * hidden_size

        # if sample_id == 15:
        #     print '-2 layer'
        #     print h.data[:, 100:150]

        # ------------------------------------------------------------------------------------------

        len1_w = sent1_linear_w.size(1)
        len2_w = sent2_linear_w.size(1)

        '''attend'''

        f1_w = self.mlp_f_w(sent1_linear_w.view(-1, self.embedding_size_w))
        f2_w = self.mlp_f_w(sent2_linear_w.view(-1, self.embedding_size_w))

        # f1_w = sent1_linear_w
        # f2_w = sent2_linear_w
        f1_w = f1_w.view(-1, len1_w, self.hidden_size_w)
        # batch_size x len1 x hidden_size

        f2_w = f2_w.view(-1, len2_w, self.hidden_size_w)
        # batch_size x len2 x hidden_size

        score1_w = torch.bmm(f1_w, torch.transpose(f2_w, 1, 2))
        # e_{ij} batch_size x len1 x len2
        score1_w = score1_w.contiguous()
        prob1_w = F.softmax(score1_w.view(-1, len2_w)).view(-1, len1_w, len2_w)
        # batch_size x len1 x len2

        score2_w = torch.transpose(score1_w.contiguous(), 1, 2)
        score2_w = score2_w.contiguous()
        # e_{ji} batch_size x len2 x len1
        prob2_w = F.softmax(score2_w.view(-1, len1_w)).view(-1, len2_w, len1_w)
        # batch_size x len2 x len1

        sent1_combine_w = torch.cat(
            (sent1_linear_w, torch.bmm(prob1_w, sent2_linear_w)), 2)
        # batch_size x len1 x (embed_size x 2)
        sent2_combine_w = torch.cat(
            (sent2_linear_w, torch.bmm(prob2_w, sent1_linear_w)), 2)

        sen1w_cnn = sent1_combine_w.permute(0, 2, 1)
        sen2w_cnn = sent2_combine_w.permute(0, 2, 1)
        sen1w_cnn = self.conv_c(sen1w_cnn)
        sen2w_cnn = self.conv_c(sen2w_cnn)
        sen1w_cnn = self.kmax_pooling(sen1w_cnn, 2, 1).squeeze(2)
        sen2w_cnn = self.kmax_pooling(sen2w_cnn, 2, 1).squeeze(2)




        # batch_size x len2 x (embed_size x 2)

        '''sum'''
        # g1_w = sent1_combine_w
        # g2_w = sent2_combine_w
        g1_w = self.mlp_g_w(sent1_combine_w.view(-1, 2 * self.embedding_size_w))
        g2_w = self.mlp_g_w(sent2_combine_w.view(-1, 2 * self.embedding_size_w))
        g1_w = g1_w.view(-1, len1_w, self.hidden_size_w)
        # # batch_size x len1 x hidden_size
        g2_w = g2_w.view(-1, len2_w, self.hidden_size_w)
        # batch_size x len2 x hidden_size

        # g1_t_w = self.mlp_t_w_2(sent1_combine_w.view(-1, 2 * self.embedding_size_w))
        # g2_t_w = self.mlp_t_w_2(sent2_combine_w.view(-1, 2 * self.embedding_size_w))
        # g1_t_w = g1_t_w.view(-1, len1_w, self.hidden_size_w)
        # g2_t_w = g2_t_w.view(-1, len2_w, self.hidden_size_w)
        #
        # g1_t_w_2 = self.mlp_t_w_2(sent1_combine_w.view(-1, 2 * self.embedding_size_w))
        # g2_t_w_2 = self.mlp_t_w_2(sent2_combine_w.view(-1, 2 * self.embedding_size_w))
        # g1_t_w_2 = g1_t_w_2.view(-1, len1_w, self.hidden_size_w)
        # g2_t_w_2 = g2_t_w_2.view(-1, len2_w, self.hidden_size_w)

        # sent1_output_w = torch.sum(g1_w, 1)  # batch_size x 1 x hidden_size
        # sent1_output_w = torch.squeeze(sent1_output_w, 1)
        sent1_output_w = torch.cat([g1_w.mean(dim=1), g1_w.max(dim=1)[0]], dim=1)

        # sent2_output_w = torch.sum(g2_w, 1)  # batch_size x 1 x hidden_size
        # sent2_output_w = torch.squeeze(sent2_output_w, 1)
        sent2_output_w = torch.cat([g2_w.mean(dim=1), g2_w.max(dim=1)[0]], dim=1)

        # input_combine_w = torch.cat((sent1_output_w, sent2_output_w), 1)

        # input_combine_w = torch.cat((sent1_output_w, sent2_output_w, \
        #                              torch.max(sent1_output_w, sent2_output_w), \
        #                              torch.abs(sent1_output_w - sent2_output_w)), 1)

        # sent1_output_t_w = torch.sum(g1_t_w, 1)  # batch_size x 1 x hidden_size
        # sent1_output_t_w = torch.squeeze(sent1_output_t_w, 1)
        # sent2_output_t_w = torch.sum(g2_t_w, 1)  # batch_size x 1 x hidden_size
        # sent2_output_t_w = torch.squeeze(sent2_output_t_w, 1)
        #
        # input_combine_t_w = torch.cat((sent1_output_t_w, sent2_output_t_w, \
        #                                torch.max(sent1_output_t_w, sent2_output_t_w), \
        #                                torch.abs(sent1_output_t_w - sent2_output_t_w)), 1)

        # sent1_output_t_w_2 = torch.sum(g1_t_w_2, 1)  # batch_size x 1 x hidden_size
        # sent1_output_t_w_2 = torch.squeeze(sent1_output_t_w_2, 1)
        # sent2_output_t_w_2 = torch.sum(g2_t_w_2, 1)  # batch_size x 1 x hidden_size
        # sent2_output_t_w_2 = torch.squeeze(sent2_output_t_w_2, 1)
        #
        # input_combine_t_w_2 = torch.cat((sent1_output_t_w_2, sent2_output_t_w_2, \
        #                                torch.max(sent1_output_t_w_2, sent2_output_t_w_2), \
        #                                torch.abs(sent1_output_t_w_2 - sent2_output_t_w_2)), 1)

        # batch_size x (2 * hidden_size)
        # h_w = self.mlp_h_w(input_combine_w)
        # h_w_t = self.mlp_h_w_t(input_combine_t_w)
        # h_w_t_2 = self.mlp_h_w_t(input_combine_t_w_2)

        # h = torch.cat([h_c, h_w,h_c_t,h_w_t,h_c_t_2,h_w_t_2], dim=1)

        h = torch.cat([sent1_output_c, sent2_output_c, sent1_output_w,\
                       sent2_output_w,c_iter,w_iter,uni_iter,uni_iter_w,\
                       sen1c_cnn,sen2c_cnn,sen1w_cnn,sen2w_cnn], dim=1)
        h_out = self.final_linear(h)

        # print 'final layer'
        # print h.data

        # log_prob = self.log_prob(h)
        # print(self.sigmoid(h_out))[:,:10]
        return h_out


def CreateBatchTensorChar(w2i_c, maxlen_c,df):
    result_sen1 = []
    result_len1 = []
    result_sen2 = []
    result_len2 = []
    # pattern = re.compile(u'[^\u4e00-\u9fa5]')
    # item_lexical = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", item_lexical)
    for sen1, sen2 in zip(df['sent1'], df['sent2']):
        # sen1_drop = pattern.sub(r'', sen1)
        sen1_drop = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", sen1)
        # sen2_drop = pattern.sub(r'', sen2)
        sen2_drop = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", sen2)
        sen1_list = [0] * maxlen_c
        sen2_list = [0] * maxlen_c
        ix_1 = 0
        ix_2 = 0
        for w1 in sen1_drop:
            if w1 in w2i_c and ix_1 < maxlen_c:
                sen1_list[ix_1] = w2i_c[w1]
                ix_1 += 1
            elif ix_1 < maxlen_c:
                sen1_list[ix_1] = w2i_c['A_c']
                ix_1 += 1
        if ix_1 == 0:
            ix_1 += 1
        for w2 in sen2_drop:
            if w2 in w2i_c and ix_2 < maxlen_c:
                sen2_list[ix_2] = w2i_c[w2]
                ix_2 += 1
            elif ix_2 < maxlen_c:
                sen2_list[ix_2] = w2i_c['A_c']
                ix_2 += 1
        if ix_2 == 0:
            ix_2 += 1









        result_sen1.append(sen1_list)
        result_sen2.append(sen2_list)
        result_len1.append(ix_1)
        result_len2.append(ix_2)
    return torch.LongTensor(result_sen1).cuda(), torch.LongTensor(result_sen2).cuda(), \
           torch.tensor(result_len1).cuda(), torch.tensor(result_len2).cuda()


def CreateBatchTensorWord(w2i_w, maxlen_w, df):
    result_sen1 = []
    result_len1 = []
    result_sen2 = []
    result_len2 = []
    # pattern = re.compile(u'[^\u4e00-\u9fa5\s]')
    for sen1, sen2 in zip(df['sent1'], df['sent2']):

        sen1_drop = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*\s]', "", sen1)
        # sen2_drop = pattern.sub(r'', sen2)
        sen2_drop = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*\s]', "", sen2)
        # sen1_drop = pattern.sub(r'', sen1)
        # sen2_drop = pattern.sub(r'', sen2)
        sen1_list = [0] * maxlen_w
        sen2_list = [0] * maxlen_w
        ix_1 = 0
        ix_2 = 0
        for w1 in sen1_drop.split():
            if w1 in w2i_w and ix_1 < maxlen_w:
                sen1_list[ix_1] = w2i_w[w1]
                ix_1 += 1
        if ix_1 == 0:
            ix_1 +=1
        for w2 in sen2_drop.split():
            if w2 in w2i_w and ix_2 < maxlen_w:
                sen2_list[ix_2] = w2i_w[w2]
                ix_2 +=1
        if ix_2 == 0:
            ix_2 +=1





        result_sen1.append(sen1_list)
        result_sen2.append(sen2_list)
        result_len1.append(ix_1)
        result_len2.append(ix_2)
    return torch.LongTensor(result_sen1).cuda(), torch.LongTensor(result_sen2).cuda(), \
           torch.tensor(result_len1).cuda(), torch.tensor(result_len2).cuda()


def CharPik(train_df):
    maxlenth_c = 0
    # text = re.sub(r"[^A-Za-z0-9]", " ", text)
    # pattern = re.compile(u'[^\u4e00-\u9fa5A-Za-z0-9*]')
    sen = pd.concat([train_df['sent1'], train_df['sent2']], ignore_index=True)
    # create dict {word:num},and sum word count
    sum_count = 0
    char_count_dict = {}
    for item in sen:
        item_lexical = item.replace('***', '*')
        item_lexical = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", item_lexical)

        for w in item_lexical:
            sum_count += 1
            if w not in char_count_dict:
                char_count_dict[w] = 0
            char_count_dict[w] += 1
    print('the total number of char is {}'.format(sum_count))

    sentences = []
    for item in sen:
        item_lexical = item.replace('***', '*')
        item_lexical = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", item_lexical)

        tmp = []
        for w in item_lexical:
            if w in char_count_dict and char_count_dict[w] > 5:
                tmp.append(w)
            else:
                tmp.append('A_c')
                # print('meet the rare char :{}'.format(w.encode('utf-8')))
        if len(tmp) > maxlenth_c:
            maxlenth_c = len(tmp)
        sentences.append(tmp)
    word_rep = opt.embedding_dims_char
    model_char = gensim.models.Word2Vec(sentences, size=word_rep, min_count=1, iter=10)
    w2i = {}
    i2w = []
    i2w.append('END')
    w2i['END'] = 0
    for item in model_char.wv.vocab:
        i2w.append(item)
        w2i[item] = len(w2i)
    weights = []
    weights.append([0] * word_rep)
    length = len(model_char.wv.vocab)
    for i in range(length):
        weights.append(model_char[i2w[i + 1]].tolist())
    weights = torch.FloatTensor(weights)
    # -------------------------------------------------------------------------------
    maxlen_unique = 0
    for sent1, sent2 in zip(train_df['sent1'], train_df['sent2']):
        sent1_lexical = sent1.replace('***', '*')
        sent1_lexical = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", sent1_lexical)

        sent2_lexical = sent2.replace('***', '*')
        sent2_lexical = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", sent2_lexical)

        tmp_count = 0
        tmp_uni_list = []
        for w in sent1_lexical:
            if w not in sent2_lexical and w not in tmp_uni_list:
                tmp_count += 1
                tmp_uni_list.append(w)
        if tmp_count > maxlen_unique:
            maxlen_unique = tmp_count
        tmp_count = 0
        tmp_uni_list = []

        for w in sent2_lexical:
            if w not in sent1_lexical and w not in tmp_uni_list:
                tmp_count += 1
                tmp_uni_list.append(w)
        if tmp_count > maxlen_unique:
            maxlen_unique = tmp_count




    return w2i, i2w, weights, model_char, maxlenth_c,maxlen_unique


def GetUnitFeat_char(train_df,w2i_c,maxlen_unique):


    uni_list_1 = []
    uni_list_2 = []
    for sent1,sent2 in zip(train_df['sent1'],train_df['sent2']):
        sent1_lexical = sent1.replace('***', '*')
        sent1_lexical = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", sent1_lexical)

        sent2_lexical = sent2.replace('***', '*')
        sent2_lexical = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", sent2_lexical)



        tmp_sent1 = [0] * maxlen_unique
        tmp_sent2 = [0] * maxlen_unique
        ix_sen1 = 0
        for w in sent1_lexical:
            if w not in sent2_lexical and ix_sen1 < maxlen_unique:
                if w in w2i_c:
                    tmp_sent1[ix_sen1] = w2i_c[w]
                else:
                    tmp_sent1[ix_sen1] = w2i_c['A_c']
                ix_sen1 +=1
        ix_sen2 = 0
        for w in sent2_lexical:
            if w not in sent1_lexical and ix_sen2 < maxlen_unique:
                if w in w2i_c:
                    tmp_sent2[ix_sen2] = w2i_c[w]
                else:
                    tmp_sent2[ix_sen2] = w2i_c['A_c']
                ix_sen2 +=1
        uni_list_1.append(tmp_sent1)
        uni_list_2.append(tmp_sent2)
    return torch.LongTensor(uni_list_1).cuda(),torch.LongTensor(uni_list_2).cuda()

def GetUnitFeat_word(train_df,w2i_w,maxlen_unique):

    uni_list_1 = []
    uni_list_2 = []
    for sent1,sent2 in zip(train_df['sent1'],train_df['sent2']):
        sent1_lexical = sent1.replace('***', '*')
        sent1_lexical = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*\s]', "", sent1_lexical)

        sent2_lexical = sent2.replace('***', '*')
        sent2_lexical = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*\s]', "", sent2_lexical)



        tmp_sent1 = [0] * maxlen_unique
        tmp_sent2 = [0] * maxlen_unique
        ix_sen1 = 0
        for w in sent1_lexical.split():
            if w not in sent2_lexical.split() and ix_sen1 < maxlen_unique:
                if w in w2i_w:
                    tmp_sent1[ix_sen1] = w2i_w[w]
                else:
                    tmp_sent1[ix_sen1] = w2i_w['A_w']
                ix_sen1 +=1
        ix_sen2 = 0
        for w in sent2_lexical.split():
            if w not in sent1_lexical.split() and ix_sen2 < maxlen_unique:
                if w in w2i_w:
                    tmp_sent2[ix_sen2] = w2i_w[w]
                else:
                    tmp_sent2[ix_sen2] = w2i_w['A_w']
                ix_sen2 +=1
        uni_list_1.append(tmp_sent1)
        uni_list_2.append(tmp_sent2)


    return torch.LongTensor(uni_list_1).cuda(),torch.LongTensor(uni_list_2).cuda()








def CharPikBigram(train_df):
    maxlenth_c = 0
    # text = re.sub(r"[^A-Za-z0-9]", " ", text)
    # pattern = re.compile(u'[^\u4e00-\u9fa5A-Za-z0-9*]')
    sen = pd.concat([train_df['sent1'], train_df['sent2']], ignore_index=True)
    # create dict {word:num},and sum word count
    sum_count = 0
    count_dict = {}
    for item in sen:
        item_lexical = item.replace('***', '*')
        item_lexical = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", item_lexical)
        item_lexical_2gram = split_string_as_list_by_ngram(input_string=item_lexical, ngram_value=2)

        for w in item_lexical_2gram:
            sum_count += 1
            if w not in count_dict:
                count_dict[w] = 0
            count_dict[w] += 1
    print('the total number of char 2gram  is {}'.format(sum_count))

    sentences = []
    for item in sen:
        item_lexical = item.replace('***', '*')
        item_lexical = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", item_lexical)
        item_lexical_2gram = split_string_as_list_by_ngram(input_string=item_lexical, ngram_value=2)
        tmp = []
        for w in item_lexical_2gram:
            if w in count_dict and count_dict[w] > 5:
                tmp.append(w)
            else:
                tmp.append('A_c')
                # print('meet the rare char :{}'.format(w.encode('utf-8')))
        if len(tmp) > maxlenth_c:
            maxlenth_c = len(tmp)
        sentences.append(tmp)
    word_rep = opt.embedding_dims_char
    model_char = gensim.models.Word2Vec(sentences, size=word_rep, min_count=1, iter=10)
    w2i = {}
    i2w = []
    i2w.append('END')
    w2i['END'] = 0
    for item in model_char.wv.vocab:
        i2w.append(item)
        w2i[item] = len(w2i)
    weights = []
    weights.append([0] * word_rep)
    length = len(model_char.wv.vocab)
    for i in range(length):
        weights.append(model_char[i2w[i + 1]].tolist())
    weights = torch.FloatTensor(weights)
    return w2i, i2w, weights, model_char, maxlenth_c


def WordPik(train_df):
    maxlenth_w = 0
    # pattern = re.compile(u'[^\u4e00-\u9fa5\s]')
    sen = pd.concat([train_df['sent1'], train_df['sent2']], ignore_index=True)

    sum_count = 0
    word_count_dict = {}
    for item in sen:
        item_lexical = item.replace('***', '*')
        item_lexical = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*\s]', "", item_lexical)

        for w in item_lexical.split():
            sum_count += 1
            if w not in word_count_dict:
                word_count_dict[w] = 0
            word_count_dict[w] += 1
    print('the total number of word is {}'.format(sum_count))

    sentences = []
    for item in sen:
        item_lexical = item.replace('***', '*')
        item_lexical = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*\s]', "", item_lexical)

        tmp = []
        for w in item_lexical.split():
            if w in word_count_dict and word_count_dict[w] > 5:
                tmp.append(w)
            else:
                tmp.append('A_w')
                # print('meet the reate word : {}'.format(w.encode('utf-8')))

        if len(tmp) > maxlenth_w:
            maxlenth_w = len(tmp)
        sentences.append(tmp)
    word_rep = opt.embedding_dims_word
    model_word = gensim.models.Word2Vec(sentences, size=word_rep, min_count=1, iter=50)
    w2i = {}
    i2w = []
    i2w.append('END')
    w2i['END'] = 0
    for item in model_word.wv.vocab:
        i2w.append(item)
        w2i[item] = len(w2i)
    weights = []
    weights.append([0] * word_rep)
    length = len(model_word.wv.vocab)
    for i in range(length):
        weights.append(model_word[i2w[i + 1]].tolist())
    weights = torch.FloatTensor(weights)

    # ---------------------------------------------------------------------------
    maxlen_unique = 0
    for sent1, sent2 in zip(train_df['sent1'], train_df['sent2']):
        sent1_lexical = sent1.replace('***', '*')
        sent1_lexical = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*\s]', "", sent1_lexical)

        sent2_lexical = sent2.replace('***', '*')
        sent2_lexical = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*\s]', "", sent2_lexical)

        tmp_count = 0
        tmp_uni_list = []
        for w in sent1_lexical.split():
            if w not in sent2_lexical.split() and w not in tmp_uni_list:
                tmp_count += 1
                tmp_uni_list.append(w)
        if tmp_count > maxlen_unique:
            maxlen_unique = tmp_count
        tmp_count = 0
        tmp_uni_list = []

        for w in sent2_lexical.split():
            if w not in sent1_lexical.split() and w not in tmp_uni_list:
                tmp_count += 1
                tmp_uni_list.append(w)
        if tmp_count > maxlen_unique:
            maxlen_unique = tmp_count





    return w2i, i2w, weights, model_word, maxlenth_w,maxlen_unique


def SplitData(df_char, df_word, k_ix):
    step = len(df_char) // 10
    valid_c = df_char[:step * 2]
    train_c = df_word[step * 2:]
    valid_w = df_word[:step * 2]
    train_w = df_word[step * 2:]
    return train_c, train_w, valid_c, valid_w


def SpitData2(df_char, df_word, k_ix):
    step = len(df_char) // 10
    train_list = []
    valid_list = []
    valid_list.extend(range(k_ix * step, (k_ix + 2) * step))

    for item in range(len(df_char)):
        if item not in valid_list:
            train_list.append(item)

    train_df_char = df_char.iloc[train_list]
    valid_df_char = df_char.iloc[valid_list]

    train_df_word = df_word.iloc[train_list]
    valid_df_word = df_word.iloc[valid_list]

    return train_df_char, train_df_word, \
           valid_df_char, valid_df_word.reset_index(drop=True)


def SplitData3(df_char, df_word, k_ix):
    data = pd.concat([df_char, df_word], axis=1)
    data.columns = ['idc', 'sent1c', 'sent2c', 'labelc', 'idw', 'sent1w', 'sent2w', 'labelw']

    data_true = data[data['labelc'] == 1]
    data_true.sort_values(by='idc', inplace=True)

    data_false = data[data['labelc'] == 0]
    data_false.sort_values(by='idc', inplace=True)

    step_t = len(data_true) // 10
    train_list_t = []
    valid_list_t = []
    valid_list_t.extend(range(k_ix * step_t, (k_ix + 2) * step_t))
    for item in range(len(data_true)):
        if item not in valid_list_t:
            train_list_t.append(item)

    train_p = data_true.iloc[train_list_t]
    valid_p = data_true.iloc[valid_list_t]

    step_f = len(data_false) // 10
    train_list_f = []
    valid_list_f = []
    valid_list_f.extend(range(k_ix * step_f, (k_ix + 2) * step_f))
    for item in range(len(data_false)):
        if item not in valid_list_f:
            train_list_f.append(item)
    train_f = data_false.iloc[train_list_f]
    valid_f = data_false.iloc[valid_list_f]

    train = pd.concat([train_p, train_f], ignore_index=True)
    valid = pd.concat([valid_p, valid_f], ignore_index=True)

    train_df_char = train[['idc', 'sent1c', 'sent2c', 'labelc']]
    train_df_char.columns = ['id', 'sent1', 'sent2', 'label']

    train_df_word = train[['idw', 'sent1w', 'sent2w', 'labelw']]
    train_df_word.columns = ['id', 'sent1', 'sent2', 'label']

    valid_df_char = valid[['idc', 'sent1c', 'sent2c', 'labelc']]
    valid_df_char.columns = ['id', 'sent1', 'sent2', 'label']

    valid_df_word = valid[['idw', 'sent1w', 'sent2w', 'labelw']]
    valid_df_word.columns = ['id', 'sent1', 'sent2', 'label']

    return train_df_char, train_df_word, valid_df_char, valid_df_word


def get_tfidf_score_and_save_word(df):
    source_object = pd.concat([df['sent1'], df['sent2']], ignore_index=True)
    corpus = source_object.tolist()  # corpus = ["This is very strange","This is very nice"]
    # TfidfVectorizer = None  # TODO TODO TODO remove this.
    print("You need to import TfidfVectorizer first, if you want to use tfidif function.")
    vectorizer = TfidfVectorizer(analyzer=lambda x: x.split(' '), min_df=3, use_idf=1, smooth_idf=1, sublinear_tf=1)
    X = vectorizer.fit_transform(corpus)
    idf = vectorizer.idf_
    dict_word_tfidf = dict(zip(vectorizer.get_feature_names(), idf))
    return dict_word_tfidf


def get_tfidf_score_and_save_char(df):
    source_object = pd.concat([df['sent1'], df['sent2']], ignore_index=True)
    corpus_tmp = source_object.tolist()  # corpus = ["This is very strange","This is very nice"]
    # TfidfVectorizer = None  # TODO TODO TODO remove this.
    corpus = []
    for sen in corpus_tmp:
        tmp = []
        for w in sen:
            tmp.append(w)
        corpus.append(' '.join(tmp))
    print("You need to import TfidfVectorizer first, if you want to use tfidif function.")
    vectorizer = TfidfVectorizer(analyzer=lambda x: x.split(' '), min_df=3, use_idf=1, smooth_idf=1, sublinear_tf=1)
    X = vectorizer.fit_transform(corpus)
    idf = vectorizer.idf_
    dict_word_tfidf = dict(zip(vectorizer.get_feature_names(), idf))
    return dict_word_tfidf

def get_tfidf_score_and_save_bigram(df):
    source_object = pd.concat([df['sent1'], df['sent2']], ignore_index=True)
    corpus_tmp = source_object.tolist()  # corpus = ["This is very strange","This is very nice"]
    # TfidfVectorizer = None  # TODO TODO TODO remove this.
    corpus = []
    for sen in corpus_tmp:
        tmp = []
        bigram_list = split_string_as_list_by_ngram(sen,ngram_value=2)
        for w in bigram_list:
            tmp.append(w)
        corpus.append(' '.join(tmp))
    print("You need to import TfidfVectorizer first, if you want to use tfidif function.")
    vectorizer = TfidfVectorizer(analyzer=lambda x: x.split(' '), min_df=3, use_idf=1, smooth_idf=1, sublinear_tf=1)
    X = vectorizer.fit_transform(corpus)
    idf = vectorizer.idf_
    dict_word_tfidf = dict(zip(vectorizer.get_feature_names(), idf))
    return dict_word_tfidf





def split_string_as_list_by_ngram(input_string, ngram_value):
    # print("input_string0:",input_string)
    input_string = "".join([string for string in input_string if string.strip()])
    # print("input_string1:",input_string)
    length = len(input_string)
    result_string = []
    for i in range(length):
        if i + ngram_value < length + 1:
            result_string.append(input_string[i:i + ngram_value])
    # print("ngram:",ngram_value,"result_string:",result_string)
    return result_string


# input string word split by '\s'


def compute_blue_ngram(x1_list, x2_list):
    """
    compute blue score use ngram information. x1_list as predict sentence,x2_list as target sentence
    :param x1_list:
    :param x2_list:
    :return:
    """
    count_dict = {}
    count_dict_clip = {}
    # 1. count for each token at predict sentence side.
    for token in x1_list:
        if token not in count_dict:
            count_dict[token] = 1
        else:
            count_dict[token] = count_dict[token] + 1
    count = np.sum([value for key, value in count_dict.items()])

    # 2.count for tokens existing in predict sentence for target sentence side.
    for token in x2_list:
        if token in count_dict:
            if token not in count_dict_clip:
                count_dict_clip[token] = 1
            else:
                count_dict_clip[token] = count_dict_clip[token] + 1

    # 3. clip value to ceiling value for that token
    count_dict_clip = {key: (value if value <= count_dict[key] else count_dict[key]) for key, value in
                       count_dict_clip.items()}
    count_clip = np.sum([value for key, value in count_dict_clip.items()])
    result = float(count_clip) / (float(count) + 0.00000001)
    return result


def get_sentence_diff_overlap_pert(index, input_string_x1, input_string_x2):
    # 0. get list from string
    input_list1 = [input_string_x1[token] for token in range(len(input_string_x1)) if input_string_x1[token].strip()]
    input_list2 = [input_string_x2[token] for token in range(len(input_string_x2)) if input_string_x2[token].strip()]
    length1 = len(input_list1)
    length2 = len(input_list2)

    # char level same
    num_same = 0
    same_word_list = []
    # 1.compute percentage of same tokens
    for word1 in input_list1:
        for word2 in input_list2:
            if word1 == word2:
                num_same = num_same + 1
                same_word_list.append(word1)
                continue
    num_same_pert_min = float(num_same) / (float(max(length1, length2)) + 0.000001)
    num_same_pert_max = float(num_same) / (float(min(length1, length2)) + 0.000001)
    num_same_pert_avg = float(num_same) / ((float(length1 + length2) + 0.000001) / 2.0)

    # word level same

    input_list1_w = list(jieba.cut(input_string_x1))
    input_list2_w = list(jieba.cut(input_string_x2))
    length1_w = len(input_list1_w)
    length2_w = len(input_list2_w)

    num_same_w = 0
    same_word_list_w = []

    # compute the percentage of same tokens wordlevel
    for word1 in input_list1_w:
        for word2 in input_list2_w:
            if word1 == word2:
                num_same_w = num_same_w + 1
                same_word_list_w.append(word1)
                continue
    num_same_pert_min_w = float(num_same_w) / (float(max(length1_w, length2_w)) + 0.000001)
    num_same_pert_max_w = float(num_same_w) / (float(min(length1_w, length2_w)) + 0.000001)
    num_same_pert_avg_w = float(num_same_w) / ((float(length1_w + length2_w) + 0.000001) / 2.0)

    # 2.compute percentage of unique tokens in each string
    input_list1_unique = set([x for x in input_list1 if x not in same_word_list])
    input_list2_unique = set([x for x in input_list2 if x not in same_word_list])
    num_diff_x1 = float(len(input_list1_unique)) / (float(length1) + 0.000001)
    num_diff_x2 = float(len(input_list2_unique)) / (float(length2) + 0.000001)

    # compute percentage of unique tokens in each string word level

    input_list1_unique_w = set([x for x in input_list1_w if x not in same_word_list_w])
    input_list2_unique_w = set([x for x in input_list2_w if x not in same_word_list_w])
    num_diff_x1_w = float(len(input_list1_unique_w)) / (float(length1_w) + 0.000001)
    num_diff_x2_w = float(len(input_list2_unique_w)) / (float(length2_w) + 0.000001)

    # if index == 0:  # print debug message
    #     print("input_string_x1:", input_string_x1)
    #     print("input_string_x2:", input_string_x2)
    #     print("same_word_list:", same_word_list)
    #     print("input_list1_unique:", input_list1_unique)
    #     print("input_list2_unique:", input_list2_unique)
    #     print(
    #     "num_same:", num_same, ";length1:", length1, ";length2:", length2, ";num_same_pert_min:", num_same_pert_min,
    #     ";num_same_pert_max:", num_same_pert_max, ";num_same_pert_avg:", num_same_pert_avg,
    #     ";num_diff_x1:", num_diff_x1, ";num_diff_x2:", num_diff_x2)

    diff_overlap_list = [num_same_pert_min, num_same_pert_max, num_same_pert_avg, \
                         num_same_pert_min_w, num_same_pert_max_w, num_same_pert_avg_w, \
                         num_diff_x1, num_diff_x2, num_diff_x1_w, num_diff_x2_w]
    return diff_overlap_list


def edit(str1, str2):
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    # print("matrix:",matrix)
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]


def token_string_as_list(string, tokenize_style='char'):
    # string = string.decode("utf-8")
    string = string.replace("***", "*")
    length = len(string)
    if tokenize_style == 'char':
        listt = [string[i] for i in range(length)]
    elif tokenize_style == 'word':
        listt = jieba.lcut(string)  # cut_all=True
    # elif tokenize_style == 'pinyin':
    #     string = " ".join(jieba.lcut(string))
    #     listt = ''.join(lazy_pinyin(string)).split()  # list:['nihao', 'wo', 'de', 'pengyou']

    listt = [x for x in listt if x.strip()]
    return listt


def get_sentence_vector(word_vec_dict, tfidf_dict, word_list, tfidf_flag=True):
    vec_sentence = 0.0
    for word in word_list:
        # print("word:",word)
        if word in word_vec_dict:
            word_vec = word_vec_dict[word]
        else:
            word_vec = None
        word_tfidf = tfidf_dict.get(word, None)
        # print("word_vec:",word_vec,";word_tfidf:",word_tfidf)
        if word_vec is None or word_tfidf is None:
            continue
        else:
            if tfidf_flag == True:
                vec_sentence += word_vec * word_tfidf
            else:
                vec_sentence += word_vec * 1.0
    vec_sentence = vec_sentence / (np.sqrt(np.sum(np.power(vec_sentence, 2))) + 0.000001)
    return vec_sentence


def cos_distance_bag_tfidf(input_string_x1, input_string_x2, word_vec_dict, tfidf_dict, tfidf_flag=True):
    # print("input_string_x1:",input_string_x1)
    # 1.1 get word vec for sentence 1
    sentence_vec1 = get_sentence_vector(word_vec_dict, tfidf_dict, input_string_x1, tfidf_flag=tfidf_flag)
    # print("sentence_vec1:",sentence_vec1)
    # 1.2 get word vec for sentence 2
    sentence_vec2 = get_sentence_vector(word_vec_dict, tfidf_dict, input_string_x2, tfidf_flag=tfidf_flag)
    # print("sentence_vec2:", sentence_vec2)
    # 2 compute cos similiarity
    numerator = np.sum(np.multiply(sentence_vec1, sentence_vec2))
    denominator = np.sqrt(np.sum(np.power(sentence_vec1, 2))) * np.sqrt(np.sum(np.power(sentence_vec2, 2)))
    cos_distance = float(numerator) / float(denominator + 0.000001)

    # print("cos_distance:",cos_distance)
    manhattan_distance = np.sum(np.abs(np.subtract(sentence_vec1, sentence_vec2)))
    # print(manhattan_distance,type(manhattan_distance),np.isnan(manhattan_distance))
    if np.isnan(manhattan_distance): manhattan_distance = 300.0
    manhattan_distance = np.log(manhattan_distance + 0.000001) / 5.0

    canberra_distance = np.sum(np.abs(sentence_vec1 - sentence_vec2) / np.abs(sentence_vec1 + sentence_vec2 + 0.000001))
    if np.isnan(canberra_distance): canberra_distance = 300.0
    canberra_distance = np.log(canberra_distance + 0.000001) / 5.0

    minkowski_distance = np.power(np.sum(np.power((sentence_vec1 - sentence_vec2), 3)), 0.33333333)
    if np.isnan(minkowski_distance): minkowski_distance = 300.0
    minkowski_distance = np.log(minkowski_distance + 0.000001) / 5.0

    euclidean_distance = np.sqrt(np.sum(np.power((sentence_vec1 - sentence_vec2), 2)))
    if np.isnan(euclidean_distance): euclidean_distance = 300.0
    euclidean_distance = np.log(euclidean_distance + 0.000001) / 5.0

    return cos_distance, manhattan_distance, canberra_distance, minkowski_distance, euclidean_distance


def UniqueWord_distance_tfidf_char(input_string_x1,input_string_x2,char_vec_word2vec_dict,tfidf_dict_char):
    unique_list_sent1 = []
    unique_list_sent2 = []
    for w1 in input_string_x1:
        if w1 not in input_string_x2:
            unique_list_sent1.append(w1)
    for w2 in input_string_x2:
        if w2 not in input_string_x1:
            unique_list_sent2.append(w2)
    sentence_vec1 = 0.0
    sentence_vec2 = 0.0
    for w in unique_list_sent1:
        if w in char_vec_word2vec_dict and tfidf_dict_char:
            sentence_vec1 += char_vec_word2vec_dict[w]* tfidf_dict_char[w]
    for w in unique_list_sent2:
        if w in char_vec_word2vec_dict and tfidf_dict_char:
            sentence_vec2 += char_vec_word2vec_dict[w] * tfidf_dict_char[w]

    numerator = np.sum(np.multiply(sentence_vec1, sentence_vec2))
    denominator = np.sqrt(np.sum(np.power(sentence_vec1, 2))) * np.sqrt(np.sum(np.power(sentence_vec2, 2)))
    cos_distance = float(numerator) / float(denominator + 0.000001)

    # print("cos_distance:",cos_distance)
    manhattan_distance = np.sum(np.abs(np.subtract(sentence_vec1, sentence_vec2)))
    # print(manhattan_distance,type(manhattan_distance),np.isnan(manhattan_distance))
    if np.isnan(manhattan_distance): manhattan_distance = 300.0
    manhattan_distance = np.log(manhattan_distance + 0.000001) / 5.0

    canberra_distance = np.sum(np.abs(sentence_vec1 - sentence_vec2) / np.abs(sentence_vec1 + sentence_vec2 + 0.000001))
    if np.isnan(canberra_distance): canberra_distance = 300.0
    canberra_distance = np.log(canberra_distance + 0.000001) / 5.0

    minkowski_distance = np.power(np.sum(np.power((sentence_vec1 - sentence_vec2), 3)), 0.33333333)
    if np.isnan(minkowski_distance): minkowski_distance = 300.0
    minkowski_distance = np.log(minkowski_distance + 0.000001) / 5.0

    euclidean_distance = np.sqrt(np.sum(np.power((sentence_vec1 - sentence_vec2), 2)))
    if np.isnan(euclidean_distance): euclidean_distance = 300.0
    euclidean_distance = np.log(euclidean_distance + 0.000001) / 5.0

    return cos_distance, manhattan_distance, canberra_distance, minkowski_distance, euclidean_distance

def UniqueWord_distance_tfidf_bigram(input_string_x1,input_string_x2,bigram_vec_word2vec_dict,tfidf_dict_bigram):
    input_string_x1 = split_string_as_list_by_ngram(input_string_x1,ngram_value=2)
    input_string_x2 = split_string_as_list_by_ngram(input_string_x2,ngram_value=2)

    unique_list_sent1 = []
    unique_list_sent2 = []
    for w1 in input_string_x1:
        if w1 not in input_string_x2:
            unique_list_sent1.append(w1)
    for w2 in input_string_x2:
        if w2 not in input_string_x1:
            unique_list_sent2.append(w2)
    sentence_vec1 = 0.0
    sentence_vec2 = 0.0
    for w in unique_list_sent1:
        if w in bigram_vec_word2vec_dict and w in tfidf_dict_bigram:
            sentence_vec1 += bigram_vec_word2vec_dict[w] * tfidf_dict_bigram[w]
    for w in unique_list_sent2:
        if w in bigram_vec_word2vec_dict and w in tfidf_dict_bigram:
            sentence_vec2 += bigram_vec_word2vec_dict[w] * tfidf_dict_bigram[w]

    numerator = np.sum(np.multiply(sentence_vec1, sentence_vec2))
    denominator = np.sqrt(np.sum(np.power(sentence_vec1, 2))) * np.sqrt(np.sum(np.power(sentence_vec2, 2)))
    cos_distance = float(numerator) / float(denominator + 0.000001)

    # print("cos_distance:",cos_distance)
    manhattan_distance = np.sum(np.abs(np.subtract(sentence_vec1, sentence_vec2)))
    # print(manhattan_distance,type(manhattan_distance),np.isnan(manhattan_distance))
    if np.isnan(manhattan_distance): manhattan_distance = 300.0
    manhattan_distance = np.log(manhattan_distance + 0.000001) / 5.0

    canberra_distance = np.sum(np.abs(sentence_vec1 - sentence_vec2) / np.abs(sentence_vec1 + sentence_vec2 + 0.000001))
    if np.isnan(canberra_distance): canberra_distance = 300.0
    canberra_distance = np.log(canberra_distance + 0.000001) / 5.0

    minkowski_distance = np.power(np.sum(np.power((sentence_vec1 - sentence_vec2), 3)), 0.33333333)
    if np.isnan(minkowski_distance): minkowski_distance = 300.0
    minkowski_distance = np.log(minkowski_distance + 0.000001) / 5.0

    euclidean_distance = np.sqrt(np.sum(np.power((sentence_vec1 - sentence_vec2), 2)))
    if np.isnan(euclidean_distance): euclidean_distance = 300.0
    euclidean_distance = np.log(euclidean_distance + 0.000001) / 5.0

    return cos_distance, manhattan_distance, canberra_distance, minkowski_distance, euclidean_distance












def data_mining_features(index, input_string_x1, input_string_x2,
                         word_vec_word2vec_dict, tfidf_dict_word, \
                         char_vec_word2vec_dict, tfidf_dict_char, \
                         bigram_vec_word2vec_dict, tfidf_dict_bigram,n_gram=8):
    """
    get data mining feature given two sentences as string.
    1)n-gram similiarity(blue score);
    2) get length of questions, difference of length
    3) how many words are same, how many words are unique
    4) question 1,2 start with how/why/when(?????????????
    5?edit distance
    6) cos similiarity using bag of words
    :param input_string_x1:
    :param input_string_x2:
    :return:
    """
    input_string_x1 = input_string_x1
    input_string_x2 = input_string_x2
    feature_list = []
    # get sepcial word
    # com_list = CommanSpecialWord(input_string_x1,input_string_x2)
    # feature_list.extend(com_list)

    unique_compare = UniqueWord_distance_tfidf_char(input_string_x1,\
                                                    input_string_x2,char_vec_word2vec_dict,tfidf_dict_char)
    feature_list.extend(unique_compare)

    unique_compare_bigram = UniqueWord_distance_tfidf_bigram(input_string_x1,input_string_x2,\
                                                             bigram_vec_word2vec_dict,tfidf_dict_bigram)

    feature_list.extend(unique_compare_bigram)

    # 1. get blue score vector
    # get blue score with n-gram
    # for i in range(n_gram):
    #     x1_list = split_string_as_list_by_ngram(input_string_x1, i + 1)
    #     x2_list = split_string_as_list_by_ngram(input_string_x2, i + 1)
    #     blue_score_i_1 = compute_blue_ngram(x1_list, x2_list)
    #     blue_score_i_2 = compute_blue_ngram(x2_list, x1_list)
    #     feature_list.append(blue_score_i_1)
    #     feature_list.append(blue_score_i_2)

    # 2. get length of questions, difference of length
    # length1 = float(len(input_string_x1))
    # length2 = float(len(input_string_x2))
    # length_diff = (float(abs(length1 - length2))) / ((length1 + length2 + 0.000001) / 2.0)
    # feature_list.append(length_diff)

    # 3. how many words are same, how many words are unique
    sentence_diff_overlap_features_list = get_sentence_diff_overlap_pert(index, input_string_x1, input_string_x2)
    feature_list.extend(sentence_diff_overlap_features_list)

    # 4. question 1,2 start with how/why/when(?????????????
    # how_why_feature_list=get_special_start_token(input_string_x1,input_string_x2,special_start_token)
    # print("how_why_feature_list:",how_why_feature_list)
    # feature_list.extend(how_why_feature_list)
    x1_list_c = token_string_as_list(input_string_x1, tokenize_style='char')
    x2_list_c = token_string_as_list(input_string_x2, tokenize_style='char')
    distance_list_word2vec_char = cos_distance_bag_tfidf(x1_list_c, x2_list_c, char_vec_word2vec_dict, tfidf_dict_char)

    # 5.edit distance
    # edit_distance = float(edit(input_string_x1, input_string_x2)) / 30.0
    # feature_list.append(edit_distance)

    # 6.cos distance from sentence embedding
    x1_list = token_string_as_list(input_string_x1, tokenize_style='word')
    # x1_list = input_string_x1.split()
    x2_list = token_string_as_list(input_string_x2, tokenize_style='word')
    # x2_list = input_string_x2.split()
    distance_list_word2vec_word = cos_distance_bag_tfidf(x1_list, x2_list, word_vec_word2vec_dict, tfidf_dict_word)

    # distance_list2 = cos_distance_bag_tfidf(x1_list, x2_list, word_vec_fasttext_dict, tfidf_dict,tfidf_flag=False)
    # sentence_diffence=np.abs(np.subtract(sentence_vec_1,sentence_vec_2))
    # sentence_multiply=np.multiply(sentence_vec_1,sentence_vec_2)
    feature_list.extend(distance_list_word2vec_word)
    feature_list.extend(distance_list_word2vec_char)
    # feature_list.extend(list(sentence_diffence))
    # feature_list.extend(list(sentence_multiply))
    return feature_list




def CreateBatchFeat(df, char_vec_word2vec_dict, word_vec_word2vec_dict,\
                    bigram_vec_word2vec_dict,tfidf_dict_char, tfidf_dict_word,tfidf_dict_bigram):
    # tfidf_dict = get_tfidf_score_and_save(word_df)

    # print('training char word2vector ....')
    # w2i_c, i2w_c, weights_c, w2v_dic_c, maxlen_c = CharPik(train_c)
    # # ------------------------------------------------------
    # torch.save(w2i_c, model_dir + '/w2i_c')
    # torch.save(i2w_c, model_dir + '/i2w_c')
    # torch.save(weights_c, model_dir + '/weights_c')
    # torch.save(maxlen_c, model_dir + '/maxlen_c')
    # # ------------------------------------------------------
    # print('training word word2vector....')
    # w2i_w, i2w_w, weights_w, w2v_dic_w, maxlen_w = WordPik(train_w)
    #
    # torch.save(w2i_w, model_dir + '/w2i_w')
    # torch.save(i2w_w, model_dir + '/i2w_w')
    # torch.save(weights_w, model_dir + '/weights_w')
    # torch.save(maxlen_w, model_dir + '/maxlen_w')
    # ----------------------------------------------------------

    feat_list = []
    for i, (_, row) in enumerate(df.iterrows()):
        features_vector = data_mining_features(i, row['sent1'], row['sent2'], \
                                               word_vec_word2vec_dict, tfidf_dict_word, \
                                               char_vec_word2vec_dict, tfidf_dict_char, \
                                               bigram_vec_word2vec_dict,tfidf_dict_bigram,n_gram=8)
        feat_list.append(features_vector)
    feat_list = np.nan_to_num(feat_list).tolist()
    feat_tensor = torch.FloatTensor(feat_list).cuda()
    return feat_tensor


def Getlabel(df):
    result = []
    for item in df['label']:
        result.append(int(item))

    return torch.LongTensor(result).cuda()


def Evalue(y_pred, y_target):
    # with open('./dataset/test_pickle', 'rb') as f:
    #     y_pred = pickle.load(f)
    # valid = pd.read_csv('./dataset/valid.csv')
    # y_target = valid['label']
    def evalue(y_target, y_pred):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for y_t, y_p in zip(y_target, y_pred):
            if y_p == 1 and y_t == y_p:
                TP += 1
            elif y_p == 0 and y_t == y_p:
                TN += 1
            elif y_p == 1 and y_t != y_p:
                FP += 1
            elif y_p == 0 and y_t != y_p:
                FN += 1
        # F1-score = 2 * precision rate * recall rate / (precision rate + recall rate)
        precision_rate = TP / (TP + FP + 0.00001)
        recall_rate = TP / (TP + FN + 0.00001)
        return 2 * precision_rate * recall_rate / (precision_rate + recall_rate + 0.00001), precision_rate, recall_rate

    result, precision_rate, recall_rate = evalue(y_target, y_pred)
    print('f1 is {}'.format(result))
    print('precision_rate ={} '.format(precision_rate))
    print('recall_rate ={} '.format(recall_rate))
    return result, precision_rate, recall_rate




def eval(model, df_c, df_w, w2i_c, w2i_w, maxlen_c, maxlen_w, model_char, model_word,model_char_2gram,\
         tfidf_dict_c, tfidf_dict_w,tfidf_dict_bigram,maxlen_unique_c,maxlen_unique_w):
    with torch.no_grad():
        model.eval()
        y_p_float = []
        y_p_int = []
        y_t = []
        criterion_eval = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 1]).float().cuda())
        df_ix = range(len(df_c))
        ix_batch = DataLoader(df_ix, batch_size=batch_size, shuffle=False, drop_last=False)

        # for ix, (sen1,sen2,len1,len2,label) in enumerate(data_batch):
        df_loss = 0.0
        count_eval = 0
        for ix in ix_batch:
            count_eval += 1
            loss = 0.0

            sen1c, sen2c, len1c, len2c = CreateBatchTensorChar(w2i_c, maxlen_c, df_c.iloc[ix.numpy().tolist()])
            sen1w, sen2w, len1w, len2w = CreateBatchTensorWord(w2i_w, maxlen_w, df_w.iloc[ix.numpy().tolist()])
            uni_tensor_1, uni_tensor_2 = GetUnitFeat_char(df_c.iloc[ix.numpy().tolist()], w2i_c, maxlen_unique_c)
            uni_tensor_1_w, uni_tensor_2_w = GetUnitFeat_word(df_w.iloc[ix.numpy().tolist()], w2i_w,
                                                              maxlen_unique_w)


            # import ipdb
            # ipdb.set_trace()


            # feat_batch = CreateBatchFeat(df_c.iloc[ix.numpy().tolist()],model_char,model_word,model_char_2gram,\
            #                              tfidf_dict_c,tfidf_dict_w,tfidf_dict_bigram)

            label = Getlabel(df_c.iloc[ix.numpy().tolist()])
            output = model(sen1c, sen2c, sen1w, sen2w , uni_tensor_1,uni_tensor_2,uni_tensor_1_w,uni_tensor_2_w)

            # output = model(sen1c, sen2c, len1c, len2c, sen1w, sen2w, len1w, len2w)
            loss = criterion_eval(output, label)
            df_loss += loss.item()

            # output2 = model(sen2c, sen1c, sen2w, sen1w, feat_batch)
            # loss = criterion_eval(output2, label)
            # df_loss += loss.item()
            # output3 = (output + output2) / 2
            for item, item3 in zip(output[:,1], label):
                y_p_float.append(item.item())
                y_t.append(int(item3.item()))
        # y_p_float_sort = sorted(y_p_float, reverse=True)
        # thresh = y_p_float_sort[int(len(y_p_float_sort) * opt.alpha)]
        thresh = 0.3
        print('the thresh is {}'.format(thresh))
        for item in y_p_float:
            if torch.sigmoid(torch.tensor([item])) > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        df_loss = df_loss / count_eval

        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        model.train()

        return df_loss, f1, precision_rate, recall_rate


def train():
    patient = 3
    print('{} the fold started'.format(k_ix))
    print('count_inix =  {} '.format(count_inix))
    print('the batch is {}'.format(batch_size))
    print('the learning rate is {}'.format(opt.learning_rate))

    train_c, train_w, valid_c, valid_w = SplitData3(char_df, word_df, k_ix)

    # -------------------------------------------------------------------------------------
    if count_inix == 0:
        tfidf_dict_w = get_tfidf_score_and_save_word(train_w)
        torch.save(tfidf_dict_w, FILE_PATH + '/tfidf_dict_word{}'.format(k_ix))
        tfidf_dict_c = get_tfidf_score_and_save_char(train_c)
        torch.save(tfidf_dict_c, FILE_PATH + '/tfidf_dict_char{}'.format(k_ix))
        tfidf_dict_bigram = get_tfidf_score_and_save_bigram((train_c))
        torch.save(tfidf_dict_bigram, FILE_PATH + '/tfidf_dict_bigram{}'.format(k_ix))
    else:
        tfidf_dict_w = torch.load(FILE_PATH + '/tfidf_dict_word{}'.format(k_ix))
        print('pre trained word tfidf_dic_dic is used!')
        tfidf_dict_c = torch.load(FILE_PATH + '/tfidf_dict_char{}'.format(k_ix))
        print('pre trained char tfidf_dic_dic is used!')
        tfidf_dict_bigram = torch.load(FILE_PATH + '/tfidf_dict_bigram{}'.format(k_ix))
        print('pre trained tfidf_dict_bigram is used!')



    # -------------------------------------------------------------------------------------
    if count_inix == 0:
        w2i_c, i2w_c, weights_c, model_char, maxlen_c,maxlen_unique_c = CharPik(train_c)


        # ------------------------------------------------------
        torch.save(w2i_c, FILE_PATH + '/w2i_c{}'.format(k_ix))
        torch.save(i2w_c, FILE_PATH + '/i2w_c{}'.format(k_ix))
        torch.save(weights_c, FILE_PATH + '/weights_c{}'.format(k_ix))
        model_char.save(FILE_PATH + '/model_char{}'.format(k_ix))
        torch.save(maxlen_c, FILE_PATH + '/maxlen_c{}'.format(k_ix))
        torch.save(maxlen_unique_c, FILE_PATH + '/maxlen_unique{}'.format(k_ix))

    else:
        w2i_c = torch.load(FILE_PATH + '/w2i_c{}'.format(k_ix))
        i2w_c = torch.load(FILE_PATH + '/i2w_c{}'.format(k_ix))
        weights_c = torch.load(FILE_PATH + '/weights_c{}'.format(k_ix))
        model_char = gensim.models.Word2Vec.load(FILE_PATH + '/model_char{}'.format(k_ix))
        maxlen_c = torch.load(FILE_PATH + '/maxlen_c{}'.format(k_ix))
        maxlen_unique_c = torch.load(FILE_PATH + '/maxlen_unique{}'.format(k_ix))
        print('pre trained w2i_c is used!')
        print('pre trained i2w_c is used!')
        print('pre trained weights_c is used!')
        print('pre trained model_char is used!')
        print('pre trained maxlen_c is used!')

    # -------------------------------------------------------------------------------------
    if count_inix == 0:
        w2i_2gram, i2w_2gram, weights_2gram, model_char_2gram, maxlen_2gram = CharPikBigram(train_c)
        # ------------------------------------------------------
        torch.save(w2i_2gram, FILE_PATH + '/w2i_2gram{}'.format(k_ix))
        torch.save(i2w_2gram, FILE_PATH + '/i2w_2gram{}'.format(k_ix))
        torch.save(weights_2gram, FILE_PATH + '/weights_2gram{}'.format(k_ix))
        model_char_2gram.save(FILE_PATH + '/model_2gram{}'.format(k_ix))
        torch.save(maxlen_2gram, FILE_PATH + '/maxlen_2gram{}'.format(k_ix))
    else:
        w2i_2gram = torch.load(FILE_PATH + '/w2i_2gram{}'.format(k_ix))
        i2w_2gram = torch.load(FILE_PATH + '/i2w_2gram{}'.format(k_ix))
        weights_2gram = torch.load(FILE_PATH + '/weights_2gram{}'.format(k_ix))
        model_char_2gram = gensim.models.Word2Vec.load(FILE_PATH + '/model_2gram{}'.format(k_ix))
        maxlen_2gram = torch.load(FILE_PATH + '/maxlen_2gram{}'.format(k_ix))
        print('pre trained w2i_2gram is used!')
        print('pre trained i2w_2gram is used!')
        print('pre trained weights_2gram is used!')
        print('pre trained model_2gram is used!')
        print('pre trained maxlen_2gram is used!')

    # ------------------------------------------------------
    if count_inix == 0:
        print('training word word2vector....')
        w2i_w, i2w_w, weights_w, model_word, maxlen_w,maxlen_unique_w = WordPik(train_w)


        torch.save(w2i_w, FILE_PATH + '/w2i_w{}'.format(k_ix))
        torch.save(i2w_w, FILE_PATH + '/i2w_w{}'.format(k_ix))
        torch.save(weights_w, FILE_PATH + '/weights_w{}'.format(k_ix))
        model_word.save(FILE_PATH + '/model_word{}'.format(k_ix))
        torch.save(maxlen_w, FILE_PATH + '/maxlen_w{}'.format(k_ix))
        torch.save(maxlen_unique_w,FILE_PATH + '/maxlen_unique{}'.format(k_ix))

    else:
        w2i_w = torch.load(FILE_PATH + '/w2i_w{}'.format(k_ix))
        i2w_w = torch.load(FILE_PATH + '/i2w_w{}'.format(k_ix))
        weights_w = torch.load(FILE_PATH + '/weights_w{}'.format(k_ix))
        model_word = gensim.models.Word2Vec.load(FILE_PATH + '/model_word{}'.format(k_ix))
        maxlen_w = torch.load(FILE_PATH + '/maxlen_w{}'.format(k_ix))
        maxlen_unique_w = torch.load(FILE_PATH + '/maxlen_unique{}'.format(k_ix))
        print('pre trained w2i_w is used!')
        print('pre trained i2w_w is used!')
        print('pre trained weights_w is used!')
        print('pre trained model_word is used!')
        print('pre trained maxlen_w is used!')
        print('pre trained maxlen_unique is used!')

    weights_c.cuda()
    weights_w.cuda()

    # model = DecomseAtten(num_embeddings_c = len(i2w_c),embedding_size_c = opt.embedding_dims_char,\
    #                      num_embeddings_w = len(i2w_w),embedding_size_w = opt.embedding_dims_word,\
    #                      hidden_size_c = opt.hidden_dims_char,hidden_size_w =opt.hidden_dims_word,\
    #                      label_size = 1,weight_char = weights_c,weight_word = weights_w)







    model = DecomseAtten(num_embeddings_c=None, embedding_size_c=opt.embedding_dims_char, \
                         num_embeddings_w=None, embedding_size_w=opt.embedding_dims_word, \
                         hidden_size_c=opt.hidden_dims_char, hidden_size_w=opt.hidden_dims_word, \
                         label_size=1, weight_char=weights_c, weight_word=weights_w).cuda()

    if count_inix != 0:
        # encoder_model.load_state_dict(torch.load(FILE_PATH + '/BestF1-0fold.pkl'))
        # atten_model.load_state_dict(torch.load(FILE_PATH + '/BestF1-0fold.pkl'))
        model.load_state_dict(torch.load(FILE_PATH + '/LastModel-{}fold.pkl'.format(k_ix)))
        print('pre trained model is used!!!!')

    # if not os.path.exists(model_dir + '/BestLoss.pkl'):
    #     model.load_state_dict(torch.load(model_dir + '/BestLoss.pkl'))
    # weights_tensor = torch.tensor([0.5, 0.5]).cuda()
    # weights_label = {0: (0, 0), 1: (0, 0)}

    df_ix = range(len(train_c))
    ix_batch = DataLoader(df_ix, batch_size=batch_size, shuffle=True, drop_last=True)

    if count_inix == 0:
        BestF1 = 0.0
        BestLoss = 88888888
    else:
        check_list = torch.load(FILE_PATH + '/F1AndLoss-{}fold.pkl'.format(k_ix))
        BestF1 = check_list[0]
        BestLoss = check_list[1]

    for e in range(epoch):
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 1]).float().cuda())
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), \
                                     lr=opt.learning_rate)
        # optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.learning_rate)
        e_loss = 0
        print('{} the epoch'.format(e))
        count = 0
        for ix in ix_batch:
            count += 1
            loss = 0.0
            # if count > 2:
            #     break
            # optimizer_encoder.zero_grad()
            optimizer.zero_grad()
            # now = datetime.datetime.now()
            # print(now.strftime('%Y-%m-%d %H:%M:%S'))
            # print('creating char tensor.......')
            sen1c, sen2c, len1c, len2c = CreateBatchTensorChar(w2i_c, maxlen_c, train_c.iloc[ix.numpy().tolist()])
            # now = datetime.datetime.now()
            # print(now.strftime('%Y-%m-%d %H:%M:%S'))
            # print('creating word tensor.......')

            sen1w, sen2w, len1w, len2w = CreateBatchTensorWord(w2i_w, maxlen_w, train_w.iloc[ix.numpy().tolist()])
            uni_tensor_1_c,uni_tensor_2_c = GetUnitFeat_char(train_c.iloc[ix.numpy().tolist()], w2i_c, maxlen_unique_c)
            uni_tensor_1_w, uni_tensor_2_w = GetUnitFeat_word(train_w.iloc[ix.numpy().tolist()], w2i_w,maxlen_unique_w)


            # feat_batch = CreateBatchFeat(train_c.iloc[ix.numpy().tolist()], model_char, model_word,model_char_2gram,\
            #                              tfidf_dict_c,tfidf_dict_w,tfidf_dict_bigram)

            label = Getlabel(train_c.iloc[ix.numpy().tolist()])
            output = model(sen1c, sen2c, sen1w, sen2w,uni_tensor_1_c,uni_tensor_2_c,uni_tensor_1_w,uni_tensor_2_w)
            loss = criterion(output, label)
            if count % 20 == 0:
                print('the loss is {}'.format(loss.item()))

            e_loss += loss.item()
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            output = model(sen2c, sen1c, sen2w, sen1w,uni_tensor_2_c,uni_tensor_1_c,uni_tensor_2_w,uni_tensor_1_w)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()


        print('the {} epoch is {}'.format(e, e_loss / (count + 0.0000001)))


        df_loss, f1, precision_rate, recall_rate = eval(model, valid_c, \
                                                        valid_w, w2i_c, w2i_w, maxlen_c, \
                                                        maxlen_w, model_char, model_word,model_char_2gram, \
                                                        tfidf_dict_c, tfidf_dict_w,tfidf_dict_bigram,\
                                                        maxlen_unique_c,maxlen_unique_w)
        print('the {}th eval loss is {}'.format(e, df_loss))
        if df_loss < BestLoss:
            BestLoss = df_loss
            see_log[1] = 'best loss is {}'.format(BestLoss)
            torch.save(model.state_dict(), FILE_PATH + '/BestLoss-{}fold.pkl'.format(k_ix))
        if f1 > BestF1:
            BestF1 = f1
            see_log[0] = 'best f1 is {}'.format(BestF1)
            torch.save(model.state_dict(), FILE_PATH + '/BestF1-{}fold.pkl'.format(k_ix))
        if f1 < BestF1 and df_loss > BestLoss:
            patient -= 1
        else:
            patient = 3
        # if patient <=0:
        #     break
    torch.save(model.state_dict(), FILE_PATH + '/LastModel-{}fold.pkl'.format(k_ix))
    print('this part of epochs is done,tha latest modeldic is stored in the path : {}'.format('LastModel'))
    check_list = [0.0, 0.0]  # [F1,Loss]
    check_list[0] = BestF1
    check_list[1] = BestLoss
    torch.save(check_list, FILE_PATH + '/F1AndLoss-{}fold.pkl'.format(k_ix))
    print('the Value of Best F1 and Best Loss is stored in {}'.format('/F1AndLoss-{}fold.pkl'.format(k_ix)))


if __name__ == '__main__':
    print('trainable embedding !!!')
    batch_size = 256
    epoch = 1
    print('the char embeddnig size is {}'.format(opt.embedding_dims_char))
    print('the word embeddnig size is {}'.format(opt.embedding_dims_word))
    print('the batch size is {}'.format(batch_size))

    # model_dir = 'model_dir'
    FILE_PATH = model_dir + '/DecomposeAtten-cnn-unique'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(FILE_PATH):
        os.mkdir(FILE_PATH)
    # df1 = pd.read_csv('train-c.csv', sep='\t', names=['id', 'sent1', 'sent2', 'label'], encoding='utf-8').head(35000)
    # df2 = pd.read_csv('train-w.csv', sep='\t', names=['id', 'sent1', 'sent2', 'label'], encoding='utf-8').head(35000)
    print('the data lenth is {}'.format(len(df1)))
    # if not the first epoch
    # k_ix = df3.iloc[0,0]
    # count_inix = df3.iloc[0,1]

    # if the first epoch
    k_ix = 0
    count_inix = 0

    see_log = ['null', 'null']
    char_df = df1
    word_df = df2

    # train .......
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(FILE_PATH):
        os.mkdir(FILE_PATH)
    train()
    #  passing data to next .....
    IxAndCount = [0, 0]
    if (count_inix + 1) % 10 == 0:
        count_inix = 0
        IxAndCount[0] = k_ix + 1
        IxAndCount[1] = count_inix
    else:
        count_inix += 1
        IxAndCount[0] = k_ix
        IxAndCount[1] = count_inix

    topai(3, pd.DataFrame([IxAndCount]))
    topai(1, df1)
    topai(2, df2)
    topai(4, pd.DataFrame(see_log))

# DataProcess(df1,df2)
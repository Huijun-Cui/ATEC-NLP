# this is the first main file
import pandas as pd
import re
import gensim

import os

import torch

import torch.nn as nn
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.models as models
class Config(object):
    def __init__(self):
        self.num_embeddings = 0
        self.embedding_dims =120
        self.hidden_dims = 100
        self.learning_rate = 0.001
        self.comman_size = 0
        self.alpha = 0.5
        self.weight = None


opt = Config()

class DRCN(nn.Module):
    def __init__(self,opt):
        super(DRCN, self).__init__()
        self.num_embeddings = opt.num_embeddings
        self.embedding_size = opt.embedding_dims
        self.opt = opt
        self.comman_size = opt.comman_size
        if opt.weight is not None:
            print(' pre train embedding is used...........................')
            self.embedding = nn.Embedding.from_pretrained(opt.weight, freeze=True)
        else:
            self.embedding = nn.Embedding(self.num_embeddings, self.embedding_size)

        self.hidden_size = opt.hidden_dims

        self.lstm = nn.LSTM(input_size=self.embedding_size + 1,\
                            hidden_size=self.hidden_size,bidirectional=True)
        self.lstm_2nd = nn.LSTM(input_size= 2 * self.hidden_size + self.embedding_size + 1,\
                                hidden_size=self.hidden_size,bidirectional=True)
        self.projection = self._projection_layers(self.hidden_size*2,self.hidden_size)

        self.projection_2layer = self._projection_layers(self.hidden_size * 2, self.hidden_size)

        self.pooling_linear = nn.Linear(4* self.hidden_size + 2* opt.embedding_dims + 2,opt.embedding_dims)

        self.scale_cnn = nn.Conv2d(in_channels=self.hidden_size * 2,\
                                   out_channels= 3,kernel_size=(1,1))
        self.dense_net = nn.Sequential(*list(models.densenet121().children())[:-1])


        self.fc = nn.Linear(1024,2)


    def _projection_layers(self, input_dim, output_dim):
        mlp_layers = []
        mlp_layers.append(nn.Linear(
            input_dim, output_dim, bias=True))
        mlp_layers.append(nn.Dropout(p=0.2))
        return nn.Sequential(*mlp_layers)  # * used to unpack list
    def _pooling_linear(self,input_dim,output_dim):
        mlp_layers = []
        # mlp_layers.append(nn.Linear(
        #     input_dim, output_dim, bias=True))
        # mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(
            input_dim, output_dim, bias=True))
        mlp_layers.append(nn.Dropout(p=0.2))
        return nn.Sequential(*mlp_layers)  # * used to unpack list


    def initial(self,batch_size):
        return torch.zeros(2 * 1,batch_size,self.hidden_size).cuda(), \
                   torch.zeros(2 * 1, batch_size, self.hidden_size).cuda()
    def initial2layer(self,batch_size):
        return torch.zeros(2 * 1, batch_size, self.hidden_size).cuda(), \
               torch.zeros(2 * 1, batch_size, self.hidden_size).cuda()


    def pack_for_rnn_seq(self,inputs, lengths):
        """
        :param inputs: [T * B * D]
        :param lengths:  [B]
        :return:
        """
        _, sorted_index = lengths.sort(descending = True)

        '''
            Reverse to decreasing order
        '''

        s_inputs = inputs.permute(1,0,2)[sorted_index].permute(1,0,2)
        lengths_list = []
        reverse_indices = [0] * lengths.size(0)
        for j, i in enumerate(sorted_index):
            lengths_list.append(lengths[i].item())
            reverse_indices[i] = j
        packed_seq = pack_padded_sequence(s_inputs,lengths_list)

        return packed_seq, reverse_indices

    def unpack_from_rnn_seq(self,packed_seq, reverse_indices,max_len):
        unpacked_seq, _ = pad_packed_sequence(packed_seq,total_length=max_len)
        return  unpacked_seq.permute(1,0,2)[reverse_indices].permute(1,0,2)
        # return torch.cat(s_inputs_list, 1)




    def forward(self,sen1,sen2,len1,len2,comman_sen1,comman_sen2):



        batch_size = sen1.size()[0]
        max_len = sen1.size()[1]
        # ----------------------------------------------------------------------------------------------
        comman_sen1 = comman_sen1.unsqueeze(2)
        comman_sen2 = comman_sen2.unsqueeze(2)
        # ----------------------------------------------------------------------------------------------



        sen1_emb = self.embedding(sen1)
        sen1_emb = torch.cat([sen1_emb,comman_sen1],dim = 2).permute(1,0,2)
        pack_sen1,reverse_index_sen1 = self.pack_for_rnn_seq(sen1_emb,len1)
        ini_hidden, ini_cell = self.initial(batch_size=batch_size)
        lstm_output_packed_sen1, _ = self.lstm(pack_sen1, (ini_hidden, ini_cell))
        lstm_output_sen1 = self.unpack_from_rnn_seq(lstm_output_packed_sen1,\
                                                    reverse_indices=reverse_index_sen1,max_len = max_len)
        lstm_output_sen1 = lstm_output_sen1.permute(1,0,2)

        # ----------------------------------------------------------------------------------------------
        sen2_emb = self.embedding(sen2)
        sen2_emb = torch.cat([sen2_emb,comman_sen2],dim = 2).permute(1,0,2)
        pack_sen2,reverse_index_sen2 = self.pack_for_rnn_seq(sen2_emb,len2)
        ini_hidden, ini_cell = self.initial(batch_size=batch_size)
        lstm_output_packed_sen2, _ = self.lstm(pack_sen2, (ini_hidden, ini_cell))
        lstm_output_sen2 = self.unpack_from_rnn_seq(lstm_output_packed_sen2,\
                                                    reverse_indices=reverse_index_sen2,max_len = max_len)
        lstm_output_sen2 = lstm_output_sen2.permute(1,0,2)
        # --------------------------------------------------------------------------------------------
        sen1_atten = self.projection(lstm_output_sen1)
        #  B * len1 * hidden_size
        sen2_atten = self.projection(lstm_output_sen2)
        #  B * len2 * hidden_size
        score1 = torch.bmm(sen1_atten, torch.transpose(sen2_atten, 1, 2))
        # B * len1 * len2
        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()
        # B * len2 * len1

        prob1 = F.softmax(score1,dim = 2)
        # batch_size x len1 x len2
        prob2 = F.softmax(score2,dim = 2)
        #  batch_size x len2 x len1

        sent1_combine = torch.cat(
            (sen1_atten, torch.bmm(prob1, sen2_atten)), 2)
        # batch_size x len1 x (hidden_size x 2)
        sent2_combine = torch.cat(
            (sen2_atten, torch.bmm(prob2, sen1_atten)), 2)


        sent1_expand = sent1_combine.unsqueeze(2)
        sent2_expand = sent2_combine.unsqueeze(1)

        iter_tensor = sent1_expand * sent2_expand

        iter_tensor = iter_tensor.permute(0,3,1,2)
        feat_extrac = self.scale_cnn(iter_tensor)



        feat_extrac = self.dense_net(feat_extrac)



        feat_extrac = feat_extrac.view(batch_size,-1)

        output = self.fc(feat_extrac)

        return output












def CreateBatchTensorWord(w2i_w, maxlen_w, df):
    result_sen1 = []
    result_len1 = []
    result_sen2 = []
    result_len2 = []
    com_sen1 = []
    com_sen2 = []

    # pattern = re.compile(u'[^\u4e00-\u9fa5]')
    # item_lexical = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", item_lexical)
    for sen1, sen2 in zip(df['sent1'], df['sent2']):
        # sen1_drop = pattern.sub(r'', sen1)
        sen1_drop = sen1.replace('* * *', '*')
        # sen2_drop = pattern.sub(r'', sen2)
        sen2_drop = sen2.replace('* * *', '*')

        sen1_list = [0] * maxlen_w
        sen2_list = [0] * maxlen_w
        com1_tmp_list = [0] * maxlen_w
        com2_tmp_list = [0] * maxlen_w
        ix_1 = 0
        ix_2 = 0
        for w1 in sen1_drop.strip().split():
            if w1 in w2i_w and ix_1 < maxlen_w:
                sen1_list[ix_1] = w2i_w[w1]
                ix_1 += 1
            elif ix_1 < maxlen_w:
                sen1_list[ix_1] = 0 # padding with zero
                ix_1 += 1
        if ix_1 == 0:
            ix_1 += 1
        for w2 in sen2_drop.strip().split():
            if w2 in w2i_w and ix_2 < maxlen_w:
                sen2_list[ix_2] = w2i_w[w2]
                ix_2 += 1
            elif ix_2 < maxlen_w:
                sen2_list[ix_2] = 0 # padding with zero
                ix_2 += 1
        if ix_2 == 0:
            ix_2 += 1

        result_sen1.append(sen1_list)
        result_sen2.append(sen2_list)
        result_len1.append(ix_1)
        result_len2.append(ix_2)

        # ------------------------------------------------------------------------------------------------
        for ix, word1 in enumerate(sen1_list):
            if word1 in sen2_list:
                com1_tmp_list[ix] = 1
        for ix, word in enumerate(sen2_list):
            if word in sen1_list:
                com2_tmp_list[ix] = 1
        com_sen1.append(com1_tmp_list)
        com_sen2.append(com2_tmp_list)

    return torch.LongTensor(result_sen1).cuda(), torch.LongTensor(result_sen2).cuda(), \
           torch.tensor(result_len1).cuda(), \
           torch.tensor(result_len2).cuda(), torch.tensor(com_sen1).float().cuda(), \
           torch.tensor(com_sen2).float().cuda()


def CreateBatchTensorChar(w2i_c, maxlen_c, df):
    result_sen1 = []
    result_len1 = []
    result_sen2 = []
    result_len2 = []
    com_sen1 = []
    com_sen2 = []

    # pattern = re.compile(u'[^\u4e00-\u9fa5]')
    # item_lexical = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", item_lexical)
    for sen1, sen2 in zip(df['sent1'], df['sent2']):
        # sen1_drop = pattern.sub(r'', sen1)
        sen1_drop = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", sen1)
        sen1_drop = sen1_drop.replace('***', '*')
        # sen2_drop = pattern.sub(r'', sen2)

        sen2_drop = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", sen2)
        sen2_drop = sen2_drop.replace('***', '*')

        sen1_list = [0] * maxlen_c
        sen2_list = [0] * maxlen_c
        com1_tmp_list = [0] * maxlen_c
        com2_tmp_list = [0] * maxlen_c
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

        # ------------------------------------------------------------------------------------------------
        for ix, word1 in enumerate(sen1_list):
            if word1 in sen2_list:
                com1_tmp_list[ix] = 1
        for ix, word in enumerate(sen2_list):
            if word in sen1_list:
                com2_tmp_list[ix] = 1
        com_sen1.append(com1_tmp_list)
        com_sen2.append(com2_tmp_list)

    return torch.LongTensor(result_sen1).cuda(), torch.LongTensor(result_sen2).cuda(), \
           torch.tensor(result_len1).cuda(), \
           torch.tensor(result_len2).cuda(), torch.tensor(com_sen1).float().cuda(), \
           torch.tensor(com_sen2).float().cuda()




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
            if w in char_count_dict and char_count_dict[w] > 1:
                tmp.append(w)
            else:
                tmp.append('A_c')
                # print('meet the rare char :{}'.format(w.encode('utf-8')))
        if len(tmp) > maxlenth_c:
            maxlenth_c = len(tmp)
        sentences.append(tmp)
    word_rep = opt.embedding_dims
    model_char = gensim.models.Word2Vec(sentences, size=word_rep, min_count=1, iter=20)
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
    maxlenth_c = 40
    return w2i, i2w, weights, model_char, maxlenth_c


def WordPik(train_df):
    maxlenth_w = 0
    # pattern = re.compile(u'[^\u4e00-\u9fa5\s]')
    sen = pd.concat([train_df['sent1'], train_df['sent2']], ignore_index=True)
    sentences = []
    for item in sen:
        item_lexical = item.replace('* * *', '*')
        item_lexical = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*\s]', "", item_lexical)

        tmp = []
        for w in item_lexical.strip().split():
            tmp.append(w)
        if len(tmp) > maxlenth_w:
            maxlenth_w = len(tmp)
        sentences.append(tmp)
    word_rep = opt.embedding_dims
    model_word = gensim.models.Word2Vec(sentences, size=word_rep, min_count=1, iter=30)
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
    return w2i, i2w, weights, model_word, maxlenth_w






class Data(object):
    def __init__(self,sen1,sen2,len1,len2,comsen1,comsen2,label = None):
        self.sen1 = sen1
        self.sen2 = sen2
        self.len1 = len1
        self.len2 = len2
        self.comsen1 = comsen1
        self.comsen2 = comsen2
        self.label = label

    def __getitem__(self, item):
        if self.label is None:
            return self.sen1.cuda()[item],self.sen2.cuda()[item],self.len1.cuda()[item],self.len2.cuda()[item],\
                   self.comsen1.cuda()[item],self.comsen2.cuda()[item]
        else:
            return self.sen1.cuda()[item], self.sen2.cuda()[item], self.len1.cuda()[item], self.len2.cuda()[item], \
                   self.comsen1.cuda()[item], self.comsen2.cuda()[item],self.label.cuda()[item]
    def __len__(self):
        return len(self.len1)




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


def eval(model, df_w, w2i_w, maxlen_w):
    with torch.no_grad():
        model.eval()
        y_p_float = []
        y_p_int = []
        y_t = []
        criterion_eval = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 1]).float().cuda())
        df_ix = range(len(df_w))
        ix_batch = DataLoader(df_ix, batch_size=batch_size, shuffle=False, drop_last=False)
        df_loss = 0.0
        count_eval = 0
        for ix in ix_batch:
            count_eval += 1
            loss = 0.0

            sen1w, sen2w, len1w, len2w, com_sen1w, com_sen2w = \
                CreateBatchTensorChar(w2i_w, maxlen_w, df_w.iloc[ix.numpy().tolist()])
            label = Getlabel(df_w.iloc[ix.numpy().tolist()])
            output = model(sen1w, sen2w, len1w, len2w, com_sen1w, com_sen2w)

            loss = criterion_eval(output, label)
            df_loss += loss.item()

            output_soft = torch.nn.functional.softmax(output, dim=1)

            # print('*' * 50)
            # print('the out put is')
            # print(output[:5])
            # print('the output softmax is')
            # print(output_soft[:5])
            # print('the label is ')
            # print(label[:5])

            for item, item3 in zip(output_soft[:, 1], label):
                y_p_float.append(item.item())
                y_t.append(int(item3.item()))
        # y_p_float_sort = sorted(y_p_float, reverse=True)
        # thresh = y_p_float_sort[int(len(y_p_float_sort) * opt.alpha)]
        f1_list = []
        y_p_int = []
        thresh = 0.17
        print('the thresh is {}***********************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.18
        print('the thresh is {}***********************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.19
        print('the thresh is {}***********************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.20
        print('the thresh is {}***********************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.21
        print('the thresh is {}***********************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.22
        print('the thresh is {}***********************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.23
        print('the thresh is {}***********************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.24
        print('the thresh is {}***********************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.25
        print('the thresh is {}***********************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.26
        print('the thresh is {}***********************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.27
        print('the thresh is {}***********************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.28
        print('the thresh is {}***********************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.29
        print('the thresh is {}***********************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.3
        print('the thresh is {}***********************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)
        y_p_int = []
        thresh = 0.31
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)
        y_p_int = []
        thresh = 0.32
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)
        y_p_int = []
        thresh = 0.33
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)
        y_p_int = []
        thresh = 0.34
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)
        y_p_int = []
        thresh = 0.35
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)
        y_p_int = []
        thresh = 0.36
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)
        y_p_int = []
        thresh = 0.37
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)
        y_p_int = []
        thresh = 0.38
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)
        y_p_int = []
        thresh = 0.39
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.40
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.41
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.42
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.43
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.44
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.45
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.46
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.47
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.48
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.49
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)

        y_p_int = []
        thresh = 0.50
        print('the thresh is {}************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)
        f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
        f1_list.append(f1)







        f1 = sorted(f1_list)[-1]

        df_loss = df_loss / count_eval


        # df_c['y_p'] = y_p_int
        # df_c['y_t'] = y_t
        # df_c['y_p_float'] = y_p_float
        # df_c.to_csv('badcase.csv')
        model.train()

    return df_loss, f1, precision_rate, recall_rate


def train(train_c,train_w,valid_c,valid_w):
    patient = 3
    print('{} the fold started'.format(k_ix))
    print('count_inix =  {} '.format(count_inix))
    print('the batch is {}'.format(batch_size))
    print('the learning rate is {}'.format(opt.learning_rate))


    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    if count_inix == 0:
        w2i_c, i2w_c, weights_c, model_char, maxlen_c = CharPik(train_c)
        # ------------------------------------------------------
        torch.save(w2i_c, FILE_PATH + '/w2i_c{}'.format(k_ix))
        torch.save(i2w_c, FILE_PATH + '/i2w_c{}'.format(k_ix))
        torch.save(weights_c,FILE_PATH + '/weights_c{}'.format(k_ix))
        model_char.save(FILE_PATH + '/model_char{}'.format(k_ix))
        torch.save(maxlen_c, FILE_PATH + '/maxlen_c{}'.format(k_ix))
    else:
        w2i_c = torch.load(FILE_PATH + '/w2i_c{}'.format(k_ix))
        i2w_c = torch.load(FILE_PATH + '/i2w_c{}'.format(k_ix))
        weights_c = torch.load(FILE_PATH + '/weights_c{}'.format(k_ix))
        model_char = gensim.models.Word2Vec.load(FILE_PATH + '/model_char{}'.format(k_ix))
        maxlen_c = torch.load(FILE_PATH + '/maxlen_c{}'.format(k_ix))
        print('pre trained w2i_w is used!')
        print('pre trained i2w_w is used!')
        print('pre trained weights_w is used!')
        print('pre trained model_word is used!')
        print('pre trained maxlen_w is used!')

    # -------------------------------------------------------------------------------------

    weights_c.cuda()
    setattr(opt, 'weight', weights_c)
    setattr(opt,'num_embeddings',len(i2w_c))

    model = DRCN(opt).cuda()
    if count_inix != 0:
        # encoder_model.load_state_dict(torch.load(FILE_PATH + '/BestF1-0fold.pkl'))
        # atten_model.load_state_dict(torch.load(FILE_PATH + '/BestF1-0fold.pkl'))
        model.load_state_dict(torch.load(FILE_PATH + '/LastModel-{}fold.pkl'.format(k_ix)))
        print('pre trained model is used!!!!')

    # if not os.path.exists(model_dir + '/BestLoss.pkl'):
    #     model.load_state_dict(torch.load(model_dir + '/BestLoss.pkl'))
    # weights_tensor = torch.tensor([0.5, 0.5]).cuda()
    # weights_label = {0: (0, 0), 1: (0, 0)}



    if count_inix == 0:
        BestF1 = 0.0
        BestLoss = 88888888
    else:
        check_list = torch.load(FILE_PATH + '/F1AndLoss-{}fold.pkl'.format(k_ix))
        BestF1 = check_list[0]
        BestLoss = check_list[1]

    label = Getlabel(train_c)



    sen1, sen2, len1, len2, comsen1, comsen2 = \
        CreateBatchTensorChar(w2i_c, maxlen_c, train_c)
    # import ipdb
    # ipdb.set_trace()
    Data_contain = Data(sen1, sen2, len1,len2,comsen1,comsen2,label)
    Data_batch = DataLoader(Data_contain,shuffle=True,drop_last=False,batch_size = batch_size)




    for e in range(epoch):
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 1]).float().cuda())
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), \
                                     lr=opt.learning_rate)
        # optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.learning_rate)
        e_loss = 0
        print('{} the epoch'.format(e))
        count = 0
        for sen1w,sen2w,len1w,len2w,comsen1w,comsen2w,label in Data_batch:
            count += 1
            optimizer.zero_grad()



            output = model(sen1w, sen2w, len1w,len2w, comsen1w, comsen2w)
            # import ipdb
            # ipdb.set_trace()
            loss = criterion(output, label)
            if count % 20 == 0:
                print('the loss is {}'.format(loss.item()))

            e_loss += loss.clone().item()
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
            optimizer.step()

        print('the {} epoch is {}'.format(e, e_loss / (count + 0.0000001)))

        df_loss, f1, precision_rate, recall_rate = eval(model, valid_c, w2i_c, maxlen_c)
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
    batch_size = 256
    # model_dir = 'model_dir'
    FILE_PATH = model_dir + '/DIIN-final-char'


    # if not the first epoch
    # k_ix = df3.iloc[0,0]
    # count_inix = df3.iloc[0,1]

    # if the first epoch
    k_ix = 5
    count_inix = 0
    # char_df = df1
    # word_df = df2
    see_log = ['null', 'null']
    epoch = 6
    # train .......
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(FILE_PATH):
        os.mkdir(FILE_PATH)


    with open(model_dir + '/fold{}'.format(k_ix), 'rb') as f:
        train_c = torch.load(f)
        valid_c = torch.load(f)
        train_w = torch.load(f)
        valid_w = torch.load(f)

    # opt.alpha = float(len(df1[df1['label'] == 1])) / float(len(df1))
    # print('the alpha is {}'.format(opt.alpha))

    train(train_c,train_w,valid_c,valid_w)
    #  passing data to next .....
    IxAndCount = [0, 0]
    if (count_inix + 1) % 40 == 0:
        count_inix = 0
        IxAndCount[0] = k_ix + 1
        IxAndCount[1] = count_inix
    else:
        count_inix += 1
        IxAndCount[0] = k_ix
        IxAndCount[1] = count_inix

    topai(3, pd.DataFrame([IxAndCount]))
    # topai(1, df1)
    # topai(2, df2)
    topai(4, pd.DataFrame(see_log))
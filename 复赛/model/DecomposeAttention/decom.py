# this is the first main file

import torch.nn.functional as F
import pandas as pd
import re
import gensim
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import jieba

class ConfigChar(object):
    def __init__(self):
        self.num_embeddings = None
        self.hidden_dims = 70
        self.embedding_dims = 70
        self.alpha = 0.5
        self.weight = None
        self.epoch = 100
        self.batch_size = 128
        self.learning_rate = 0.001
opt = ConfigChar()

class DA(nn.Module):
    def __init__(self,opt):
        super(DA, self).__init__()
        self.hidden_dims = opt.hidden_dims
        self.emb_layer = nn.Embedding(num_embeddings=opt.num_embeddings,embedding_dim=opt.embedding_dims,padding_idx=0)
        self.projection = self._mlp_layers(opt.embedding_dims,self.hidden_dims)
        self.combined_linear = self._mlp_layers_combined(2 * self.hidden_dims,self.hidden_dims)
        self.afterpool = self._mlp_layers_afterpool(4 * self.hidden_dims,2*self.hidden_dims)
        self.fc = nn.Linear(2 * self.hidden_dims,2)

    def forward(self, sen1,sen2):
        sen1_emb = self.emb_layer(sen1)
        sen2_emb = self.emb_layer(sen2)

        sen1_project = self.projection(sen1_emb)
        sen2_project = self.projection(sen2_emb)

        score1_atten = torch.bmm(sen1_project,torch.transpose(sen2_project,1,2))
        score2_atten = torch.transpose(score1_atten.contiguous(),1,2).contiguous()

        prob1 = F.softmax(score1_atten,dim = 2)
        prob2 = F.softmax(score2_atten,dim = 2)

        sen1_combine = torch.cat([sen1_project,torch.bmm(prob1,sen2_project)],dim = 2)
        sen2_combine = torch.cat([sen2_project,torch.bmm(prob2,sen1_project)],dim = 2)

        sen1_vec = self.combined_linear(sen1_combine)
        sen2_vec = self.combined_linear(sen2_combine)

        sen1_pooling = torch.cat([sen1_vec.mean(dim = 1),sen1_vec.max(dim =1)[0]],dim = 1)
        sen2_pooling = torch.cat([sen2_vec.mean(dim=1), sen2_vec.max(dim=1)[0]], dim=1)

        vec = torch.cat([sen1_pooling,sen2_pooling],dim = 1)
        vec = self.afterpool(vec)
        return self.fc(vec)






    def _mlp_layers(self, input_dim, output_dim):
        mlp_layers = []
        mlp_layers.append(nn.Linear(
            input_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(p=0.2))
        return nn.Sequential(*mlp_layers)
    def _mlp_layers_combined(self,input_dim,output_dim):
        mlp_layers = []
        mlp_layers.append(nn.Linear(input_dim,input_dim))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(p = 0.2))
        mlp_layers.append(nn.Linear(input_dim,output_dim))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(p = 0.2))
        return nn.Sequential(*mlp_layers)
    def _mlp_layers_afterpool(self,input_dim,output_dim):
        mlp_layers = []
        mlp_layers.append(nn.BatchNorm1d(input_dim))
        mlp_layers.append(nn.Linear(input_dim,input_dim))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.BatchNorm1d(input_dim))
        mlp_layers.append(nn.Linear(input_dim,output_dim))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(p = 0.2))
        return nn.Sequential(*mlp_layers)










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
        # -------------------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------------------

        tmp = []
        for w in item_lexical:
            if w in char_count_dict and char_count_dict[w] > 3:
                tmp.append(w)
            else:
                tmp.append('R')
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
        weights.append(model_char[i2w[i + 1]])
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
    model_word = gensim.models.Word2Vec(sentences, size=word_rep, min_count=1, iter=100)
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


def SpitData2(df_char, df_word, k_ix):
    step = len(df_char) // 10
    train_list = []
    valid_list = []
    valid_list.extend(range(k_ix * step, (k_ix + 1) * step))

    for item in range(len(df_char)):
        if item not in valid_list:
            train_list.append(item)

    train_df_char = df_char.iloc[train_list]
    valid_df_char = df_char.iloc[valid_list]

    train_df_word = df_word.iloc[train_list]
    valid_df_word = df_word.iloc[valid_list]

    return train_df_char, train_df_word, \
           valid_df_char, valid_df_word.reset_index(drop=True)

def Evalue(y_pred,y_target):
    # with open('./dataset/test_pickle', 'rb') as f:
    #     y_pred = pickle.load(f)
    # valid = pd.read_csv('./dataset/valid.csv')
    # y_target = valid['label']
    def evalue(y_target,y_pred):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for y_t,y_p in zip(y_target,y_pred):
            if y_p == 1 and y_t == y_p:
                TP += 1
            elif y_p == 0 and y_t == y_p:
                TN +=1
            elif y_p == 1 and y_t != y_p:
                FP +=1
            elif y_p == 0 and y_t != y_p:
                FN +=1
        # F1-score = 2 * precision rate * recall rate / (precision rate + recall rate)
        precision_rate = TP / (TP + FP + 0.00001)
        recall_rate = TP / (TP + FN + 0.00001)
        return 2 * precision_rate * recall_rate / (precision_rate + recall_rate + 0.00001),precision_rate,recall_rate
    result,precision_rate,recall_rate = evalue(y_target,y_pred)
    print('f1 is {}'.format(result))
    print('precision_rate ={} '.format(precision_rate))
    print('recall_rate ={} '.format(recall_rate))
    return result,precision_rate,recall_rate




def CreateDataChar(df,w2i,i2w,maxlen):
    # Here i2w is used for debug
    data_s1 = []
    data_s2 = []
    len1 = []
    len2 = []
    for s1,s2 in zip(df['sent1'],df['sent2']):
        s1_tmp = [0] * maxlen
        s2_tmp = [0] * maxlen
        # ----------------------------------------------------------------------------
        ix = 0

        s1_drop = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", s1)

        s1_drop = s1_drop.replace('***', '*')
        # -----------------------------------------------------------------------------


        # -----------------------------------------------------------------------------

        for w in s1_drop:
            if w in w2i and ix < maxlen:
                s1_tmp[ix] = w2i[w]
                ix +=1
            elif ix < maxlen:
                s1_tmp[ix] = w2i['R']
                ix +=1
        len1.append(ix)
        # ----------------------------------------------------------------------------
        ix = 0

        s2_drop = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", s2)

        s2_drop = s2_drop.replace('***', '*')
        # -----------------------------------------------------------------------------

        # -----------------------------------------------------------------------------
        for w in s2_drop:
            if w in w2i and ix < maxlen:
                s2_tmp[ix] = w2i[w]
                ix +=1
            elif ix < maxlen:
                s2_tmp[ix] = w2i['R']
                ix +=1
        len2.append(ix)
        data_s1.append(s1_tmp)
        data_s2.append(s2_tmp)
    return torch.LongTensor(data_s1).cuda(),torch.LongTensor(data_s2).cuda(),\
           torch.tensor(len1).cuda(),torch.tensor(len2).cuda()

def SplitData(df_char, df_word, k_ix):
    step = len(df_char) // 10
    train_list = []
    valid_list = []
    valid_list.extend(range(k_ix * step, (k_ix + 1) * step))

    for item in range(len(df_char)):
        if item not in valid_list:
            train_list.append(item)

    train_df_char = df_char.iloc[train_list]
    valid_df_char = df_char.iloc[valid_list]

    train_df_word = df_word.iloc[train_list]
    valid_df_word = df_word.iloc[valid_list]

    return train_df_char, train_df_word, \
           valid_df_char, valid_df_word.reset_index(drop=True)



class Data(object):
    def __init__(self,sen1,sen2,len1,len2,label = None):
        self.sen1 = sen1
        self.sen2 = sen2
        self.len1 = len1
        self.len2 = len2
        self.label = label
    def __getitem__(self, item):
        if self.label is None:
            return self.sen1[item],self.sen2[item],self.len1[item],self.len2[item]
        else:
            return self.sen1[item], self.sen2[item], self.len1[item], self.len2[item],self.label[item]
    def __len__(self):
        return len(self.sen1)


def GetLabel(df):
    result = []
    for item in df['label']:
        result.append(int(item))

    return torch.LongTensor(result).cuda()
def eval(model,valid_c,w2i_c,i2w_c,maxlen):
    with torch.no_grad():

        model.eval()

        y_p_float = []
        y_p_int = []
        y_t = []

        sen1,sen2,len1,len2 = CreateDataChar(valid_c, w2i_c, i2w_c, maxlen)

        label = GetLabel(valid_c)

        DataBox = Data(sen1, sen2, len1, len2, label)

        data_obj = DataLoader(DataBox, batch_size=opt.batch_size, shuffle=True, drop_last=True)

        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 1]).float().cuda())

        loss_sum = 0.0

        count_batch = 0

        for ix, (sen1, sen2, len1, len2, label) in enumerate(data_obj):

            count_batch +=1

            output = model(*(sen1, sen2, len1, len2))

            loss = criterion(output, label)

            loss_sum += loss.item()

            output_soft = torch.nn.functional.softmax(output,dim = 1)

            for item1,item2 in zip(output_soft[:,-1],label):
                y_p_float.append(item1)
                y_t.append(int(item2.item()))
        loss_sum = loss_sum / (count_batch+ 0.0000000001)

        y_p_int = []
        thresh = 0.3
        print('the thrsh is {} ************************************************'.format(thresh))
        for item in y_p_float:
            if item > thresh:
                y_p_int.append(1)
            else:
                y_p_int.append(0)

        f1,precision_rate,rescall_rate = Evalue(y_p_int,y_t)

    model.train()
    return loss_sum,f1,precision_rate,rescall_rate














def train(k_ix,count_inix):
    print('{} the fold started'.format(k_ix))
    print('count_inix =  {} '.format(count_inix))
    print('the batch is {}'.format(opt.batch_size))
    print('the learning rate is {}'.format(opt.learning_rate))


    print('Is spliting the data .........')
    train_c, train_w, valid_c, valid_w = SplitData(char_df, word_df, k_ix)


    print('Is training the word2vec and getting the w2i,i2w,weights,maxlen')
    w2i_c, i2w_c, weights_c, model_char, maxlen_c = CharPik(train_c)

    setattr(opt,'weight',weights_c)

    setattr(opt, 'num_embeddings', len(i2w_c))

    # create tensor data from DataFrame
    sen1,sen2,len1,len2 = CreateDataChar(train_c,w2i_c,i2w_c,maxlen_c)

    label = GetLabel(train_c)
    # create data contian box

    DataBox = Data(sen1,sen2,len1,len2,label)

    model = DA(opt).cuda()

    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 1]).float().cuda())

    e = opt.epoch

    for i in range(e):
        print('The {} th epoch----------------------'.format(i))
        eloss = 0.0

        data_obj = DataLoader(DataBox, batch_size=opt.batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), \
                                     lr=opt.learning_rate)
        for ix,(sen1,sen2,len1,len2,label) in enumerate(data_obj):
            # import ipdb
            # ipdb.set_trace()

            optimizer.zero_grad()

            output = model(*(sen1,sen2))

            loss = criterion(output,label)

            eloss +=loss

            loss.backward()

            optimizer.step()

            if ix % 20 == 0:
                print('The batch loss is {}'.format(loss.item()))
        loss_sum, f1, precision_rate, rescall_rate = eval(model,valid_c,w2i_c,i2w_c,maxlen_c)
        print('the eval loss is {}'.format(loss_sum))
if __name__ == '__main__':
    print('the char embeddnig size is {}'.format(opt.embedding_dims))
    print('the batch size is {}'.format(opt.batch_size))

    model_dir = 'model_dir'
    FILE_PATH = model_dir + '/Siamese-gpu'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(FILE_PATH):
        os.mkdir(FILE_PATH)
    df1 = pd.read_csv('train-c.csv', sep='\t', names=['id', 'sent1', 'sent2', 'label'], encoding='utf-8').head(35000)
    df2 = pd.read_csv('train-w.csv', sep='\t', names=['id', 'sent1', 'sent2', 'label'], encoding='utf-8').head(35000)

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
    train(k_ix,count_inix)
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














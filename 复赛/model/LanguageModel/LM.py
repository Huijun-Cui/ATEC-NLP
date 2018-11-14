# encoding=utf8
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
        self.batch_size =256
        self.learning_rate = 0.001
opt = ConfigChar()
class LM(nn.Module):
    def __init__(self,opt):
        super(LM, self).__init__()
        self.num_embeddings = opt.num_embeddings
        self.hidden_dims = opt.hidden_dims
        self.embedding_dims = opt.embedding_dims
        self.emb_layer = nn.Embedding(num_embeddings=self.num_embeddings,embedding_dim= self.embedding_dims)
        self.lstm = nn.LSTM(input_size=self.embedding_dims,hidden_size=self.hidden_dims)
        self.out_layer = nn.Linear(self.hidden_dims,self.num_embeddings)
        self.act = nn.Softmax()
    def forward(self, sen,len):
        batch_size = sen.size()[0]
        max_len = sen.size()[1]
        emb_sen = self.emb_layer(sen).permute(1,0,2)

        packed_seq, reverse_index_sen = self.pack_for_rnn_seq(emb_sen, len)

        h, c = self.initial(batch_size)

        output_packed, _ = self.lstm(packed_seq, (h, c))

        output_padded = self.unpack_from_rnn_seq(output_packed,reverse_indices=reverse_index_sen,max_len=max_len)
        outout = self.out_layer(output_padded)

        return outout

    def initial(self,batch_size):
        return torch.zeros(1,batch_size,self.hidden_dims).cuda(),\
               torch.zeros(1,batch_size,self.hidden_dims).cuda()

    def pack_for_rnn_seq(self,inputs, lengths):
        """
        :param inputs: [T * B * D]
        :param lengths:  [B]
        :return:
        """
        _, sorted_indices = lengths.sort()
        '''
            Reverse to decreasing order
        '''
        r_index = reversed(list(sorted_indices))

        s_inputs_list = []
        lengths_list = []
        reverse_indices = np.zeros(lengths.size(0), dtype=np.int64)

        for j, i in enumerate(r_index):
            s_inputs_list.append(inputs[:, i, :].unsqueeze(1))
            try:
                lengths_list.append(lengths[i].item())
            except:
                print('test')
            reverse_indices[i] = j

        reverse_indices = list(reverse_indices)

        s_inputs = torch.cat(s_inputs_list, 1)
        packed_seq = pack_padded_sequence(s_inputs,lengths_list)

        return packed_seq, reverse_indices

    def unpack_from_rnn_seq(self,packed_seq, reverse_indices,max_len):
        unpacked_seq, _ = pad_packed_sequence(packed_seq,total_length=max_len)
        s_inputs_list = []

        for i in reverse_indices:
            s_inputs_list.append(unpacked_seq[:, i, :].unsqueeze(1))
        return torch.cat(s_inputs_list, 1)








def CharPik(train_df):
    count_addr = 0
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
                # print(item_lexical)
        for w in item_lexical:
            sum_count += 1
            if w not in char_count_dict:
                char_count_dict[w] = 0
            char_count_dict[w] += 1
    print('{} times addr names happend and replaced with D character'.format(count_addr))
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
    data_s1_tar = []
    data_s2_tar = []
    len1 = []
    len2 = []
    for s1,s2 in zip(df['sent1'],df['sent2']):
        s1_tmp = [0] * maxlen
        s2_tmp = [0] * maxlen
        s1_target_tmp = [0] * maxlen
        s2_target_tmp = [0] * maxlen
        # ----------------------------------------------------------------------------
        ix = 0

        s1_drop = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", s1)

        s1_drop = s1_drop.replace('***', '*')
        # -----------------------------------------------------------------------------
        # delete addr list
        # -----------------------------------------------------------------------------

        for w in s1_drop:
            if w in w2i and ix < maxlen:
                s1_tmp[ix] = w2i[w]
                ix +=1
            elif ix < maxlen:
                s1_tmp[ix] = w2i['R']
                ix +=1

        # create target sentence corresponding to sen1_tmp
        for ix_t in range(maxlen-1):
            s1_target_tmp[ix_t] = s1_tmp[ix_t + 1]
        s1_target_tmp[-1] = 0
        data_s1_tar.append(s1_target_tmp)

        len1.append(ix)
        # ----------------------------------------------------------------------------
        ix = 0

        s2_drop = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", s2)

        s2_drop = s2_drop.replace('***', '*')
        # -----------------------------------------------------------------------------
        # delete addr list
        # -----------------------------------------------------------------------------
        for w in s2_drop:
            if w in w2i and ix < maxlen:
                s2_tmp[ix] = w2i[w]
                ix +=1
            elif ix < maxlen:
                s2_tmp[ix] = w2i['R']
                ix +=1
        # create target sentence corresponding to sen2_tmp
        for ix_t in range(maxlen-1):
            s2_target_tmp[ix_t] = s2_tmp[ix_t + 1]
        s2_target_tmp[-1] = 0
        data_s2_tar.append(s2_target_tmp)





        len2.append(ix)
        data_s1.append(s1_tmp)
        data_s2.append(s2_tmp)
    return torch.LongTensor(data_s1).cuda(),torch.LongTensor(data_s2).cuda(),\
           torch.tensor(len1).cuda(),torch.tensor(len2).cuda(),\
           torch.tensor(data_s1_tar).cuda(),torch.tensor(data_s2_tar).cuda()

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
    def __init__(self,sen1,sen2,len1,len2,sen1_tar,sen2_tar,label = None):
        self.sen1 = sen1
        self.sen2 = sen2
        self.len1 = len1
        self.len2 = len2
        self.sen1_tar = sen1_tar
        self.sen2_tar = sen2_tar
        self.label = label
    def __getitem__(self, item):
        if self.label is None:
            return self.sen1[item],self.sen2[item],self.len1[item],\
                   self.len2[item],self.sen1_tar[item],self.sen2_tar[item]
        else:
            None
    def __len__(self):
        return len(self.len1)


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

        sen1,sen2,len1,len2,sen1_tar,sen2_tar = CreateDataChar(valid_c, w2i_c, i2w_c, maxlen)
        # import ipdb
        # ipdb.set_trace()

        # label = GetLabel(valid_c)

        DataBox = Data(sen1, sen2, len1, len2, sen1_tar,sen2_tar)

        data_obj = DataLoader(DataBox, batch_size=opt.batch_size, shuffle=False, drop_last=True)

        criterion = torch.nn.CrossEntropyLoss()

        loss_sum = 0.0
        count_batch = 0
        for ix, (sen1, sen2, len1, len2, sen1_tar,sen2_tar) in enumerate(data_obj):
            count_batch +=1
            output = model(*(sen1,len1))
            # if ix == 0:
            #     output_see = output.contiguous().permute(1,0,2).max(dim = 2)[1]
            #     s = ' '.encode('ascii', 'ignore').decode('ascii')
            #
            #     for item in output_see[0]:
            #        s +=  i2w_c[item.item()].encode('ascii', 'ignore').decode('ascii') + ' '
            #     print(s)
            #     s_t = ' '
            #     for item in sen1_tar[0]:
            #         s_t += i2w_c[item] + ' '
            #     print(s_t)

            loss = criterion(output.permute(1,0,2).contiguous().view(-1, opt.num_embeddings),\
                             sen1_tar.view(-1))

            loss_sum += loss.item()

        loss_sum = loss_sum / (count_batch+ 0.0000000001)

        print('the eval loos is {}'.format(loss_sum))

    model.train()
    return loss_sum













def train(k_ix,count_inix):
    print('{} the fold started'.format(k_ix))
    print('count_inix =  {} '.format(count_inix))
    print('the batch is {}'.format(opt.batch_size))
    print('the learning rate is {}'.format(opt.learning_rate))


    print('Is spliting the data .........')
    train_c, train_w, valid_c, valid_w = SplitData(char_df, word_df, k_ix)

    print('getting the addr list ....')
    # addr_list = get_addr_list()
    # print(' '.join(addr_list))
    print('Is training the word2vec and getting the w2i,i2w,weights,maxlen')
    w2i_c, i2w_c, weights_c, model_char, maxlen_c = CharPik(train_c)
    setattr(opt,'weight',weights_c)
    setattr(opt,'num_embeddings',len(i2w_c))

    # create tensor data from DataFrame
    sen1,sen2,len1,len2,sen1_tar,sen2_tar = CreateDataChar(train_c,w2i_c,i2w_c,maxlen_c)

    # label = GetLabel(train_c)
    # create data contian box

    DataBox = Data(sen1,sen2,len1,len2,sen1_tar,sen2_tar)

    model = LM(opt).cuda()

    criterion = torch.nn.CrossEntropyLoss()

    e = opt.epoch
    BestLoss = 888888
    for i in range(e):
        print('The {} th epoch----------------------'.format(i))
        eloss = 0.0

        data_obj = DataLoader(DataBox, batch_size=opt.batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), \
                                     lr=opt.learning_rate)
        for ix,(sen1,sen2,len1,len2,sen1_tar,sen2_tar) in enumerate(data_obj):
            # import ipdb
            # ipdb.set_trace()

            optimizer.zero_grad()

            output = model(*(sen1,len1))

            loss = criterion(output.permute(1,0,2).contiguous().view(-1,opt.num_embeddings),\
                             sen1_tar.contiguous().view(-1))

            eloss +=loss.item()

            loss.backward()

            optimizer.step()
            # ----------------------------------------------------------------
            optimizer.zero_grad()

            output = model(*(sen2, len2))

            loss = criterion(output.permute(1,0,2).contiguous().view(-1,opt.num_embeddings), \
                             sen2_tar.contiguous().view(-1))

            eloss += loss.item()

            loss.backward()

            optimizer.step()


            if ix % 20 == 0:
                print('The batch loss is {}'.format(loss.item()))
        print('{} the epoch loss is {}'.format(i,eloss / 2 / (ix + 0.00000001)))
        loss_eval = eval(model,valid_c,w2i_c,i2w_c,maxlen_c)
        if loss_eval < BestLoss:
            torch.save(model.state_dict(), 'bestlml.pkl')


if __name__ == '__main__':
    print('the char embeddnig size is {}'.format(opt.embedding_dims))
    print('the batch size is {}'.format(opt.batch_size))

    model_dir = 'model_dir'
    FILE_PATH = model_dir + '/Siamese-gpu'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(FILE_PATH):
        os.mkdir(FILE_PATH)
    df1 = pd.read_csv('train-c.csv', sep='\t', names=['id', 'sent1', 'sent2', 'label'], encoding='utf-8').head(28000)
    df2 = pd.read_csv('train-w.csv', sep='\t', names=['id', 'sent1', 'sent2', 'label'], encoding='utf-8').head(28000)

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














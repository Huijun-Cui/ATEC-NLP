import torch
from torch import nn
import os
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
import datetime





class Config(object):
    def __init__(self):
        self.embedding_dims_word = 64
        self.hidden_dims_word = 64
        self.embedding_dims_char = 64
        self.hidden_dims_char = 64
        self.learning_rate = 0.001
        self.alpha = 0.3
        self.mlp_hidden_dim = 100


opt = Config()




class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, dropout_prob):
        super(MLP,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        mlp_layers = []
        for i in range(num_layers):
            if i == 0:
                layer_input_dim = input_dim
            else:
                layer_input_dim = hidden_dim
            linear_layer = nn.Linear(in_features=layer_input_dim,
                                     out_features=hidden_dim)
            relu_layer = nn.ReLU()
            dropout_layer = nn.Dropout(dropout_prob)
            mlp_layer = nn.Sequential(linear_layer, relu_layer, dropout_layer)
            mlp_layers.append(mlp_layer)
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, input):
        """
        Args:
            input (Variable): A float variable of size
                (batch_size, input_dim).
        Returns:
            output (Variable): A float variable of size
                (batch_size, hidden_dim), which is the result of
                applying MLP to the input argument.
        """

        return self.mlp(input)


class NLIClassifier(nn.Module):

    def __init__(self, sentence_dim, hidden_dim, num_layers, num_classes,
                 dropout_prob):
        super(NLIClassifier,self).__init__()
        self.sentence_dim = sentence_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob

        self.mlp = MLP(input_dim=4 * sentence_dim, hidden_dim=hidden_dim,
                       num_layers=num_layers, dropout_prob=dropout_prob)
        self.clf_linear = nn.Linear(in_features=hidden_dim,
                                    out_features=num_classes)

    def forward(self, pre, hyp):
        mlp_input = torch.cat([pre, hyp, (pre - hyp).abs(), pre * hyp], dim=1)
        mlp_output = self.mlp(mlp_input)
        output = self.clf_linear(mlp_output)
        return output


class ShortcutStackedEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dims):
        super(ShortcutStackedEncoder,self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

        for i in range(self.num_layers):
            lstm_input_dim = input_dim + 2*sum(hidden_dims[:i])
            lstm_layer = nn.LSTM(
                input_size=lstm_input_dim, hidden_size=hidden_dims[i],
                bidirectional=True, batch_first=False)
            setattr(self, 'lstm_layer_{}'.format(i), lstm_layer)

    def get_lstm_layer(self, i):
        return getattr(self, 'lstm_layer_{}'.format(i))

    def forward(self, input, lengths):
        prev_lstm_output = None
        max_lenth = input.size()[0]
        lstm_input = input
        for i in range(self.num_layers):
            if i > 0:
                lstm_input = torch.cat([lstm_input, prev_lstm_output], dim=2)
            lstm_input_packed, reverse_indices = pack_for_rnn_seq(
                inputs=lstm_input, lengths=lengths)
            lstm_layer = self.get_lstm_layer(i)
            lstm_output_packed, _ = lstm_layer(lstm_input_packed)
            lstm_output = unpack_from_rnn_seq(
                packed_seq=lstm_output_packed, reverse_indices=reverse_indices,max_len=max_lenth)
            prev_lstm_output = lstm_output
        sentence_vector = torch.max(prev_lstm_output, dim=0)[0]
        return sentence_vector

class NLIModel(nn.Module):

    def __init__(self, num_words, word_dim, lstm_hidden_dims,
                 mlp_hidden_dim, mlp_num_layers, num_classes, dropout_prob,weights_c = None):
        super(NLIModel,self).__init__()
        self.num_words = num_words
        self.word_dim = word_dim
        self.lstm_hidden_dims = lstm_hidden_dims
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_num_layers = mlp_num_layers
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob


        if weights_c is not None:
            print('char pre train is used...........................')
            self.word_embedding = nn.Embedding.from_pretrained(weights_c, freeze=True)
        else:
            self.word_embedding = nn.Embedding(num_embeddings=num_words,
                                               embedding_dim=word_dim)
        self.encoder = ShortcutStackedEncoder(
            input_dim=word_dim, hidden_dims=lstm_hidden_dims)
        self.classifier = NLIClassifier(
            sentence_dim=2 * lstm_hidden_dims[-1], hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers, num_classes=num_classes,
            dropout_prob=dropout_prob)

    def forward(self, pre_input, pre_lengths, hyp_input, hyp_lengths):
        """
        Args:
            pre_input (Variable): A long variable containing indices for
                premise words. Size: (max_length, batch_size).
            pre_lengths (Tensor): A long tensor containing lengths for
                sentences in the premise batch.
            hyp_input (Variable): A long variable containing indices for
                hypothesis words. Size: (max_length, batch_size).
            pre_lengths (Tensor): A long tensor containing lengths for
                sentences in the hypothesis batch.
        Returns:
            output (Variable): A float variable containing the
                unnormalized probability for each class
        :return:
        """
        pre_input = pre_input.permute(1,0)
        hyp_input = hyp_input.permute(1,0)
        pre_input_emb = self.word_embedding(pre_input)
        hyp_input_emb = self.word_embedding(hyp_input)
        pre_vector = self.encoder(input=pre_input_emb, lengths=pre_lengths)
        hyp_vector = self.encoder(input=hyp_input_emb, lengths=hyp_lengths)
        classifier_output = self.classifier(pre=pre_vector, hyp=hyp_vector)
        # return classifier_output
        return pre_vector,hyp_vector
class WordCharShortCut(nn.Module):
    def __init__(self,num_words_c, word_dim_c, lstm_hidden_dims_c, \
                 num_words_w, word_dim_w, lstm_hidden_dims_w,
                 mlp_hidden_dim, mlp_num_layers, num_classes, dropout_prob,weights_c = None, weights_w=None
                 ):
        super(WordCharShortCut, self).__init__()
        self.charM = NLIModel(num_words_c,word_dim_c,lstm_hidden_dims_c,mlp_hidden_dim,mlp_num_layers,\
                             num_classes,dropout_prob,weights_c)
        self.wordM = NLIModel(num_words_w,word_dim_w,lstm_hidden_dims_w,mlp_hidden_dim,mlp_num_layers,\
                             num_classes,dropout_prob,weights_w)
        self.classifier = NLIClassifier(
            sentence_dim=2 * lstm_hidden_dims_c[-1] + 2* lstm_hidden_dims_w[-1], hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers, num_classes=num_classes,
            dropout_prob=dropout_prob)

    def forward(self, sen1c,sen2c,len1c,len2c,sen1w,sen2w,len1w,len2w):
        sen1c_o,sen2c_o = self.charM(sen1c,len1c,sen2c,len2c)
        sen1w_o,sen2w_o = self.wordM(sen1w,len1w,sen2w,len2w)

        pre = torch.cat([sen1c_o,sen1w_o],dim = 1)
        hyp = torch.cat([sen2c_o,sen2w_o],dim = 1)
        output = self.classifier(pre,hyp)
        return output









def pack_for_rnn_seq(inputs, lengths):
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
        lengths_list.append(lengths[i])
        reverse_indices[i] = j

    reverse_indices = list(reverse_indices)

    s_inputs = torch.cat(s_inputs_list, 1)
    packed_seq = pack_padded_sequence(s_inputs, lengths_list)

    return packed_seq, reverse_indices


def unpack_from_rnn_seq(packed_seq, reverse_indices,max_len):
    unpacked_seq, _ = pad_packed_sequence(packed_seq,total_length = max_len)
    s_inputs_list = []

    for i in reverse_indices:
        s_inputs_list.append(unpacked_seq[:, i, :].unsqueeze(1))
    return torch.cat(s_inputs_list, 1)











def CreateBatchTensorChar(w2i_c, maxlen_c, df):
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
        for w1, w2 in zip(sen1_drop, sen2_drop):
            if w1 in w2i_c and ix_1 < maxlen_c:
                sen1_list[ix_1] = w2i_c[w1]
                ix_1 += 1
            elif ix_1 < maxlen_c:
                sen1_list[ix_1] = w2i_c['A_c']
            if w2 in w2i_c and ix_2 < maxlen_c:
                sen2_list[ix_2] = w2i_c[w2]
                ix_2 += 1
            elif ix_2 < maxlen_c:
                sen2_list[ix_2] = w2i_c['A_c']
                ix_2 += 1
        if ix_1 == 0:
            ix_1 += 1
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

        sen1_drop = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", sen1)
        # sen2_drop = pattern.sub(r'', sen2)
        sen2_drop = re.sub(u'[^\u4e00-\u9fa5A-Za-z0-9*]', "", sen2)
        # sen1_drop = pattern.sub(r'', sen1)
        # sen2_drop = pattern.sub(r'', sen2)
        sen1_list = [0] * maxlen_w
        sen2_list = [0] * maxlen_w
        ix_1 = 0
        ix_2 = 0
        for w1, w2 in zip(sen1_drop.split(), sen2_drop.split()):
            if w1 in w2i_w and ix_1 < maxlen_w:
                sen1_list[ix_1] = w2i_w[w1]
                ix_1 += 1
            elif ix_1 < maxlen_w:
                sen1_list[ix_1] = w2i_w['A_w']
                ix_1 += 1
            if w2 in w2i_w and ix_2 < maxlen_w:
                sen2_list[ix_2] = w2i_w[w2]
                ix_2 += 1
            elif ix_2 < maxlen_w:
                sen2_list[ix_2] = w2i_w['A_w']
                ix_2 += 1
        if ix_1 == 0:
            ix_1 += 1
        if ix_2 == 0:
            ix_2 += 1
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
            if w in char_count_dict and char_count_dict[w] > 3:
                tmp.append(w)
            else:
                tmp.append('A_c')
                # print('meet the rare char :{}'.format(w.encode('utf-8')))
        if len(tmp) > maxlenth_c:
            maxlenth_c = len(tmp)
        sentences.append(tmp)
    word_rep = opt.embedding_dims_char
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
            if w in word_count_dict and word_count_dict[w] > 10:
                tmp.append(w)
            else:
                tmp.append('A_w')
                # print('meet the reate word : {}'.format(w.encode('utf-8')))

        if len(tmp) > maxlenth_w:
            maxlenth_w = len(tmp)
        sentences.append(tmp)
    word_rep = opt.embedding_dims_word
    model_word = gensim.models.Word2Vec(sentences, size=word_rep, min_count=1, iter=20)
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
        weights.append(model_word[i2w[i + 1]])
    weights = torch.FloatTensor(weights)
    return w2i, i2w, weights, model_word, maxlenth_w


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




def SplitData3(df_char,df_word,k_ix):
    data =pd.concat([df_char,df_word],axis = 1)
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

    train = pd.concat([train_p,train_f], ignore_index=True)
    valid = pd.concat([valid_p,valid_f], ignore_index=True)


    train_df_char = train[['idc','sent1c','sent2c','labelc']]
    train_df_char.columns = ['id','sent1','sent2','label']

    train_df_word = train[['idw', 'sent1w', 'sent2w', 'labelw']]
    train_df_word.columns = ['id', 'sent1', 'sent2', 'label']

    valid_df_char = valid[['idc','sent1c','sent2c','labelc']]
    valid_df_char.columns = ['id', 'sent1', 'sent2', 'label']

    valid_df_word = valid[['idw', 'sent1w', 'sent2w', 'labelw']]
    valid_df_word.columns = ['id', 'sent1', 'sent2', 'label']





    return train_df_char, train_df_word,valid_df_char, valid_df_word


















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

    # 2.compute percentage of unique tokens in each string
    input_list1_unique = set([x for x in input_list1 if x not in same_word_list])
    input_list2_unique = set([x for x in input_list2 if x not in same_word_list])
    num_diff_x1 = float(len(input_list1_unique)) / (float(length1) + 0.000001)
    num_diff_x2 = float(len(input_list2_unique)) / (float(length2) + 0.000001)

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

    diff_overlap_list = [num_same_pert_min, num_same_pert_max, num_same_pert_avg, num_diff_x1, num_diff_x2]
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

    canberra_distance = np.sum(np.abs(sentence_vec1 - sentence_vec2) / np.abs(sentence_vec1 + sentence_vec2))
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
                         char_vec_word2vec_dict,tfidf_dict_char,n_gram=8):
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
    # 1. get blue score vector
    feature_list = []
    # get blue score with n-gram
    for i in range(n_gram):
        x1_list = split_string_as_list_by_ngram(input_string_x1, i + 1)
        x2_list = split_string_as_list_by_ngram(input_string_x2, i + 1)
        blue_score_i_1 = compute_blue_ngram(x1_list, x2_list)
        blue_score_i_2 = compute_blue_ngram(x2_list, x1_list)
        feature_list.append(blue_score_i_1)
        feature_list.append(blue_score_i_2)

    # 2. get length of questions, difference of length
    length1 = float(len(input_string_x1))
    length2 = float(len(input_string_x2))
    length_diff = (float(abs(length1 - length2))) / ((length1 + length2 + 0.000001) / 2.0)
    feature_list.append(length_diff)

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
    edit_distance = float(edit(input_string_x1, input_string_x2)) / 30.0
    feature_list.append(edit_distance)

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


def CreateBatchFeat(df,char_vec_word2vec_dict,word_vec_word2vec_dict,tfidf_dict_char,tfidf_dict_word):
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
                                               word_vec_word2vec_dict,tfidf_dict_word, \
                                               char_vec_word2vec_dict,tfidf_dict_char,n_gram=8)
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






def eval(model, df_c, df_w, w2i_c, w2i_w, maxlen_c, maxlen_w,model_char, model_word, tfidf_dict_c,tfidf_dict_w):
    model.eval()
    y_p_float = []
    y_p_int = []
    y_t = []
    criterion_eval = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 5]).float().cuda())
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
        # feat_batch = CreateBatchFeat(df_c.iloc[ix.numpy().tolist()],model_char,model_word, tfidf_dict_c,tfidf_dict_w)


        label = Getlabel(df_c.iloc[ix.numpy().tolist()])

        # output = model(sen1c, sen2c, sen1w, sen2w)
        # output = model(sen1c, len1c, sen2c, len2c)

        output = model(sen1c, sen2c, len1c, len2c, sen1w, sen2w, len1w, len2w)
        loss = criterion_eval(output, label)
        df_loss += loss.item()

        # output2 = model(sen2c, sen1c, sen2w, sen1w, feat_batch)
        # loss = criterion_eval(output2, label)
        # df_loss += loss.item()
        # output3 = (output + output2) / 2
        for item, item3 in zip(output[:,1], label):
            y_p_float.append(item.item())
            y_t.append(int(item3.item()))

    thresh = 0.63
    for item in y_p_float:
        if torch.sigmoid(torch.tensor([item])) > thresh:
            y_p_int.append(1)
        else:
            y_p_int.append(0)
    df_loss = df_loss / count_eval

    f1, precision_rate, recall_rate = Evalue(y_p_int, y_t)
    model.train()

    return df_loss, f1, precision_rate, recall_rate,


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
    else:
        tfidf_dict_w = torch.load(FILE_PATH + '/tfidf_dict_word{}'.format(k_ix))
        print('pre trained word tfidf_dic_dic is used!')
        tfidf_dict_c = torch.load(FILE_PATH + '/tfidf_dict_char{}'.format(k_ix))
        print('pre trained char tfidf_dic_dic is used!')

    # -------------------------------------------------------------------------------------
    if count_inix == 0:
        w2i_c, i2w_c, weights_c, model_char, maxlen_c = CharPik(train_c)
        # ------------------------------------------------------
        torch.save(w2i_c, FILE_PATH + '/w2i_c{}'.format(k_ix))
        torch.save(i2w_c, FILE_PATH + '/i2w_c{}'.format(k_ix))
        torch.save(weights_c, FILE_PATH + '/weights_c{}'.format(k_ix))
        model_char.save(FILE_PATH + '/model_char{}'.format(k_ix))
        torch.save(maxlen_c, FILE_PATH + '/maxlen_c{}'.format(k_ix))
    else:
        w2i_c = torch.load(FILE_PATH + '/w2i_c{}'.format(k_ix))
        i2w_c = torch.load(FILE_PATH + '/i2w_c{}'.format(k_ix))
        weights_c = torch.load(FILE_PATH + '/weights_c{}'.format(k_ix))
        model_char = gensim.models.Word2Vec.load(FILE_PATH + '/model_char{}'.format(k_ix))
        maxlen_c = torch.load(FILE_PATH + '/maxlen_c{}'.format(k_ix))
        print('pre trained w2i_c is used!')
        print('pre trained i2w_c is used!')
        print('pre trained weights_c is used!')
        print('pre trained model_char is used!')
        print('pre trained maxlen_c is used!')

    # ------------------------------------------------------
    if count_inix == 0:
        print('training word word2vector....')
        w2i_w, i2w_w, weights_w, model_word, maxlen_w = WordPik(train_w)

        torch.save(w2i_w, FILE_PATH + '/w2i_w{}'.format(k_ix))
        torch.save(i2w_w, FILE_PATH + '/i2w_w{}'.format(k_ix))
        torch.save(weights_w, FILE_PATH + '/weights_w{}'.format(k_ix))
        model_word.save(FILE_PATH + '/model_word{}'.format(k_ix))
        torch.save(maxlen_w, FILE_PATH + '/maxlen_w{}'.format(k_ix))
    else:
        w2i_w = torch.load(FILE_PATH + '/w2i_w{}'.format(k_ix))
        i2w_w = torch.load(FILE_PATH + '/i2w_w{}'.format(k_ix))
        weights_w = torch.load(FILE_PATH + '/weights_w{}'.format(k_ix))
        model_word = gensim.models.Word2Vec.load(FILE_PATH + '/model_word{}'.format(k_ix))
        maxlen_w = torch.load(FILE_PATH + '/maxlen_w{}'.format(k_ix))
        print('pre trained w2i_w is used!')
        print('pre trained i2w_w is used!')
        print('pre trained weights_w is used!')
        print('pre trained model_word is used!')
        print('pre trained maxlen_w is used!')

    weights_c.cuda()
    weights_w.cuda()

    # model = DecomseAtten(num_embeddings_c = len(i2w_c),embedding_size_c = opt.embedding_dims_char,\
    #                      num_embeddings_w = len(i2w_w),embedding_size_w = opt.embedding_dims_word,\
    #                      hidden_size_c = opt.hidden_dims_char,hidden_size_w =opt.hidden_dims_word,\
    #                      label_size = 1,weight_char = weights_c,weight_word = weights_w)

    # model = DecomseAtten(num_embeddings_c=None, embedding_size_c=opt.embedding_dims_char, \
    #                      num_embeddings_w=None, embedding_size_w=opt.embedding_dims_word, \
    #                      hidden_size_c=opt.hidden_dims_char, hidden_size_w=opt.hidden_dims_word, \
    #                      label_size=1, weight_char=weights_c, weight_word=weights_w).cuda()




    model = WordCharShortCut(num_words_c=len(i2w_c),word_dim_c=opt.embedding_dims_char,lstm_hidden_dims_c = [100,100],\
                     num_words_w = len(i2w_w),word_dim_w = opt.embedding_dims_word,\
                             lstm_hidden_dims_w= [100,100],mlp_hidden_dim=opt.mlp_hidden_dim,\
                             mlp_num_layers=2,num_classes=2,dropout_prob=0.2,weights_c=weights_c,weights_w = weights_w
                     ).cuda()

    if count_inix != 0:
        # encoder_model.load_state_dict(torch.load(FILE_PATH + '/BestF1-0fold.pkl'))
        # atten_model.load_state_dict(torch.load(FILE_PATH + '/BestF1-0fold.pkl'))
        model.load_state_dict(torch.load(FILE_PATH + '/BestF1-0fold.pkl'))
        print('pre trained model is used!!!!')

    # if not os.path.exists(model_dir + '/BestLoss.pkl'):
    #     model.load_state_dict(torch.load(model_dir + '/BestLoss.pkl'))
    # weights_tensor = torch.tensor([0.5, 0.5]).cuda()
    # weights_label = {0: (0, 0), 1: (0, 0)}

    df_ix = range(len(train_c))
    ix_batch = DataLoader(df_ix, batch_size=batch_size, shuffle=True, drop_last=False)

    if count_inix == 0:
        BestF1 = 0.0
        BestLoss = 88888888
    else:
        check_list = torch.load(FILE_PATH + '/F1AndLoss-{}fold.pkl'.format(k_ix))
        BestF1 = check_list[0]
        BestLoss = check_list[1]

    for e in range(epoch):
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 5]).float().cuda())
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

            # now = datetime.datetime.now()
            # print(now.strftime('%Y-%m-%d %H:%M:%S'))
            # print('creating feat_batch tensor.......')
            # feat_batch = CreateBatchFeat(train_c.iloc[ix.numpy().tolist()],\
            #                              model_char,model_word, tfidf_dict_c,tfidf_dict_w)

            label = Getlabel(train_c.iloc[ix.numpy().tolist()])
            # now = datetime.datetime.now()
            # print(now.strftime('%Y-%m-%d %H:%M:%S'))
            # print('feed in to network.......')
            output = model(sen1c,sen2c,len1c,len2c, sen1w, sen2w, len1w, len2w)


            # now = datetime.datetime.now()
            # print(now.strftime('%Y-%m-%d %H:%M:%S'))
            # print('feed  network is done...')

            # feat_batch = []
            # for i in range(batch_size):
            #     feat_batch.append([0] * 28)
            # feat_batch = torch.tensor(feat_batch).float().cuda()
            loss = criterion(output, label)
            if count % 20 == 0:
                print('the loss is {}'.format(loss.item()))

            e_loss += loss.item()
            loss.backward()
            optimizer.step()
            # optimizer_encoder.step()

            # optimizer.zero_grad()
            # optimizer_encoder.zero_grad()

            # output = model(sen2c, sen1c, sen2w, sen1w, feat_batch)
            # try:
            #     loss = criterion(output + 0.00000001, label)
            # except:
            #     print(output)
            #     print(label)
            #
            # loss.backward()
            # optimizer.step()
            # optimizer_encoder.step()

            # if count % 100 == 0:
            #     df_loss, f1, precision_rate, recall_rate = eval(model, valid_c, valid_w, w2i_c, w2i_w, maxlen_c,
            #                                                     maxlen_w,model_word,tfidf_dict)
            #     if df_loss < BestLoss:
            #         BestLoss = df_loss
            #         torch.save(model.state_dict(),\
            #                    model_dir + '/CharWordFeat' + '/WordCharFeaatBestLoss-{}fold.pkl'.format(k_ix))
            #     if f1 > BestF1:
            #         BestF1 = f1
            #         torch.save(model.state_dict(),\
            #                    model_dir + '/CharWordFeat' + '/WordCharFeaatBestF1-{}fold.pkl'.format(k_ix))

        print('the {} epoch is {}'.format(e, e_loss / (count + 0.0000001)))

        df_loss, f1, precision_rate, recall_rate = eval(model, valid_c, \
                                                        valid_w, w2i_c, w2i_w, maxlen_c, \
                                                        maxlen_w,model_char,model_word, \
                                                        tfidf_dict_c,tfidf_dict_w)
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
    epoch = 100
    print('the char embeddnig size is {}'.format(opt.embedding_dims_char))
    print('the word embeddnig size is {}'.format(opt.embedding_dims_word))
    print('the batch size is {}'.format(batch_size))

    model_dir = 'model_dir'
    FILE_PATH = model_dir + '/DecomposeAtten'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(FILE_PATH):
        os.mkdir(FILE_PATH)
    df1 = pd.read_csv('train-c.csv', sep='\t', names=['id', 'sent1', 'sent2', 'label'], encoding='utf-8').head(35000)
    df2 = pd.read_csv('train-w.csv', sep='\t', names=['id', 'sent1', 'sent2', 'label'], encoding='utf-8').head(35000)
    opt.alpha = float(len(df1[df1['label'] == 1])) / float(len(df1))
    print('the alpha is {}'.format(opt.alpha))
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




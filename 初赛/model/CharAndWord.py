import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.optim as optim
class Config(object):
    def __init__(self):
        self.embedding_dims_word = 5
        self.hidden_dims_word = 5
        self.embedding_dims_char = 5
        self.hidden_dims_char  = 5
        self.embedding_dims_char = 5
opt = Config()
class CharAndWord(nn.Module):
    def __init__(self, vocab_size_char,vocab_size_word,opt,weight_char = None,weight_word=None):
        super(CharAndWord, self).__init__()
        self.vocab_size_char = vocab_size_char
        self.vocab_size_word = vocab_size_word
        self.opt = opt
        # Layers
        if weight_char is not None:
            print('char pre train is used...........................')
            self.embedding_table_char = nn.Embedding.from_pretrained(weight_char, freeze=True)
        else:
            self.embedding_table_char = nn.Embedding(num_embeddings=self.vocab_size_char, \
                                                embedding_dim=self.opt.embedding_dims_char,)

        if weight_word is not None:
            print('char pre train is used...........................')
            self.embedding_table_word = nn.Embedding.from_pretrained(weight_word, freeze=True)
        else:
            self.embedding_table_word = nn.Embedding(num_embeddings=self.vocab_size_word, \
                                                embedding_dim=self.opt.embedding_dims_word,)

        self.word_2layer_lstm = nn.LSTM(input_size=self.opt.embedding_dims_word,hidden_size=opt.hidden_dims_word,\
                                        num_layers=2)
        self.word_bi_lstm = nn.LSTM(input_size=self.opt.embedding_dims_word,hidden_size=opt.hidden_dims_word, \
                                    bidirectional = True)
        self.char_bi_lstm = nn.LSTM(input_size=self.opt.embedding_dims_char,hidden_size=opt.hidden_dims_char,\
                                    bidirectional=True)
        self.dropout = nn.Dropout(p = 0.05)
        # self.bi_linear_char = nn.Bilinear(2 * self.opt.hidden_dims_char,2 * self.opt.hidden_dims_char,50)
        # self.bi_linear_word = nn.Bilinear(2 * self.opt.hidden_dims_word,2 * self.opt.hidden_dims_word,50)

        self.linear_relu = nn.Linear(7* self.opt.hidden_dims_word + 6 * self.opt.hidden_dims_char,32)
        self.linear_sigmoid = nn.Linear(7* self.opt.hidden_dims_word + 6 * self.opt.hidden_dims_char,48)

        self.linear_output = nn.Linear(32+48,1)

        # self.lstm_rnn = nn.LSTM(input_size=self.opt.embedding_dims, hidden_size=self.opt.hidden_dims, \
        #                         bias=True, num_layers=self.layer)
        # self.char_bi_lstm = nn.LSTM(input_size=self.opt.embedding_dims_char,hidden_size=opt.hidden_dims_char, \
        #                             bidirectional = True)


    def initial(self,name,batch_size):
        if name == 'word_2layer_lstm':
            return Variable(torch.zeros(2 * 1,batch_size,self.opt.hidden_dims_word)), \
                   Variable(torch.zeros(2 * 1, batch_size, self.opt.hidden_dims_word))
        if name == 'word_bi_lstm':
            return Variable(torch.zeros(2*  1,batch_size,self.opt.hidden_dims_word)), \
                   Variable(torch.zeros(2 * 1, batch_size, self.opt.hidden_dims_word))
        if name == 'char_bi_lstm':
            return Variable(torch.zeros(2 * 1,batch_size,self.opt.hidden_dims_char)), \
                   Variable(torch.zeros(2 * 1, batch_size, self.opt.hidden_dims_char))


    def forward(self, sen1_char,sen2_char,len1_char,len2_char,sen1_w,sen2_w,len1_w,len2_w):
        """
        char level ,implement bisltm

        """

        batch_size = sen1_char.size()[0]
        index_char_sen1 = len1_char.sort(descending=True)[1]
        index_recover_sen1_char = torch.LongTensor([index_char_sen1.numpy().\
                                              tolist().index(item) for item in range(batch_size)])

        sen1_char_emb = self.embedding_table_char(sen1_char)
        sen1_char_emb_sort = sen1_char_emb[index_char_sen1]
        len1_char_sort = len1_char[index_char_sen1]


        ini_hidden, ini_cell = self.initial(name = 'char_bi_lstm',batch_size=batch_size)
        # permute it to Seq * B * V
        input_sen1_char = sen1_char_emb_sort.permute(1, 0, 2)
        pack_sen1_char = nn.utils.rnn.pack_padded_sequence(input_sen1_char, len1_char_sort)
        _, (h_dec_ori_sen1_char, _) = self.char_bi_lstm(pack_sen1_char, (ini_hidden, ini_cell))
        # h_out = h_dec_ori[self.layer - 1][index_recover]
        h_dec_ori_char = h_dec_ori_sen1_char.permute(1,0,2).reshape(batch_size,-1)
        h_out_sen1_char_bi = h_dec_ori_char[index_recover_sen1_char]


        index_char_sen2 = len2_char.sort(descending=True)[1]
        index_recover_sen2_char = torch.LongTensor([index_char_sen2.numpy(). \
                                                   tolist().index(item) for item in range(batch_size)])

        sen2_char_emb = self.embedding_table_char(sen2_char)
        sen2_char_emb_sort = sen2_char_emb[index_char_sen2]
        len2_char_sort = len2_char[index_char_sen2]

        ini_hidden, ini_cell = self.initial(name='char_bi_lstm', batch_size=batch_size)
        # permute it to Seq * B * V
        input_sen2_char = sen2_char_emb_sort.permute(1, 0, 2)
        pack_sen2_char = nn.utils.rnn.pack_padded_sequence(input_sen2_char, len2_char_sort)
        _, (h_dec_ori_sen2_char, _) = self.char_bi_lstm(pack_sen2_char, (ini_hidden, ini_cell))
        h_dec_ori_sen2_char = h_dec_ori_sen2_char.permute(1, 0, 2).reshape(batch_size, -1)
        h_out_sen2_char_bi = h_dec_ori_sen2_char[index_recover_sen2_char]




        """
         word level implement bi-lstm 
        """

        # sen1 word
        index_word_sen1 = len1_w.sort(descending=True)[1]
        index_recover_sen1_w = torch.LongTensor([index_word_sen1.numpy().\
                                                   tolist().index(item) for item in range(batch_size)])

        sen1_word_emb = self.embedding_table_word(sen1_w)
        sen1_word_emb_sort = sen1_word_emb[index_word_sen1]
        len1_word_sort = len1_w[index_word_sen1]

        ini_hidden, ini_cell = self.initial(name='word_bi_lstm', batch_size=batch_size)

        # permute it to Seq * B * V
        input_sen1_word = sen1_word_emb_sort.permute(1, 0, 2)
        pack_sen1_word = nn.utils.rnn.pack_padded_sequence(input_sen1_word, len1_word_sort)

        _, (h_dec_ori_sen1_word, _) = self.word_bi_lstm(pack_sen1_word, (ini_hidden, ini_cell))

        h_dec_ori_sen1_word = h_dec_ori_sen1_word.permute(1, 0, 2).reshape(batch_size, -1)
        h_out_sen1_word_bi = h_dec_ori_sen1_word[index_recover_sen1_w]

        # sen2 word
        index_word_sen2 = len2_w.sort(descending = True)[1]
        index_recover_sen2_w = torch.LongTensor([index_word_sen2.numpy(). \
                                                   tolist().index(item) for item in range(batch_size)])

        sen2_word_emb = self.embedding_table_word(sen2_w)
        sen1_word_emb_sort = sen2_word_emb[index_word_sen2]
        len2_word_sort = len2_w[index_word_sen2]

        ini_hidden, ini_cell = self.initial(name='word_bi_lstm', batch_size=batch_size)

        # permute it to Seq * B * V
        input_sen2_word = sen1_word_emb_sort.permute(1, 0, 2)
        pack_sen2_word = nn.utils.rnn.pack_padded_sequence(input_sen2_word, len2_word_sort)

        _, (h_dec_ori_sen2_word, _) = self.word_bi_lstm(pack_sen2_word, (ini_hidden, ini_cell))

        h_dec_ori_sen2_word = h_dec_ori_sen2_word.permute(1, 0, 2).reshape(batch_size, -1)
        h_out_sen2_word_bi = h_dec_ori_sen2_word[index_recover_sen2_w]

        '''
        word level implement 2later lstm
        '''

        ini_hidden, ini_cell = self.initial(name='word_2layer_lstm', batch_size=batch_size)
        _, (h_dec_ori_sen1_word, _) = self.word_2layer_lstm(pack_sen1_word, (ini_hidden, ini_cell))
        h_out_sen1_word_2layer = h_dec_ori_sen1_word[1][index_recover_sen1_w]

        ini_hidden, ini_cell = self.initial(name='word_2layer_lstm', batch_size=batch_size)
        _, (h_dec_ori_sen2_word, _) = self.word_2layer_lstm(pack_sen2_word, (ini_hidden, ini_cell))
        h_out_sen2_word_2layer = h_dec_ori_sen2_word[1][index_recover_sen2_w]


        mul_word = h_out_sen1_word_bi * h_out_sen2_word_bi
        sub_word = torch.abs(h_out_sen1_word_bi - h_out_sen2_word_bi)
        maximum_word = torch.max(h_out_sen1_word_bi,h_out_sen2_word_bi)
        # the out space is set to 50
        # bi_feat_word = self.bi_linear_word(h_out_sen1_word_bi,h_out_sen2_word_bi)

        # plus_word = h_out_sen1_word_bi + h_out_sen2_word_bi
        mul_char = h_out_sen1_char_bi * h_out_sen2_char_bi
        sub_char = torch.abs(h_out_sen1_char_bi - h_out_sen2_char_bi)
        maximum_char = torch.max(h_out_sen1_char_bi,h_out_sen2_char_bi)
        # bi_feat_char = self.bi_linear_char(h_out_sen1_char_bi,h_out_sen2_char_bi)
        # plus_char = h_out_sen1_char_bi + h_out_sen2_char_bi


        sub_word_2layer = torch.abs(h_out_sen1_word_2layer - h_out_sen2_word_2layer)

        matchlist  = self.dropout(torch.cat([mul_word,sub_word,maximum_word,sub_char, \
                                             mul_char,maximum_char,\
                                             sub_word_2layer],dim = 1))


        match_list_relu = self.linear_relu(matchlist)
        match_list_sigmoid = self.linear_sigmoid(matchlist)

        output = self.linear_output(torch.cat([match_list_relu,match_list_sigmoid],dim = 1))

        return torch.sigmoid(output)

if __name__ == '__main__':
    sen1_c = torch.randint(1,10,(5,6)).long()
    sen2_c = torch.randint(1,10,(5,6)).long()
    len1_c = torch.LongTensor([3,5,3,4,1])
    len2_c = torch.LongTensor([4,5,5,2,4])

    sen1_w = torch.randint(1, 10, (5, 4)).long()
    sen2_w = torch.randint(1, 10, (5, 4)).long()
    len1_w = torch.LongTensor([4, 2, 3, 2,3])
    len2_w = torch.LongTensor([4, 1, 3, 4,2])

    label = torch.tensor([1,0,0,1,0]).float()
    m = CharAndWord(vocab_size_char = 10,  vocab_size_word = 10,opt = opt,weight_char = None, weight_word =None)
    criterion = nn.BCELoss()
    out = m(sen1_c,sen2_c,len1_c,len2_c,sen1_w,sen2_w,len1_w,len2_w)
    epoch = 100
    optimizer = optim.Adam(m.parameters())
    for e in range(epoch):
        y_pre = []
        optimizer.zero_grad()
        out = m(sen1_c, sen2_c, len1_c, len2_c, sen1_w, sen2_w, len1_w, len2_w)
        loss = criterion(out,label)
        loss.backward()
        optimizer.step()
        for item in out:
            if item > 0.5:
                y_pre.append(1)
            else:
                y_pre.append(0)
        y_pre = torch.tensor(y_pre).float()
        print(torch.sum(y_pre == label).float().item() / len(y_pre))



































import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.optim as optim
import torch.nn.functional as F
class Config(object):
    def __init__(self):
        self.embedding_dims = 5
        self.cnn_outchannels = 5
        self.hidden_dims = 5
        self.k_size = 3
opt = Config()


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)



class RnnCnn(nn.Module):
    def __init__(self, vocab_size,embed_dims,hidden_size,cnn_out_channels,layer=1, weight=None):
        super(RnnCnn, self).__init__()
        self.layer = layer
        self.vocab_size = vocab_size
        self.opt = opt
        self.embed_dims = embed_dims
        self.name = 'RnnCnn'
        self.cnn_out_channels = cnn_out_channels
        self.lstm_hidden_size = hidden_size
        self.kernel_size = 3
        self.layer = layer
        self.bi_direct = 2
        self.dropout = nn.Dropout(p = 0.05)
        if weight is not None:
            print('pre train is used...........................')
            self.embedding_table = nn.Embedding.from_pretrained(weight, freeze=True)
        else:
            self.embedding_table = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dims,
                                                padding_idx=0, max_norm=None, scale_grad_by_freq=True, sparse=False)
        self.lstm_rnn = nn.LSTM(input_size= embed_dims, hidden_size=self.lstm_hidden_size, \
                                bias=False,bidirectional=True, num_layers=self.layer)

        self.cnn = nn.Conv1d(self.lstm_hidden_size * 2,self.cnn_out_channels,kernel_size= self.kernel_size,\
                             bias= False,padding= self.kernel_size//3)

        self.relu_linear = nn.Linear(6 * self.lstm_hidden_size + 2 * self.cnn_out_channels,32)
        self.sigmoid_linear = nn.Linear(6 * self.lstm_hidden_size + 2 * self.cnn_out_channels,48)
        self.output = nn.Linear(80,1)

    def initialize_hidden_plus_cell(self, batch_size):
        init_hidden = Variable(torch.zeros(self.layer * self.bi_direct, batch_size, self.lstm_hidden_size))
        init_cell = Variable(torch.zeros(self.layer  * self.bi_direct, batch_size,  self.lstm_hidden_size))
        return init_hidden, init_cell



    def forward(self, sen1,sen2,len1,len2):
        seq_lenth = sen1.size()[1]
        batch_size = sen1.size()[0]
        sen_emb = self.embedding_table(sen1)
        sen1_ix = len1.sort(descending=True)[1]
        sen1_ix_recover = torch.LongTensor([sen1_ix.numpy().tolist().index(item) for item in range(len(sen1_ix))])

        sen1_sort = sen_emb[sen1_ix]
        len1_sort = len1[sen1_ix]

        init_hidden, init_cell = self.initialize_hidden_plus_cell(batch_size=batch_size)
        sen1_input = sen1_sort.permute(1, 0, 2)

        sen1_pack = nn.utils.rnn.pack_padded_sequence(sen1_input, len1_sort)

        out_sen1, (h_dec_ori_sen1, c_sen1) = self.lstm_rnn(sen1_pack, (init_hidden, init_cell))

        out_sen1_padd = nn.utils.rnn.pad_packed_sequence(out_sen1,total_length = seq_lenth)[0]
        out_sen1_bfirst = out_sen1_padd.permute(1,0,2)
        # h_out = h_dec_ori[self.layer - 1][sen1_ix_recover].permute(1,0,2).view(batch_size,-1)
        h_out_sen1 = h_dec_ori_sen1.permute(1,0,2).reshape(batch_size,-1)


        out_sen1_recover = out_sen1_bfirst[sen1_ix_recover]
        out_sen1_forcnn = out_sen1_recover.permute(0,2,1)
        cnn_out_sen1 = F.tanh(self.cnn(out_sen1_forcnn))
        kmax_sen1 = kmax_pooling(cnn_out_sen1, dim=2, k=1).squeeze(2)

        #--------------------------------------------------------------------------------------------
        sen_emb = self.embedding_table(sen2)
        sen2_ix = len2.sort(descending=True)[1]
        sen2_ix_recover = torch.LongTensor([sen2_ix.numpy().tolist().index(item) for item in range(len(sen2_ix))])

        sen2_sort = sen_emb[sen2_ix]
        len2_sort = len2[sen2_ix]

        init_hidden, init_cell = self.initialize_hidden_plus_cell(batch_size=batch_size)
        sen2_input = sen2_sort.permute(1, 0, 2)

        sen1_pack = nn.utils.rnn.pack_padded_sequence(sen2_input, len2_sort)

        out_sen2, (h_dec_ori_sen2, c_sen2) = self.lstm_rnn(sen1_pack, (init_hidden, init_cell))

        out_sen2_padd = nn.utils.rnn.pad_packed_sequence(out_sen2, total_length=seq_lenth)[0]
        out_sen2_bfirst = out_sen2_padd.permute(1, 0, 2)
        # h_out = h_dec_ori[self.layer - 1][sen1_ix_recover]
        # h_out = h_dec_ori[self.layer - 1][sen1_ix_recover].permute(1,0,2).view(batch_size,-1)
        h_out_sen2 = h_dec_ori_sen2.permute(1, 0, 2).reshape(batch_size, -1)



        out_sen2_recover = out_sen2_bfirst[sen2_ix_recover]
        out_sen2_forcnn = out_sen2_recover.permute(0, 2, 1)
        cnn_out_sen2 = F.tanh(self.cnn(out_sen2_forcnn))
        kmax_sen2 = kmax_pooling(cnn_out_sen2, dim=2, k=1).squeeze(2)

        sub_feat =  torch.abs(h_out_sen1 - h_out_sen2)
        max_feat = torch.max(h_out_sen1,h_out_sen2)
        muti_feat = h_out_sen1 * h_out_sen2

        matchlist = self.dropout(torch.cat([sub_feat,max_feat,muti_feat,kmax_sen1,kmax_sen2], dim=1))
        relu_feat = F.relu(self.relu_linear(matchlist))
        sig_feat = F.sigmoid(self.sigmoid_linear(matchlist))

        output = self.output(torch.cat([relu_feat,sig_feat],dim = 1))

        return F.sigmoid(output)



class RnnCnnSigle(nn.Module):
    def __init__(self, vocab_size,embed_dims,hidden_size,cnn_out_channels,layer=1, weight=None):
        super(RnnCnnSigle, self).__init__()
        self.layer = layer
        self.vocab_size = vocab_size
        self.opt = opt
        self.embed_dims = embed_dims
        self.name = 'RnnCnn'
        self.cnn_out_channels = cnn_out_channels
        self.lstm_hidden_size = hidden_size
        self.kernel_size = 5
        self.layer = layer
        self.bi_direct = 1
        self.dropout = nn.Dropout(p = 0.05)
        if weight is not None:
            print('pre train is used...........................')
            self.embedding_table = nn.Embedding.from_pretrained(weight, freeze=True)
        else:
            self.embedding_table = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dims,
                                                padding_idx=0, max_norm=None, scale_grad_by_freq=True, sparse=False)
        self.lstm_rnn = nn.LSTM(input_size= embed_dims, hidden_size=self.lstm_hidden_size, \
                                bias=False,bidirectional=False, num_layers=self.layer)

        self.cnn = nn.Conv1d(self.lstm_hidden_size,self.cnn_out_channels,kernel_size= self.kernel_size,\
                             bias= False,padding= self.kernel_size//3)

        self.relu_linear = nn.Linear(3 * self.lstm_hidden_size + 2 * self.cnn_out_channels,32)
        self.sigmoid_linear = nn.Linear(3 * self.lstm_hidden_size + 2 * self.cnn_out_channels,48)
        self.output = nn.Linear(80,1)

    def initialize_hidden_plus_cell(self, batch_size):
        init_hidden = Variable(torch.zeros(self.layer * self.bi_direct, batch_size, self.lstm_hidden_size))
        init_cell = Variable(torch.zeros(self.layer  * self.bi_direct, batch_size,  self.lstm_hidden_size))
        return init_hidden, init_cell



    def forward(self, sen1,sen2,len1,len2):
        seq_lenth = sen1.size()[1]
        batch_size = sen1.size()[0]
        sen1_emb = self.embedding_table(sen1)
        sen1_ix = len1.sort(descending=True)[1]
        sen1_ix_recover = torch.LongTensor([sen1_ix.numpy().tolist().index(item) for item in range(len(sen1_ix))])

        sen1_sort = sen1_emb[sen1_ix]
        len1_sort = len1[sen1_ix]

        init_hidden, init_cell = self.initialize_hidden_plus_cell(batch_size=batch_size)
        sen1_input = sen1_sort.permute(1, 0, 2)

        sen1_pack = nn.utils.rnn.pack_padded_sequence(sen1_input, len1_sort)

        out_sen1, (h_dec_ori_sen1, c_sen1) = self.lstm_rnn(sen1_pack, (init_hidden, init_cell))

        out_sen1_padd = nn.utils.rnn.pad_packed_sequence(out_sen1,total_length = seq_lenth)[0]
        out_sen1_bfirst = out_sen1_padd.permute(1,0,2)
        # h_out = h_dec_ori[self.layer - 1][sen1_ix_recover].permute(1,0,2).view(batch_size,-1)
        h_out_sen1 = h_dec_ori_sen1.permute(1,0,2).reshape(batch_size,-1)[sen1_ix_recover]


        out_sen1_recover = out_sen1_bfirst[sen1_ix_recover]
        out_sen1_forcnn = out_sen1_recover.permute(0,2,1)
        # sen1_forcnn = sen1_emb.permute(0,2,1)
        cnn_out_sen1 = F.tanh(self.cnn(out_sen1_forcnn))
        kmax_sen1 = kmax_pooling(cnn_out_sen1, dim=2, k=1).squeeze(2)

        #--------------------------------------------------------------------------------------------
        sen_emb2 = self.embedding_table(sen2)
        sen2_ix = len2.sort(descending=True)[1]
        sen2_ix_recover = torch.LongTensor([sen2_ix.numpy().tolist().index(item) for item in range(len(sen2_ix))])

        sen2_sort = sen_emb2[sen2_ix]
        len2_sort = len2[sen2_ix]

        init_hidden, init_cell = self.initialize_hidden_plus_cell(batch_size=batch_size)
        sen2_input = sen2_sort.permute(1, 0, 2)

        sen1_pack = nn.utils.rnn.pack_padded_sequence(sen2_input, len2_sort)

        out_sen2, (h_dec_ori_sen2, c_sen2) = self.lstm_rnn(sen1_pack, (init_hidden, init_cell))

        out_sen2_padd = nn.utils.rnn.pad_packed_sequence(out_sen2, total_length=seq_lenth)[0]
        out_sen2_bfirst = out_sen2_padd.permute(1, 0, 2)
        # h_out = h_dec_ori[self.layer - 1][sen1_ix_recover]
        # h_out = h_dec_ori[self.layer - 1][sen1_ix_recover].permute(1,0,2).view(batch_size,-1)
        h_out_sen2 = h_dec_ori_sen2.permute(1, 0, 2).reshape(batch_size, -1)[sen2_ix_recover]



        out_sen2_recover = out_sen2_bfirst[sen2_ix_recover]
        out_sen2_forcnn = out_sen2_recover.permute(0, 2, 1)
        cnn_out_sen2 = F.tanh(self.cnn(out_sen2_forcnn))
        kmax_sen2 = kmax_pooling(cnn_out_sen2, dim=2, k=1).squeeze(2)

        sub_feat =  torch.abs(h_out_sen1 - h_out_sen2)
        max_feat = torch.max(h_out_sen1,h_out_sen2)
        muti_feat = h_out_sen1 * h_out_sen2

        matchlist = self.dropout(torch.cat([sub_feat,max_feat,muti_feat,kmax_sen1,kmax_sen2], dim=1))
        relu_feat = F.relu(self.relu_linear(matchlist))
        sig_feat = F.sigmoid(self.sigmoid_linear(matchlist))

        output = self.output(torch.cat([relu_feat,sig_feat],dim = 1))

        return F.sigmoid(output)
































if __name__ == '__main__':
    sen1 = torch.randint(0,10,(3,5)).long()
    sen2 = torch.randint(1, 10, (3, 5)).long()
    label = torch.tensor([1,0,1])
    len1 = torch.tensor([3,3,3])
    len2 = torch.tensor([3, 3, 3])
    # vocab_size, embed_dims, hidden_size, cnn_out_channels, layer = 1, weight = None
    m = RnnCnnSigle(vocab_size = 10,embed_dims = 4,hidden_size=3,cnn_out_channels= 3)
    out = m(sen1,sen2,len1,len2)


































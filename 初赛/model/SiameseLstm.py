import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.optim as optim
class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, opt, layer=1,weight=None):
        super(LSTMEncoder, self).__init__()
        self.layer = layer
        self.vocab_size = vocab_size
        self.opt = opt
        self.name = 'sim_encoder'
        # Layers
        if weight is not None:
            print('pre train is used...........................')
            self.embedding_table = nn.Embedding.from_pretrained(weight, freeze=True)
        else:
            self.embedding_table = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.opt.embedding_dims,
                                                padding_idx=0, max_norm=None, scale_grad_by_freq=True, sparse=False)
        self.lstm_rnn = nn.LSTM(input_size=self.opt.embedding_dims, hidden_size=self.opt.hidden_dims,\
                                bias=True,num_layers=self.layer)
        # self.linear = nn.Linear(self.opt.hidden_dims,self.opt.hidden_dims)
        # self.linear2 = nn.Linear(self.opt.hidden_dims,self.opt.hidden_dims//2)
        # self.dropout = nn.Dropout(p=0.2)
        # self.softplus = torch.nn.Softplus()

    def initialize_hidden_plus_cell(self, batch_size):
        zero_hidden = Variable(torch.randn(self.layer, batch_size, self.opt.hidden_dims))
        zero_cell = Variable(torch.randn(self.layer, batch_size, self.opt.hidden_dims))
        return zero_hidden, zero_cell

    def forward(self,sen,sen_len):
        """ Performs a forward pass through the network. """
        self.embedding_table.weight.data[0] = 0
        sen_emb = self.embedding_table(sen)

        index = sen_len.sort(descending=True)[1]
        index_recover = torch.LongTensor([index.numpy().tolist().index(item) for item in range(len(index))])

        sen_sort = sen_emb[index]
        sen_len_sort = sen_len[index]
        batch_size = len(sen_len)

        zero_hidden, zero_cell = self.initialize_hidden_plus_cell(batch_size=batch_size)
        input_data = sen_sort.permute(1, 0,2)
        try:
            pack = nn.utils.rnn.pack_padded_sequence(input_data, sen_len_sort)
        except:
            print('no')
        output, (h_dec_ori, c) = self.lstm_rnn(pack, (zero_hidden, zero_cell))
        h_out = h_dec_ori[self.layer -1][index_recover]
        # h_out_linear = torch.tan(self.linear(h_out))
        return h_out


class SiameseClassifier(nn.Module):
    def __init__(self, vocab_size, opt, pretrained_embeddings=None, is_train=False):
        super(SiameseClassifier, self).__init__()
        self.opt = opt
        self.encoder_a = self.encoder_b = LSTMEncoder(vocab_size, self.opt, is_train)
        # Initialize pre-trained embeddings, if given
        if pretrained_embeddings is not None:
            None


    def forward(self,input_data_sen1,input_data_sen2):
        hidden_a, cell_a = self.encoder_a.initialize_hidden_plus_cell(input_data_sen1.size()[1])
        output_a,_,_ = self.encoder_a(input_data_sen1,hidden_a, cell_a )


        hidden_b, cell_b = self.encoder_b.initialize_hidden_plus_cell(input_data_sen1.size()[1])
        output_b, _, _ = self.encoder_a(input_data_sen2, hidden_b, cell_b)


        self.encoding_a = output_a[-1]
        self.encoding_b = output_b[-1]


        self.prediction = torch.exp(-torch.norm((self.encoding_a - self.encoding_b), 1,1))
        return self.prediction

class LogsiticClassify(nn.Module):
    def __init__(self,hidden_size):
        super(LogsiticClassify, self).__init__()
        self.linear1 = torch.nn.Linear(hidden_size * 3,1)
        # self.linear2 = torch.nn.Linear(hidden_size*2,1)
        # self.dropout = nn.Dropout(p = 0.2)
    def forward(self, input):
        outout = torch.sigmoid(self.linear1(input))
        # outout2 = torch.sigmoid(self.linear2(outout))
        return outout







class p(object):
    def __init__(self):
        self.learning_rate = 0.1
        self.hidden_dims = 8
        self.embedding_dims = 10
        self.beta_1 = 0.9

if __name__ == '__main__':
    cri = nn.MSELoss()
    b_data1 = torch.randint(0, 10, (7,5), dtype=torch.long)
    b_data2 = torch.randint(0, 10, (7,5), dtype=torch.long)
    y_label = Variable(torch.randint(0,2,(5,)))
    hidden = torch.zeros(5,5)
    cell = torch.zeros(5, 5)
    opt = p()
    m = SiameseClassifier(vocab_size = 10,opt = opt)
    optimizer = optim.Adam(m.parameters())
    for i in range(100):
        loss = 0
        optimizer.zero_grad()
        pre = m(b_data1,b_data2)
        loss +=cri(pre,y_label)
        loss.backward()
        optimizer.step()
        result = []
        for item in pre:
            if item.item() > 0.5:
                result.append(1)
            else:
                result.append(0)
        acc = 0.0
        for pre,tar in zip(result,y_label):
            if pre == int(tar.item()):
                acc += 1
        print(acc / len(result))

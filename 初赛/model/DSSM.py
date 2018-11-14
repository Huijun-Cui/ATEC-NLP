

import torch
import torch.nn as nn
import datetime
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

LETTER_GRAM_SIZE = 3  # See section 3.2.
WINDOW_SIZE = 3  # See section 3.2.
TOTAL_LETTER_GRAMS = int(3 * 1e4)  # Determined from data. See section 3.2.
WORD_DEPTH = WINDOW_SIZE * TOTAL_LETTER_GRAMS  # See equation (1).
# Uncomment it, if testing
# WORD_DEPTH = 1000
K = 300  # Dimensionality of the max-pooling layer. See section 3.4.
L = 128  # Dimensionality of latent semantic space. See section 3.5.
J = 4  # Number of random unclicked documents serving as negative examples for a query. See section 4.
FILTER_LENGTH = 2  # We only consider one time step for convolutions.


def kmax_pooling(x, dim, k):
    # index = []
    # for item in a:
    #     tmp_ix = item[item != 0].topk(1)[1].item()
    #     index.append([tmp_ix])
    # index = torch.LongTensor(index)
    # index = []
    # for x_2d in x:
    #     for item in x_2d:
    #         tmp_ix = item[item != 0].topk(1)[1].item()
    #         index.append([tmp_ix])

    x_cp = x.clone()
    x_cp[x_cp == 0] -888888
    return x.gather(dim = dim,index = x_cp.topk(1,dim = dim)[1])




    # index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    # # # index = x.topk(k, dim=dim)[1]
    # return x.gather(dim, index)






class CLSM(nn.Module):
    def __init__(self,voc_size = WORD_DEPTH,word_rep = 30,\
                 out_channels = K,k_size = FILTER_LENGTH,sementic_size = L,weight = 0):
        super(CLSM, self).__init__()
        # layers for query
        if isinstance(weight,int):
            self.embedding = nn.Embedding(voc_size,word_rep,padding_idx= 0)
        else:
            print('Pretrained embedding is used!!!!!!')
            self.embedding = nn.Embedding.from_pretrained(weight,freeze=True)
        self.query_conv = nn.Conv1d(word_rep, out_channels, k_size,bias=False,padding= k_size //2)
        self.query_sem = nn.Linear(out_channels, sementic_size)
        # self.query_sem_second = nn.Linear(sementic_size,sementic_size //2)
        # layers for docsing is used!!!!!!

        self.doc_conv = nn.Conv1d(word_rep, out_channels, k_size,bias= False,padding= k_size //2)
        self.doc_sem = nn.Linear(out_channels, sementic_size)
        self.decision = nn.Linear(sementic_size*2, 1)
        # self.dropout = nn.Dropout(p=0.2)
        # learning gamma
        # self.learn_gamma = nn.Conv1d(1, 1, 1)

    def forward(self, q_raw, pos_raw):
        batch_size = q_raw.size()[0]
        # Query model. The paper uses separate neural nets for queries and documents (see section 5.2).
        # To make it compatible with Conv layer we reshape it to: (batch_size, WORD_DEPTH, query_len)
        q = self.embedding(q_raw)
        q = q.transpose(1, 2)
        # In this step, we transform each word vector with WORD_DEPTH dimensions into its
        # convolved representation with K dimensions. K is the number of kernels/filters
        # being used in the operation. Essentially, the operation is taking the dot product
        # of a single weight matrix (W_c) with each of the word vectors (l_t) from the
        # query matrix (l_Q), adding a bias vector (b_c), and then applying the tanh activation.
        # That is, h_Q = tanh(W_c ? l_Q + b_c). Note: the paper does not include bias units.
        # now = datetime.datetime.now()
        # print(now.strftime('%Y-%m-%d %H:%M:%S'))
        # print('Is query conving ....')
        q_c = self.query_conv(q)
        # Next, we apply a max-pooling layer to the convolved query matrix.
        # now = datetime.datetime.now()
        # print(now.strftime('%Y-%m-%d %H:%M:%S'))
        # print('Is query kmax pooling ....')
        q_k = kmax_pooling(x= q_c, dim = 2, k = 1).view(q_c.size()[0],-1,1)
        q_k = q_k.transpose(1, 2)

        # In this step, we generate the semantic vector represenation of the query. This
        # is a standard neural network dense layer, i.e., y = tanh(W_s ? v + b_s). Again,
        # the paper does not include bias units.
        # now = datetime.datetime.now()
        # print(now.strftime('%Y-%m-%d %H:%M:%S'))
        # print('Is query semantic transfer ....')
        q_s =self.query_sem(q_k)
        pos = self.embedding(pos_raw)
        pos = pos.transpose(1, 2)
        # now = datetime.datetime.now()
        # print(now.strftime('%Y-%m-%d %H:%M:%S'))
        # print('Is odc conving ....')
        pos_c = self.doc_conv(pos)

        # now = datetime.datetime.now()
        # print(now.strftime('%Y-%m-%d %H:%M:%S'))
        # print('Is doc kmax pooling  ....')
        pos_k = kmax_pooling(x = pos_c, dim = 2, k =  1).view(pos_c.size()[0],-1,1)
        pos_k = pos_k.transpose(1, 2)

        # now = datetime.datetime.now()
        # print(now.strftime('%Y-%m-%d %H:%M:%S'))
        # print('Is doc semantic transfer ....')
        pos_s =F.tanh(self.doc_sem(pos_k))

        q_s = q_s.reshape(batch_size,-1)
        pos_s = pos_s.reshape(batch_size,-1)
        # max_feat = torch.max(q_s,pos_s)
        # concat_rep = torch.cat([(q_s - pos_s)**2,(q_s + pos_s),max_feat], dim=1)
        # out_put = torch.sigmoid(self.decision(concat_rep))
        out_put = self.decision(torch.cat([q_s,pos_s],dim = 1))
        # out_put = torch.sigmoid(torch.sum(q_s * pos_s,dim = 1))


        return torch.sigmoid(out_put)

class CLSMMutiKernelSize(nn.Module):
    def __init__(self,in_channels = WORD_DEPTH,word_rep = 500,out_channels = K,sementic_size = L):
        super(CLSM, self).__init__()
        # layers for query
        self.embedding = nn.Embedding(WORD_DEPTH,word_rep,padding_idx= 0)
        self.query_conv_k1 = nn.Conv1d(in_channels = word_rep, out_channels = out_channels, k_size = 1,bias=False)
        self.query_conv_k3 = nn.Conv1d(in_channels = word_rep, out_channels = out_channels, k_size = 3,bias=False)
        self.query_conv_k5 = nn.Conv1d(in_channels=word_rep, out_channels=out_channels, k_size=5, bias=False)
        self.query_sem = nn.Linear(out_channels, sementic_size)
        # layers for docs
        self.doc_conv_k1 = nn.Conv1d(in_channels=word_rep, out_channels=out_channels, k_size=1, bias=False)
        self.doc_conv_k3 = nn.Conv1d(in_channels=word_rep, out_channels=out_channels, k_size=3, bias=False)
        self.doc_conv_k5 = nn.Conv1d(in_channels=word_rep, out_channels=out_channels, k_size=5, bias=False)
        self.doc_sem = nn.Linear(out_channels, sementic_size)
        self.decision = nn.Linear(sementic_size * 2, 2)
        # learning gamma
        # self.learn_gamma = nn.Conv1d(1, 1, 1)

    def forward(self, q_raw, pos_raw):
        # Query model. The paper uses separate neural nets for queries and documents (see section 5.2).
        # To make it compatible with Conv layer we reshape it to: (batch_size, WORD_DEPTH, query_len)
        q = self.embedding(q_raw)
        q = q.transpose(1, 2)
        # In this step, we transform each word vector with WORD_DEPTH dimensions into its
        # convolved representation with K dimensions. K is the number of kernels/filters
        # being used in the operation. Essentially, the operation is taking the dot product
        # of a single weight matrix (W_c) with each of the word vectors (l_t) from the
        # query matrix (l_Q), adding a bias vector (b_c), and then applying the tanh activation.
        # That is, h_Q = tanh(W_c ? l_Q + b_c). Note: the paper does not include bias units.
        # now = datetime.datetime.now()
        # print(now.strftime('%Y-%m-%d %H:%M:%S'))
        # print('Is query conving ....')
        q_c_kersize_1 = F.tanh(self.query_conv_k1(q))
        # Next, we apply a max-pooling layer to the convolved query matrix.
        # now = datetime.datetime.now()
        # print(now.strftime('%Y-%m-%d %H:%M:%S'))
        # print('Is query kmax pooling ....')
        q_k_kersize_1 = kmax_pooling(q_c, 2, 1)
        q_k_kersize_2 = q_k.transpose(1, 2)

        # In this step, we generate the semantic vector represenation of the query. This
        # is a standard neural network dense layer, i.e., y = tanh(W_s ? v + b_s). Again,
        # the paper does not include bias units.
        # now = datetime.datetime.now()
        # print(now.strftime('%Y-%m-%d %H:%M:%S'))
        # print('Is query semantic transfer ....')
        q_s = F.tanh(self.query_sem(q_k))

        pos = self.embedding(pos_raw)
        pos = pos.transpose(1, 2)
        # now = datetime.datetime.now()
        # print(now.strftime('%Y-%m-%d %H:%M:%S'))
        # print('Is odc conving ....')
        pos_c = F.tanh(self.doc_conv(pos))
        # now = datetime.datetime.now()
        # print(now.strftime('%Y-%m-%d %H:%M:%S'))
        # print('Is doc kmax pooling  ....')
        pos_k = kmax_pooling(pos_c, 2, 1)
        pos_k = pos_k.transpose(1, 2)
        # now = datetime.datetime.now()
        # print(now.strftime('%Y-%m-%d %H:%M:%S'))
        # print('Is doc semantic transfer ....')
        pos_s = F.tanh(self.doc_sem(pos_k))


        concat_rep = torch.cat([q_s, pos_s], dim=2)
        out_put = torch.sigmoid(self.decision(concat_rep))


        return out_put




if __name__ == '__main__':
    from torch.nn import CrossEntropyLoss
    from torch.autograd import Variable
    import numpy as np
    sample_size = 10
    WORD_DEPTH = 10
    l_Qs = []
    pos_l_Ds = []
    criterion = CrossEntropyLoss()
    for i in range(sample_size):
        print('test:{}'.format(i))
        query_len = 5
        l_Q = torch.randint(0, 10, (query_len,)).tolist()
        l_Qs.append(l_Q)

        doc_len = 5
        l_D = torch.randint(0, 10, (doc_len,)).tolist()
        pos_l_Ds.append(l_D)
    y = [1,1,0,1,0,0,1,1,0,0]

    y = Variable(torch.LongTensor(y))
    l_Qs = Variable(torch.FloatTensor(l_Qs))
    pos_l_Ds = Variable(torch.FloatTensor(pos_l_Ds))

    model = CLSM(in_channels = 10,out_channels =5 ,k_size = 5,sementic_size = 6)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    while(True):
        print('new epoch....')
        for i in range(10):
            loss = 0
            optimizer.zero_grad()
            result = model(l_Qs.long(),pos_l_Ds.long())
            loss += criterion(result,y.float())
            loss.backward()
            optimizer.step()
            print('the loss is {}'.format(loss.data[0]))
        y_pred = model(l_Qs.long(),pos_l_Ds.long())
        y_pred_list = []
        for item in y_pred:
            if item > 0.5:
                y_pred_list.append(1)
            else:
                y_pred_list.append(0)
        y_pred_list = torch.tensor(y_pred_list)
        print('the accuracy is {}'.format((torch.sum(y_pred_list == y).data[0])))










# coding=utf8
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import datetime
import re
import os
import pickle
import numpy as np
from sklearn.decomposition import TruncatedSVD


def CosDist(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.sum(vec1 * vec2) / (np.sqrt(np.sum(vec1 * vec1)) * np.sqrt(np.sum(vec2 * vec2)))


def EuclideanDist(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.sqrt(np.sum((vec1 - vec2) * (vec1 - vec2)))

def UnigramBOW(train,sample):
    if os.path.exists('./dataset/FeaturePkl/UnigramBOW.pkl'):
        with open('./dataset/FeaturePkl/UnigramBOW.pkl','rb') as f_load:
            result = pickle.load(f_load)
            return result
    BagOfWordsExtractor = CountVectorizer(max_df=0.999, min_df=0.1, max_features=opt.maxNumFeatures,
                                      analyzer='char',
                                      binary=True, lowercase=True)

    BagOfWordsExtractor.fit(pd.concat([train.ix[:, 'sen1'], train.ix[:, 'sen2']]).unique())

    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Is creating unigram bow feature....')
    sen1_bow_rep = BagOfWordsExtractor.transform(sample.ix[:, 'sen1'])
    sen2_bow_rep = BagOfWordsExtractor.transform(sample.ix[:, 'sen2'])
    X = (sen1_bow_rep != sen2_bow_rep).astype(int)
    sample['f_bag_words'] = [X[i, :].toarray()[0] for i in range(0, len(sample))]
    for j in range(0, len(sample['f_bag_words'][0])):
        sample['z_bag_words' + str(j)] = [sample['f_bag_words'][i][j] for i in range(0, len(sample))]
    sample.fillna(0.0)
    now = datetime.datetime.now()
    print('unigtam bow feature is created~~')
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('*'* 80)
    colums = ['z_bag_words' + str(item) for item in range(len(sample['f_bag_words'][0]))]
    # colums.append('id')
    sample.to_csv('./dataset/charlevel_unigram_feature.csv',columns = colums)
    with open('./dataset/FeaturePkl/UnigramBOW.pkl','wb') as f_UnigramBOW:
        pickle.dump(sample[colums],f_UnigramBOW)
    return sample[colums]


def JaccarcUnigram(train,sample):
    def Jaccarc(sen1, sen2):
        tot_len = len(sen1) + len(sen2)
        same = 0
        for word1 in sen1:
            for word2 in sen2:
                if (word1 == word2):
                    same += 1
        return float(same) / (float(tot_len - same) + 0.000001)
    # if os.path.exists('')
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Is creating Jaccarc Unigram similarity feature....')
    def lexical(x):
        x = re.sub(r"\*", "", x)
        return x
    sample['sen1'] = sample['sen1'].apply(lexical)
    sample['sen2'] = sample['sen2'].apply(lexical)
    jac_result = []
    for item1,item2 in zip(sample['sen1'],sample['sen2']):
        jac_result.append(Jaccarc(item1,item2))
    sample['jaccarc_1gram'] = pd.Series(jac_result)
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Jaccarc Unigram similarity feature is finished')
    print('*' * 80)
    return pd.DataFrame(sample['jaccarc_1gram'])
def LengthCrossFeature(train,sample):

    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Is creating length feature... ')
    long_sen = []
    short_sen = []
    sum_lenth_sen = []

    for item1,item2 in zip(sample['sen1'],sample['sen2']):
        dif_value = max(len(item1),len(item2)) - min(len(item1),len(item2))
        long_sen.append(dif_value / max(len(item1),len(item2)))
        short_sen.append(dif_value / min(len(item1),len(item2)))
        sum_lenth_sen.append(dif_value / (len(item1) + len(item2)))
    long_sen = pd.Series(long_sen)
    short_sen = pd.Series(short_sen)
    sum_lenth_sen = pd.Series(sum_lenth_sen)

    result = pd.concat([long_sen,short_sen,sum_lenth_sen],axis=1,keys=['dif_rate_long','dif_rate_short','dif_rate_sum'])

    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('The length feature is done~~')
    print('*'*80)
    return result



def DiceUnigram(train,sample):
    def Dice(q1words, q2words):
        tot = len(q1words) + len(q2words) + 1
        same = 0
        for word1 in q1words:
            for word2 in q2words:
                if (word1 == word2):
                    same += 1
        return 2 * float(same) / (float(tot) + 0.000001)
    # if os.path.exists('')
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Is creating Dice Unigram similarity feature....')

    def lexical(x):
        x = re.sub(r"\*", "", x)
        return x

    sample['sen1'] = sample['sen1'].apply(lexical)
    sample['sen2'] = sample['sen2'].apply(lexical)
    dice_result = []
    for item1, item2 in zip(sample['sen1'], sample['sen2']):
        dice_result.append(Dice(item1, item2))
    sample['dice_1gram'] = pd.Series(dice_result)
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Dice Unigram similarity feature is finished')
    print('*' * 80)
    return pd.DataFrame(sample['dice_1gram'])


def OchiaiUnigram(train,sample):
    def Ochiai(q1words, q2words):
        tot = len(q1words) * len(q2words) + 1
        same = 0
        for word1 in q1words:
            for word2 in q2words:
                if (word1 == word2):
                    same += 1
        return float(same) / (np.sqrt(float(tot)) + 0.000001)

    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Is creating Ochiai Unigram similarity feature....')

    def lexical(x):
        x = re.sub(r"\*", "", x)
        return x

    sample['sen1'] = sample['sen1'].apply(lexical)
    sample['sen2'] = sample['sen2'].apply(lexical)
    ochiai_result = []
    for item1, item2 in zip(sample['sen1'], sample['sen2']):
        ochiai_result.append(Ochiai(item1, item2))
    sample['ochiai_1gram'] = pd.Series(ochiai_result)
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Ochiai Unigram similarity feature is finished')
    print('*' * 80)
    return pd.DataFrame(sample['ochiai_1gram'])
def TfIdfUnigram(train,sample): #with the implement of tiidf,extract two kinds of distance feature
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Is creating tfidf feature... ')
    count_vecor = CountVectorizer(analyzer = 'char',max_features = opt.maxNumFeatures)
    tfidf_vector = TfidfTransformer()
    count_vecor.fit(pd.concat([train.ix[:, 'sen1'], train.ix[:, 'sen2']]).unique())
    corpus_sen1 = count_vecor.transform(sample['sen1'])
    tfidf_sen1 = tfidf_vector.fit_transform(corpus_sen1).toarray()
    corpus_sen2 = count_vecor.transform(sample['sen2'])
    tfidf_sen2 = tfidf_vector.fit_transform(corpus_sen2).toarray()
    cos_dist = []
    euclidean_dist = []


    for item1 ,item2 in zip(tfidf_sen1,tfidf_sen2):
        cos_dist.append(CosDist(item1,item2))
        euclidean_dist.append(EuclideanDist(item1,item2))
    sample['cos_dist'] = pd.Series(cos_dist)
    sample['euc_dist'] = pd.Series(euclidean_dist)

    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('tfidf feature is done ')
    with open('./dataset/FeaturePkl/TfIdfUnigram.pkl','wb') as f_tfidf:
        pickle.dump(sample[['cos_dist','euc_dist']],f_tfidf)
    return sample[['cos_dist','euc_dist']]

def SVD_TfIdf(train,sample):
    # if os.path.exists('./dataset/FeaturePkl/svd.pkl'):
    #     with open('./dataset/FeaturePkl/svd.pkl') as f:
    #         result = pickle.load(f)
    #         return result
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Is running SVD... ')


    count_vecor = CountVectorizer(analyzer='char', max_features=opt.maxNumFeatures)
    tfidf_vector = TfidfTransformer()
    count_vecor.fit(pd.concat([train.ix[:, 'sen1'], train.ix[:, 'sen2']]).unique())
    corpus_sen1 = count_vecor.transform(sample['sen1'])
    tfidf_sen1 = tfidf_vector.fit_transform(corpus_sen1).toarray()
    corpus_sen2 = count_vecor.transform(sample['sen2'])
    tfidf_sen2 = tfidf_vector.fit_transform(corpus_sen2).toarray()

    svd = TruncatedSVD(n_components= 150, n_iter=7)
    svd_sen1 = svd.fit_transform(np.array(tfidf_sen1))
    svd_sen2 = svd.fit_transform(np.array(tfidf_sen2))

    cos_result = []
    euc_result = []

    for item1,item2 in zip(svd_sen1,svd_sen2):
        cos_result.append(CosDist(svd_sen1,svd_sen2))
        euc_result.append(EuclideanDist(svd_sen1,svd_sen2))
    cos_result = pd.Series(cos_result)
    euc_result = pd.Series(euc_result)
    sample['svd_cos_dis'] = cos_result
    sample['svd_euc_dis'] = euc_result

    with open('./dataset/FeaturePkl/svd.pkl','wb') as f_svd:
        pickle.dump(sample[['svd_cos_dis','svd_euc_dis']],f_svd)
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('SVD is done ~~')
    return result















    










if __name__ == '__main__':
    fp_train = '../dataset/train.csv'
    fp_valid = '../dataset/valid.csv'
    fp_test = '../dataset/test.csv'
    train = pd.read_csv(fp_train)
    valid = pd.read_csv(fp_valid)
    test  = pd.read_csv(fp_test)
    sample = pd.concat([train,valid,test]).reset_index(drop = True)
    import pickle
    result = TfIdfUnigram(train,sample)
    with open('../dataset/feature.pkl','wb') as f:
        pickle.dump(result,f)

import pandas as pd
from config import opt
import numpy as np
import random
def SplitData(fp):
    df = data = pd.read_csv('atec_nlp_sim_train.csv',sep = '\t',\
                            header = None,names = ['id','sen1','sen2','label'])
    data_list = np.array(df).tolist()
    random.shuffle(data_list)
    lenth  = len(data_list)
    step = lenth // 10
    train_data = pd.DataFrame(data_list[:step * 6],columns =['id','sen1','sen2','label'])
    valid_data = pd.DataFrame(data_list[step*6:step*8],columns =['id','sen1','sen2','label'])
    test_data = pd.DataFrame(data_list[step*8:],columns =['id','sen1','sen2','label'])
    train_data['data_flag'] = 1
    valid_data['data_flag'] = 2
    test_data['data_flag'] = 3
    train_data.to_csv('../dataset/train.csv')
    valid_data.to_csv('../dataset/valid.csv')
    test_data.to_csv('../dataset/test.csv')

if __name__ == '__main__':
    SplitData(fp = '../dataset/atec_nlp_sim_train.csv')
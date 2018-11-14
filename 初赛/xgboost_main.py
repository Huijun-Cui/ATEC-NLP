# import xgboost as xgb
# from sklearn.linear_model import LogisticRegression
# from features import *
# from features.deepmodel import *
import pandas as pd
import numpy as np
import pickle
import datetime
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
        print('precision_rate ={} '.format(precision_rate))
        recall_rate = TP / (TP + FN + 0.00001)
        print('recall_rate ={} '.format(recall_rate))
        return 2 * precision_rate * recall_rate / (precision_rate + recall_rate + 0.00001),precision_rate,recall_rate
    result,precision_rate,recall_rate = evalue(y_target,y_pred)
    return result,precision_rate,recall_rate
def train(**kwargs):
    for k,v in kwargs:
        setattr(opt,k,v)
    train = pd.read_csv('./dataset/train.csv',index_col = 0)
    sample = pd.read_csv('./dataset/merged_data.csv',index_col = 0)
    result = get_feat(train,sample)
    result['data_flag'] = sample['data_flag']
    result['label'] = sample['label']
    train = result[(result['data_flag'] == 1)|(result['data_flag'] == 2)]
    valid = result[result['data_flag'] == 3]
    X_train_columns = set(train.columns)
    X_train_columns.remove('label')
    X_train_columns.remove('data_flag')
    X_train_columns = list(X_train_columns)


    X_valid_columns = set(valid.columns)
    X_valid_columns.remove('label')
    X_valid_columns.remove('data_flag')
    X_valid_columns = list(X_valid_columns)

    params = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'objective': 'binary:logistic',
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'max_depth': 8,
        'gamma': 0.005,
        # 'reg_alpha': 1,
        # 'reg_lambda': 10,
        'tree_method': 'exact'
    }


    clf = xgb.XGBClassifier(**params)
    # clf = LogisticRegression()

    # with open('./dataset/log.pkl','wb') as f_log:
    #     pickle.dump(train[X_train_columns],f_log)
    #     return
    train = train.fillna(0.5)
    clf.fit(np.array(train[X_train_columns]),np.array(train['label']))
    y_predic = clf.predict(np.array(valid[X_valid_columns]))
    with open('./dataset/test_pickle','wb') as f:
        pickle.dump(y_predic,f)

def test():
    fp_train = './dataset/train.csv'
    fp_valid = './dataset/valid.csv'
    fp_test = './dataset/test.csv'
    train = pd.read_csv(fp_train)
    valid = pd.read_csv(fp_valid)
    test = pd.read_csv(fp_test)
    sample = pd.concat([train, valid, test]).reset_index(drop=True)
    import pickle
    result = SVD_TfIdf(train, sample)
    with open('./dataset/svd.pkl', 'wb') as f:
        pickle.dump(result, f)

if __name__ == '__main__':
    # train()
    # Evalue()
    # test()
    CleanRawData()







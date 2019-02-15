import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold
import warnings
import gc
import time
import sys
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,roc_auc_score,roc_curve
import seaborn,numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

def measure_importance():
    train_data=pd.read_csv('data/train.csv')
    test_data=pd.read_csv('data/test.csv')
    features=[c for c in train_data.columns if c not in ['ID_code','target']]
    target=train_data['target']
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=2019)
    oof=np.zeros(len(train_data))
    predictions=np.zeros(len(test_data))
    features_importance_df=pd.DataFrame()
    param = {
        'num_leaves': 10,
        'max_bin': 119,
        'min_data_in_leaf': 11,
        'learning_rate': 0.02,
        'min_sum_hessian_in_leaf': 0.00245,
        'bagging_fraction': 1.0,
        'bagging_freq': 5,
        'feature_fraction': 0.05,
        'lambda_l1': 4.972,
        'lambda_l2': 2.276,
        'min_gain_to_split': 0.65,
        'max_depth': 14,
        'save_binary': True,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False,
    }
    print('start KFold {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    for fold_,(trn_index,val_index)  in enumerate(skf.split(train_data.values,target.values)):
        print('fold {}'.format(fold_))
        print('start-time {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        trn_d=lgb.Dataset(train_data.iloc[trn_index][features],label=target.iloc[trn_index])
        val_d=lgb.Dataset(train_data.iloc[val_index][features],label=target.iloc[val_index])
        clf=lgb.train(params=param,train_set=trn_d,num_boost_round=10000,valid_sets=[trn_d,val_d],
                      verbose_eval=1000,early_stopping_rounds=100)
        oof[val_index]=clf.predict(train_data.iloc[val_index][features],num_iteration=clf.best_iteration)
        fold_importance_df=pd.DataFrame()
        fold_importance_df['feature']=features
        fold_importance_df['importance']=clf.feature_importance()
        fold_importance_df['fold']=fold_+1
        features_importance_df=pd.concat([features_importance_df,fold_importance_df],axis=0)
        predictions+=clf.predict(test_data[features],num_iteration=int(clf.best_iteration/5))
        print('end-time {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print("score: {:<8.5f}".format(roc_auc_score(target, oof)))
    features_importance_df.to_csv('temp/feature_importance.csv')
    predictions_df=pd.DataFrame(columns=['test_ID','prediction'])
    predictions_df['test_ID']=test_data['ID_code']
    predictions_df['prediction']=predictions
    predictions_df.to_csv('temp/prediction.csv')

def measure_importance_plus():
    pred=pd.read_csv('temp/prediction.csv')
    feature_importance=pd.read_csv('temp/feature_importance.csv')
    pred['prediction']=pred['prediction'].apply(lambda x:1 if x>=2.5 else 0)
    pred=pred.drop(columns=['Unnamed: 0'])
    pred.to_csv('temp/prediction_v1.csv')




if __name__=='__main__':
    measure_importance_plus()



    




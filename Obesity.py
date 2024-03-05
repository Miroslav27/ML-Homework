
import h2o
from h2o.automl import H2OAutoML
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import StandardScaler
import optuna
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold,cross_val_score
from lightgbm import LGBMClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import json
# Create the scaler
scaler = StandardScaler()
# Initialize the OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown='ignore')
# Label Encoder
encoder = LabelEncoder()
from sklearn.base import BaseEstimator, TransformerMixin


def preprocessor(df):
    df["Gender"] = df['Gender'].map({'Male': 1, 'Female': 0})
    df["family_history_with_overweight"] = df["family_history_with_overweight"].map({'yes': 1, 'no': 0})
    df["FAVC"] = df["FAVC"].map({'yes': 1, 'no': 0})
    df["SMOKE"] = df["SMOKE"].map({'yes': 1, 'no': 0})
    df["SCC"] = df["SCC"].map({'yes': 1, 'no': 0})
    # Ordinal Encoding
    df["CALC"] = df["CALC"].map({"Always":1,'Frequently': 0.75,'Sometimes': 0.5, 'no': 0.25})
    df["CAEC"] = df["CAEC"].map({"Always":1,'Frequently': 0.75,'Sometimes': 0.5, 'no': 0.25})
    df["Age"]= df["Age"]*51
    df["Weight"] = df["Weight"]*100
    df["Height"] = df["Height"]*100
    # Transform the data
    # Fit and transform the OneHotEncoder on the training data
    if df.shape[0] > 1:  # if df is not a single row
        ohe.fit(df[["MTRANS",]])#'BMI_cat'
    transformed = ohe.transform(df[["MTRANS",]])#'BMI_cat'

        # Convert the transformed data into a DataFrame
    dummies = pd.DataFrame(transformed, columns=ohe.get_feature_names_out(["MTRANS",]))#'BMI_cat'

        # Concatenate the one-hot encoded columns to the DataFrame
    df = pd.concat([df.reset_index(drop=False), dummies.reset_index(drop=True)], axis=1).drop(["MTRANS",], axis=1)#,'BMI_cat',"BMI"'Weight','Height'])
    df.index=df["id"]
    return df.drop([
        "id",
    ],axis=1)

def summary(df):
    print(f'data shape: {df.shape}')
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values
    summ['%missing'] = df.isnull().sum().values / len(df)* 100
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    return summ

numeric_features = ['FCVC','NCP',]#'CH2O','FAF','TUE','CAEC','CALC','Age', 'Height', 'Weight']

train=pd.read_csv("kaggle/input/Obesity/train.csv", index_col="id")
df_extra=pd.read_csv("kaggle/input/Obesity/ObesityDataSet.csv")

# try 252 1651
def get_df(train=train,df_extra=df_extra,random_state=1651,sample_size=252):
    train = pd.read_csv("kaggle/input/Obesity/train.csv", index_col="id")
    df_extra = pd.read_csv("kaggle/input/Obesity/ObesityDataSet.csv")

    df_val=train.sample(sample_size,random_state=random_state)
    #train.drop(df_val.index, inplace=True)

    #df_extra = pd.concat([df_extra,df_extra], ignore_index=True)
    df_extra["id"]=df_extra.index+20758
    df_extra.index=df_extra["id"]
    df_extra.drop("id",axis=1,inplace=True)
    X=pd.concat([train,df_extra])
    y = X['NObeyesdad']
    y = encoder.fit_transform(y)
    X.drop("NObeyesdad",axis=1,inplace=True)

    X = preprocessor(X)
    scaler.fit(X[numeric_features])
    X[numeric_features] = scaler.transform(X[numeric_features])
    print("X:", X.shape)
    return X,y


X,y = get_df()
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42,test_size=0.2,)
test =pd.read_csv("kaggle/input/Obesity/train.csv",index_col="id")
X_test = preprocessor(test)
#print("X:",X.shape)



"""
def objective_lgbm(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
        'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-8, 1.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.0, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 1e-8, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'num_class': 7,
        'verbosity': -1,
    }

    model = LGBMClassifier(**params)
    #model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    scores = cross_val_score(model, X, y, cv=5)
    return scores.mean()


study = optuna.create_study(direction='maximize')
study.optimize(objective_lgbm, n_trials=200)
with open('best_params_lgbm.json', 'w') as f:
    json.dump(study.best_params, f)
print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))


import xgboost as xgb

dMatrix = xgb.DMatrix(data=X_train, label=y_train)  ## For xgboost based cross validation we are creating this DMatrix

def optimize(trial):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        'grow_policy': trial.suggest_categorical('grow_policy', ["depthwise", "lossguide"]),
        #'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 6, 16),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.25, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'objective': 'multi:softprob',
        'num_class': 7,
        'eval_metric': 'mlogloss',
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 40),
        #'num_boost_round': trial.suggest_int('num_boost_round', 500, 2000),
        #'early_stopping_rounds':40,
    }

    #xgbClassifier = XGBClassifier(**params)
    cv_results = xgb.cv(params, dtrain,  nfold=5, metrics="mlogloss",early_stopping_rounds=30,num_boost_round=trial.suggest_int('num_boost_round', 400, 2000), seed=0)
    return cv_results["test-mlogloss-mean"].values[-1]

# Create a study object and optimize the objective function
study = optuna.create_study(direction="minimize")
study.optimize(optimize, n_trials=100)
print(study.best_params)

# Print the best hyperparameters and their corresponding log loss
print("Best trial:")
trial = study.best_trial
with open('best_params_xgb.json', 'w') as f:
    json.dump(study.best_params, f)
print("Hyperparameters: {}".format(trial.params))
print("Log Loss: {:.4f}".format(trial.value))
"""

params_xgb0={
  'grow_policy': 'lossguide',
  'max_depth': 9,
  'learning_rate': 0.008335918739236352,
  'subsample': 0.9079943398401875,
  'colsample_bytree': 0.4458982129537323,
  'gamma': 0.043626322735708385,
  'reg_lambda': 0.0006556357396441031,
  'reg_alpha': 0.001521332316091334,
  'min_child_weight': 10,
  'n_estimators': 1955
}
params_xgb1 = {
  'grow_policy': 'depthwise',
  'max_depth': 12,
  'learning_rate': 0.009846133450528309,
  'subsample': 0.5821997079739659,
  'colsample_bytree': 0.38987040243090915,
  'gamma': 0.11176304677270593,
  'reg_lambda': 0.002285592470128509,
  'reg_alpha': 0.3728365616396322,
  'min_child_weight': 12,
  'n_estimators': 1822
}
params_xgb2 = {'grow_policy': 'lossguide',
               'n_estimators': 1220,
               'max_depth': 8,
               'learning_rate': 0.0075271004489529765,
               'subsample': 0.664488344717449,
               'colsample_bytree': 0.5259168659771589,
               'gamma': 1.0462026341034819e-06,
               'reg_lambda': 3.637228306403305e-06,
               'reg_alpha': 0.10878009064912963,
               'min_child_weight': 2}
params_xgb3 = {
  'grow_policy': 'lossguide',
  'max_depth': 8,
  'learning_rate': 0.009813499789831781,
  'subsample': 0.8745962595702094,
  'colsample_bytree': 0.5023515839918677,
  'gamma': 0.0033100711487706264,
  'reg_lambda': 0.0011988681442636167,
  'reg_alpha': 9.920863990482234e-05,
  'min_child_weight': 32,
  'n_estimators': 1888
}

params_xgb4 = {'grow_policy': 'depthwise',
               'n_estimators': 982,
               'learning_rate': 0.050053726931263504,
               'gamma': 0.5354391952653927,
               'subsample': 0.7060590452456204,
               'colsample_bytree': 0.37939433412123275,
               'max_depth': 23,
               'min_child_weight': 21,
               'reg_lambda': 9.150224029846654e-08,
               'reg_alpha': 5.671063656994295e-08
              }
params_xgb5 = {'learning_rate': 0.02764286542457336,
               'max_depth': 9,
               'subsample': 0.8342830987447667,
               'n_estimators': 636,
               'colsample_bytree': 0.4944341098958961,
               'reg_lambda': 0.8488907389256313,
               'reg_alpha': 0.7844346042103796,
               'min_child_weight': 7,
               'gamma': 0.003221866785727264}

#https://www.kaggle.com/code/gulnihall/eda-lgbm-xgb-catboost-optuna-stacking-c#-7|-Model-Development
params_xgb6 = {'booster' : 'gbtree',
                'objective': 'multi:softmax',
                'verbosity' : 0,
                'tree_method' : "hist",
                'max_depth': 6,
                'learning_rate': 0.04497658670208754,
                'n_estimators': 366,
                'gamma': 0.002800024141530852,
                'min_child_weight': 2.480933321536105e-07,
                'subsample': 0.768632799287001,
                'colsample_bytree': 0.5526098554131518,
                'reg_alpha': 0.0010473611571215269,
                'reg_lambda': 0.49607928387141825,
                'eval_metric': 'mlogloss'}

params_xgb7 = {
  'grow_policy': 'depthwise',
  'n_estimators': 1158,
  'max_depth': 13,
  'learning_rate': 0.0077587811702919795,
  'subsample': 0.8070717130286618,
  'colsample_bytree': 0.43299344764174735,
  'gamma': 1.5709811812896947e-05,
  'reg_lambda': 5.647141816901323e-06,
  'reg_alpha': 6.762799004977035e-06,
  'min_child_weight': 11
}
params_xgb8 = {
  'grow_policy': 'depthwise',
  'n_estimators': 1166,
  'max_depth': 14,
  'learning_rate': 0.007570593954789896,
  'subsample': 0.7970081543540385,
  'colsample_bytree': 0.44386775843022674,
  'gamma': 8.33640319817041e-06,
  'reg_lambda': 0.0001581815907009602,
  'reg_alpha': 7.663463123541296e-06,
  'min_child_weight': 11
}
params_xgb9 = {
  'grow_policy': 'depthwise',
  'n_estimators': 950,
  'max_depth': 13,
  'learning_rate': 0.00996316354928434,
  'subsample': 0.7892385707620094,
  'colsample_bytree': 0.447599014750162,
  'gamma': 2.6806448913854925e-05,
  'reg_lambda': 7.062347060173951e-05,
  'reg_alpha': 4.786349647835046e-05,
  'min_child_weight': 14
}
params_xgb10 = {
  'grow_policy': 'depthwise',
  'n_estimators': 1399,
  'max_depth': 14,
  'learning_rate': 0.0076925518472302935,
  'subsample': 0.8161027375358113,
  'colsample_bytree': 0.4408775227150271,
  'gamma': 0.00012040079685435656,
  'reg_lambda': 0.00011621442696565732,
  'reg_alpha': 1.1135168562054327e-05,
  'min_child_weight': 11
}
params_lgbm0 = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "verbosity": -1,
    "boosting_type": "gbdt",
    #"random_state": random_state,
    "num_class": 7,
    'learning_rate': 0.026288471489806956,
    'n_estimators': 476,
    'lambda_l1': 0.06453656361518342,
    'lambda_l2': 0.3055301874218116,
    'max_depth': 9,
    'colsample_bytree': 0.4204715418169824,
    'subsample': 0.8720200608473527,
    'min_child_samples': 15}
#https://www.kaggle.com/code/moazeldsokyx/pgs4e2-highest-score-lgbm-hyperparameter-tuning/notebook
params_lgbm1 = {
    "objective": "multiclass",          # Objective function for the model
    "metric": "multi_logloss",          # Evaluation metric
    "verbosity": -1,                    # Verbosity level (-1 for silent)
    "boosting_type": "gbdt",            # Gradient boosting type
    "random_state": 42,       # Random state for reproducibility
    "num_class": 7,                     # Number of classes in the dataset
    'learning_rate': 0.030962211546832760,  # Learning rate for gradient boosting
    'n_estimators': 500,                # Number of boosting iterations
    'lambda_l1': 0.009667446568254372,  # L1 regularization term
    'lambda_l2': 0.04018641437301800,   # L2 regularization term
    'max_depth': 10,                    # Maximum depth of the trees
    'colsample_bytree': 0.40977129346872643,  # Fraction of features to consider for each tree
    'subsample': 0.9535797422450176,    # Fraction of samples to consider for each boosting iteration
    'min_child_samples': 26             # Minimum number of data needed in a leaf
}
params_lgbm2 = {
  'num_leaves': 190,
  'min_data_in_leaf': 77,
  'learning_rate': 0.0600224787270186,
  'min_sum_hessian_in_leaf': 0.0011998948593426365,
  'feature_fraction': 0.5822708469193747,
  'lambda_l1': 3.141453230725767e-08,
  'lambda_l2': 1.216403975028404e-05,
  'min_gain_to_split': 0.23489544037037496,
  'max_depth': 11,
  "verbosity": -1,

}
# https://www.kaggle.com/code/chinmayadatt/comparative-study-of-performance-0-92015#LGBM
params_lgbm3 = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "random_state": 42,
    "num_class": 7,
    'learning_rate': 0.031,
    'n_estimators': 550,
    'lambda_l1': 0.010,
    'lambda_l2': 0.040,
    'max_depth': 20,
    'colsample_bytree': 0.413,
    'subsample': 0.97,
    'min_child_samples': 25,
    'class_weight':'balanced'
}
#https://www.kaggle.com/code/gulnihall/eda-lgbm-xgb-catboost-optuna-stacking-c#-7|-Model-Development
params_lgbm4= {
    "objective": "multiclass",          # Objective function for the model
    "metric": "multi_logloss",          # Evaluation metric
    "verbosity": -1,                    # Verbosity level (-1 for silent)
    "boosting_type": "gbdt",            # Gradient boosting type
    "random_state": 42,       # Random state for reproducibility
    "num_class": 7,                     # Number of classes in the dataset
    'learning_rate': 0.030962211546832760,  # Learning rate for gradient boosting
    'n_estimators': 500,                # Number of boosting iterations
    'lambda_l1': 0.009667446568254372,  # L1 regularization term
    'lambda_l2': 0.04018641437301800,   # L2 regularization term
    'max_depth': 10,                    # Maximum depth of the trees
    'colsample_bytree': 0.40977129346872643,  # Fraction of features to consider for each tree
    'subsample': 0.9535797422450176,    # Fraction of samples to consider for each boosting iteration
    'min_child_samples': 26             # Minimum number of data needed in a leaf
}
params_lgbm5 = {
  'num_leaves': 173,
  'min_data_in_leaf': 68,
  'learning_rate': 0.07768622709261129,
  'min_sum_hessian_in_leaf': 0.00014661414175381319,
  'feature_fraction': 0.5851711280736737,
  'lambda_l1': 4.0318160434224576e-05,
  'lambda_l2': 9.512807755058468e-06,
  'min_gain_to_split': 0.0032679346082665375,
  'max_depth': 9,
  "verbosity": -1,
}
params_cat0={
    'iterations': 1000,
    'learning_rate': 0.13762007048684638,
    'depth': 5,
    'l2_leaf_reg': 5.285199432056192,
    'bagging_temperature': 0.6029582154263095,
    'verbose': 0,
}
params_cat1={'objective'           : 'MultiClass',
                          'eval_metric'         : "Accuracy",
                          'bagging_temperature' : 0.4,
                          'colsample_bylevel'   : 0.65,
                          'iterations'          : 1000,
                          'learning_rate'       : 0.038,
                          'od_wait'             : 12,
                          'max_depth'           : 5,
                          'l2_leaf_reg'         : 0.70,
                          'min_data_in_leaf'    : 9,
                          'random_strength'     : 0.175,
                          'max_bin'             : 100,
                          'verbose'             : 0,
                          "grow_policy"         : "Lossguide",
            }
params_cat2 = {'iterations': 953,
                'depth': 6,
                'learning_rate': 0.09824912127635886,
                'random_strength': 18,
                'bagging_temperature': 0.027325563890724644,
                'border_count': 159,
                'l2_leaf_reg': 0.5810014509693859,
                'verbose': 0
              }
params_cat3 = {'learning_rate': 0.14,
              'depth': 5,
              'l2_leaf_reg': 5.3,
              'bagging_temperature': 0.6,
              'iterations':1500,
              'random_seed': 42,
              'verbose': 200}
estimators = [
    ('xgb0', XGBClassifier(**params_xgb0)),
    ('xgb1', XGBClassifier(**params_xgb1)),
    ('xgb2', XGBClassifier(**params_xgb2)),
    ('xgb3', XGBClassifier(**params_xgb3)),
    ('xgb4', XGBClassifier(**params_xgb4)),
    ('xgb5', XGBClassifier(**params_xgb5)),
    ('xgb6', XGBClassifier(**params_xgb6)),
    ('xgb7', XGBClassifier(**params_xgb7)),
    ('xgb8', XGBClassifier(**params_xgb8)),
    ('xgb9', XGBClassifier(**params_xgb9)),
    ('lgbm0', LGBMClassifier(**params_lgbm0)),
    ('lgbm1', LGBMClassifier(**params_lgbm1)),
    ('lgbm2', LGBMClassifier(**params_lgbm2)),
    ('lgbm3', LGBMClassifier(**params_lgbm3)),
    ('lgbm5', LGBMClassifier(**params_lgbm5)),
    ('cat0', CatBoostClassifier(**params_cat0)),
    ('cat1', CatBoostClassifier(**params_cat1)),
    ('cat2', CatBoostClassifier(**params_cat2)),

]



from sklearn.model_selection import GridSearchCV
import random

estimators = [
    #('xgb0', XGBClassifier(**params_xgb0)),
    #('xgb1', XGBClassifier(**params_xgb1)),
    #('xgb2', XGBClassifier(**params_xgb2)),
    #('xgb3', XGBClassifier(**params_xgb5)),
    #('xgb10', XGBClassifier(**params_xgb10)),
    #('xgb4', XGBClassifier(**params_xgb4)),
    ('xgb5', XGBClassifier(**params_xgb5)),
    #('xgb7', XGBClassifier(**params_xgb7)),
    #('xgb9', XGBClassifier(**params_xgb9)),
    ('lgbm0', LGBMClassifier(**params_lgbm0)),
    ('lgbm1', LGBMClassifier(**params_lgbm1)),
    #('lgbm2', LGBMClassifier(**params_lgbm2)),
    #('lgbm5', LGBMClassifier(**params_lgbm5)),
    #('cat0',CatBoostClassifier(**params_cat0)),
    #('cat1',CatBoostClassifier(**params_cat1)),
    #('cat2',CatBoostClassifier(**params_cat2)),
    #('cat3',CatBoostClassifier(**params_cat3)),

]
#for name,model in estimators:
#    model.fit(X_train,y_train)
#    scores = cross_val_score(model, X, y, cv=5)
#    print(name,model.score(X_val, y_val),f"{name}: Cross-validation score: {scores.mean()}")

# Define your estimators and their hyperparameters
voting_clf = VotingClassifier(estimators=estimators, voting='soft')
voting_clf.fit(X_train,y_train)
print(voting_clf.score(X_val, y_val))
def objective(trial):
    # Define the thresholds as hyperparameters
    thresholds = [trial.suggest_float(f'threshold_{i}', 0.1, 0.9) for i in range(7)]

    # Use the thresholds to make predictions
    probs = voting_clf.predict_proba(X_val)
    preds = np.zeros_like(probs)
    for i in range(7):
        preds[:, i] = (probs[:, i] > thresholds[i]).astype(int)

    # Choose the class with the highest probability after thresholding
    final_preds = np.argmax(preds, axis=1)

    # Calculate the accuracy score
    score = accuracy_score(y_val, final_preds)

    return score

# Create the Optuna study
study = optuna.create_study(direction='maximize')

# Run the optimization
study.optimize(objective, n_trials=3600)

# Get the best thresholds
best_thresholds = [study.best_params[f'threshold_{i}'] for i in range(7)]
#{'threshold_0': 0.3176389630476676, 'threshold_1': 0.40020807071070735, 'threshold_2': 0.301916356068547, 'threshold_3': 0.563831809217096, 'threshold_4': 0.46876782581380544, 'threshold_5': 0.45239717449811845, 'threshold_6': 0.1414812765876114}.
#0.9116746829908177 and parameters: {'threshold_0': 0.3213022167336726, 'threshold_1': 0.3483958725890254, 'threshold_2': 0.3458940713792042, 'threshold_3': 0.5591032111996349, 'threshold_4': 0.695938356929664, 'threshold_5': 0.43381123555047296, 'threshold_6': 0.17856544421826767}
print(best_thresholds)
"""
def objective(trial):
    # Suggest a number of classifiers to use
    n_classifiers = trial.suggest_int('n_classifiers',3, 4)

    # Randomly choose n_classifiers from the estimators list
    chosen_estimators = random.sample(estimators, n_classifiers)

    # Create a new voting classifier with the chosen estimators
    voting_clf = VotingClassifier(estimators=chosen_estimators, voting='soft')

    # Suggest weights for the chosen classifiers
    weights = []
    for i, estimator in enumerate(chosen_estimators):
        weight = trial.suggest_float(estimator[0], 0.1, 1.0)
        weights.append(weight)
    # Set the weights and fit the classifier
    voting_clf.set_params(weights=weights)
    # voting_clf.fit(X_train, y_train)
    X, y = get_df(
        sample_size=444,#trial.suggest_int("sample",128,256), #0.9156298773690079
        random_state=1035520,#trial.suggest_int("random_state",1,20000)
    )
    # Predict and calculate accuracy
    # y_pred = voting_clf.predict(X_val)
    # accuracy = accuracy_score(y_val, y_pred)
    scores = cross_val_score(voting_clf, X, y, cv=5)
    return scores.mean()


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_weights = study.best_params
print(f"Best weights: {best_weights}")
"""
"""


def objective(trial):
    # Suggest a number of classifiers to use
    n_classifiers = trial.suggest_int('n_classifiers', 3, 4)

    # Randomly choose n_classifiers from the estimators list
    chosen_estimators = random.sample(estimators, n_classifiers)

    # Create a new voting classifier with the chosen estimators
    voting_clf = VotingClassifier(estimators=chosen_estimators, voting='soft')

    # Suggest weights for the chosen classifiers
    weights = []
    for i, estimator in enumerate(chosen_estimators):
        weight = trial.suggest_float(estimator[0], 0.1, 1.0)
        weights.append(weight)
    # Set the weights and fit the classifier
    voting_clf.set_params(weights=weights)
    voting_clf.fit(X_train, y_train)

    # Predict and calculate accuracy
    y_pred = voting_clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=120)

best_weights = study.best_params
print(f"Best weights: {best_weights}")
with open('best_params_voting.json', 'w') as f:
    json.dump(study.best_params, f)
print("Hyperparameters: {}".format(study.params))
print("Log Loss: {:.4f}".format(study.value))
"""
"""
h2o.init()

# Assuming X_train is your features and y_train is your target variable
df_train = X_train.copy()
df_train['target'] = y_train

# Convert pandas dataframe to h2o frame
h2o_df = h2o.H2OFrame(df_train)

# Specify the target and features
y = 'target'
x = h2o_df.columns
x.remove(y)

# Run AutoML for 20 base models
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=h2o_df)

# View the AutoML Leaderboard
lb = aml.leaderboard
print(lb.head(rows=lb.nrows))  # Print all rows instead of default (10 rows)

# The leader model is stored here
aml.leader
"""
"""
"""
"""
import catboost as cb
from optuna.integration import CatBoostPruningCallback
def objective_cat(trial):

    param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "used_ram_limit": "3gb",
        "eval_metric": "Accuracy",
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

    gbm = cb.CatBoostClassifier(**param)
    X, y = get_df(
        sample_size=444,  # trial.suggest_int("sample",128,256), #0.9156298773690079
        random_state=1035520,  # trial.suggest_int("random_state",1,20000)
    )
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2)
    pruning_callback = CatBoostPruningCallback(trial, "Accuracy")
    gbm.fit(
        train_x,
        train_y,
        eval_set=[(valid_x, valid_y)],
        verbose=0,
        early_stopping_rounds=100,
        callbacks=[pruning_callback],
    )

    # evoke pruning manually.
    pruning_callback.check_pruned()

    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(valid_y, pred_labels)

    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective_cat, n_trials=5)
with open('best_params_cat.json', 'w') as f:
    json.dump(study.best_params, f)
print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))
"""
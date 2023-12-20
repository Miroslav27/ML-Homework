import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss


#Load DataSet
df=pd.read_csv("kaggle/input/titanic/train.csv")# From Kaggle Titanic competition, should be here.
#Filling NAN with 0
df.fillna(-1000,inplace=True)
#Checking data
print(df.shape,df.head(),df.info(),df.describe(),df.count(),)

#Labels
y=df["Survived"]
# Dataset without labels
X=df[["Pclass","Sex","Age","SibSp","Parch","Fare"]]
# OnehotEncoding with Pandas
X["Sex"]=pd.get_dummies(X["Sex"],dtype=int).drop("male",axis=1)

print(X.info(),X.head(),X.count(),X.corrwith(y))


#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.12,)
# Using pipeline from SKLEARN docs
model = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2"))
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # probabilty of 1 for log_loss

# Accuracy and logloss from sklearn metrics
accuracy = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_pred_proba)

print(f'Accuracy: {accuracy}')
print(f'Log Loss: {loss}')

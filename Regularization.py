
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV

import matplotlib.pyplot as plt
import joblib



ds = pd.read_csv("kaggle/input/kc-house-data/kc_house_data.csv") #Folder Should be created!!!

print(ds.shape, ds.describe(), ds.head(), ds.info(), ds.isna().sum(), sep="\n")

ds = ds.fillna(0)
ds.isna().sum()
ds.drop("date", axis=1, inplace=True)
ds.head()
columns = abs(ds.corrwith(ds.price)) > 0.2  # those columns will have impact
X = ds.loc[:, columns].drop("price",axis=1)
y= ds.loc[:,"price"]/1000

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#print(X_test_scaled[:5])

alpha=0.05
models=[Lasso(alpha=alpha),Ridge(alpha=alpha),ElasticNet(alpha=alpha)]
for model in models:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f'{model.__str__()} Accuracy: {model.score(X_test_scaled, y_test):.6f}  MSE: {mean_squared_error(y_pred,y_test):.6f}')



#  train/test/cv split
X_train_less, X_cv,y_train_less,y_cv = train_test_split(X_train,y_train, test_size=0.2, random_state=42)

# Ridge Regression with CV
d_list = [0,1, 2, 3,4]
ridge_cv_model = RidgeCV(alphas=[1.0], cv=5)

scores_degree={}
# Features polynomial traansformation
for d in d_list:
    poly = PolynomialFeatures(degree=d)
    X_cv_poly = poly.fit_transform(X_cv)
    scores = cross_val_score(ridge_cv_model, X_cv_poly, y_cv, cv=5, scoring='neg_mean_squared_error')
    print(f'Degree {d}, Mean Squared Error: {-scores.mean()}')
    scores_degree[d]=-scores.mean()

best_degree= min(scores_degree, key=scores_degree.get)

# best Poly model on train dataset
poly = PolynomialFeatures(degree=best_degree)
X_train_poly = poly.fit_transform(X_train)
ridge_cv_model.fit(X_train_poly, y_train)

# Test
X_test_poly = poly.transform(X_test)
mse_test = mean_squared_error(y_test, ridge_cv_model.predict(X_test_poly))
print(f'Best Degree: {best_degree}, Test Mean Squared Error: {mse_test}')
joblib.dump(ridge_cv_model, f'kaggle/working/{ridge_cv_model.__str__()}.joblib')
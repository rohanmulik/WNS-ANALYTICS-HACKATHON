# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 13:15:10 2018

@author: AJIT MULIK
"""

import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data_train = pd.read_csv('train_LZdllcl.csv')

data_test = pd.read_csv('test_2umaH9m.csv')

print(data_train.shape)
print(data_train.describe())
print(data_train.info())


data_train["previous_year_rating"] = data_train["previous_year_rating"].fillna('3')
data_test["previous_year_rating"] = data_test["previous_year_rating"].fillna('3')
data_train["education"] = data_train["education"].fillna("Bachelor's")
data_test["education"] = data_test["education"].fillna("Bachelor's")




labelEnc=LabelEncoder()
cat_vars=['department','education',"gender","recruitment_channel",]
for col in cat_vars:
    data_train[col]=labelEnc.fit_transform(data_train[col])
    data_test[col]=labelEnc.fit_transform(data_test[col])

data_train.head()

from sklearn import preprocessing

std_scale = preprocessing.StandardScaler().fit(data_train[['age', 'length_of_service','avg_training_score','previous_year_rating']])
data_train[['age', 'length_of_service','avg_training_score','previous_year_rating']] = std_scale.transform(data_train[['age', 'length_of_service','avg_training_score','previous_year_rating']])

std_scale = preprocessing.StandardScaler().fit(data_test[['age', 'length_of_service','avg_training_score','previous_year_rating']])
data_test[['age', 'length_of_service','avg_training_score','previous_year_rating']] = std_scale.transform(data_test[['age', 'length_of_service','avg_training_score','previous_year_rating']])

predictors = ["department","gender", "recruitment_channel", "no_of_trainings","age",
              "previous_year_rating", "length_of_service","KPIs_met >80%","awards_won?","avg_training_score"]



from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(28,28,28),max_iter=500)
mlp.fit(data_train[predictors], data_train["is_promoted"])
MLPClassifier(activation='sigmoid', alpha=0.03, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(28, 28, 28), learning_rate='constant',
       learning_rate_init=0.025, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=50,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)
cv = ShuffleSplit(n_splits=10, test_size=0.4, random_state=50)
scores = cross_val_score(mlp, data_train[predictors], data_train["is_promoted"], scoring='f1',cv=cv)
print(scores.mean())

predictions = ["department", "gender","recruitment_channel","no_of_trainings","age",
              "previous_year_rating", "length_of_service","KPIs_met >80%","awards_won?","avg_training_score"]

eclf1 = VotingClassifier(estimators=[
        ('mlp',mlp)], voting='soft')
eclf1 = eclf1.fit(data_train[predictors], data_train["is_promoted"])
predictions=eclf1.predict(data_train[predictors])
predictions

test_predictions=eclf1.predict(data_test[predictors])

test_predictions=test_predictions.astype(int)
submission = pd.DataFrame({
        "employee_id": data_test["employee_id"],
        "is_promoted": test_predictions
    })

submission.to_csv("submission_dummy.csv", index=False)
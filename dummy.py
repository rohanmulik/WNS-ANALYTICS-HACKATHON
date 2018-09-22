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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


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



#RANDOM FOREST##########################################



predictors = ["department","gender", "recruitment_channel","no_of_trainings","age",
              "previous_year_rating", "length_of_service","KPIs_met >80%","awards_won?","avg_training_score"]

rf = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, 
                            min_samples_leaf=1)
kf = KFold(data_train.shape[0], n_folds=5, random_state=1)
cv = ShuffleSplit(n_splits=10, test_size=0.4, random_state=50)

predictions = cross_validation.cross_val_predict(rf, data_train[predictors],data_train["is_promoted"],cv=kf)
predictions = pd.Series(predictions)
scores = cross_val_score(rf, data_train[predictors], data_train["is_promoted"],
                                          scoring='f1', cv=kf)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

rf = RandomForestClassifier(random_state=1, n_estimators=100, max_depth=12,min_samples_split=6, min_samples_leaf=4)
rf.fit(data_train[predictors],data_train["is_promoted"])
kf = KFold(data_train.shape[0], n_folds=5, random_state=1)
predictions = cross_validation.cross_val_predict(rf, data_train[predictors],data_train["is_promoted"],cv=kf)
predictions = pd.Series(predictions)
scores = cross_val_score(rf, data_train[predictors], data_train["is_promoted"],scoring='f1', cv=kf)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
########################################################################





selector = SelectKBest(f_classif, k=5)
selector.fit(data_train[predictors], data_train["is_promoted"])

scores = -np.log10(selector.pvalues_)
indices = np.argsort(scores)[::-1]
sorted_important_features=[]
for i in indices:
    sorted_important_features.append(predictors[i])
from sklearn import cross_validation

# Initialize our algorithm
lr = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  
cv = ShuffleSplit(n_splits=10, test_size=0.4, random_state=50)
scores = cross_val_score(lr, data_train[predictors], data_train["is_promoted"], scoring='f1',cv=cv)
print(scores.mean())




adb=AdaBoostClassifier()
adb.fit(data_train[predictors],data_train["is_promoted"])
cv = ShuffleSplit(n_splits=10, test_size=0.4, random_state=50)
scores = cross_val_score(adb, data_train[predictors], data_train["is_promoted"], scoring='f1',cv=cv)
print(scores.mean())

#########################################################
#SVC MODEL

svc = SVC()
svc.fit(data_train[predictors], data_train["is_promoted"])
cv = ShuffleSplit(n_splits=10, test_size=0.4, random_state=50)
scores = cross_val_score(svc, data_train[predictors], data_train["is_promoted"], scoring='f1',cv=cv)
print(scores.mean())

#######################################################

perceptron = Perceptron()
perceptron.fit(data_train[predictors], data_train["is_promoted"])
cv = ShuffleSplit(n_splits=10, test_size=0.4, random_state=50)
scores = cross_val_score(perceptron, data_train[predictors], data_train["is_promoted"], scoring='f1',cv=cv)
print(scores.mean())

#######################################################


decision_tree = DecisionTreeClassifier()
decision_tree.fit(data_train[predictors], data_train["is_promoted"])
cv = ShuffleSplit(n_splits=10, test_size=0.4, random_state=50)
scores = cross_val_score(decision_tree, data_train[predictors], data_train["is_promoted"], scoring='f1',cv=cv)
print(scores.mean())

###################################################


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
        ('lr', lr), ('rf', rf), ('adb', adb),('mlp',mlp),('decision_tree',decision_tree)], voting='soft')
eclf1 = eclf1.fit(data_train[predictors], data_train["is_promoted"])
predictions=eclf1.predict(data_train[predictors])
predictions

test_predictions=eclf1.predict(data_test[predictors])

test_predictions=test_predictions.astype(int)
submission = pd.DataFrame({
        "employee_id": data_test["employee_id"],
        "is_promoted": test_predictions
    })

submission.to_csv("submission.csv", index=False)

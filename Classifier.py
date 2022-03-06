#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 12:22:04 2022

@author: jayanthvasanthkumar
"""

#Imporing the default library set
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

#Import the preprocessing libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix

#Import the ML model libraries
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

#Import the model evaluation libraries
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

path = "/Users/jayanthvasanthkumar/Documents/Files/Personal Projects/Practice/Titanic/"
testFile = "test.csv"
trainFile = "train.csv"
testResultsFile = "gender_submission.csv"


#Information about the features:
'''
PassengerID = ID of Passenger (int)
Survived = Status of survival (0 or 1)
* Pclass = Passenger class (1=First; 2=Second; 3=Third)
Name = Passenger name (String)
Sex = Gender of passenger (String)
Age = Age of passenger (int)
SibSp = Number of siblings/spouses aboard (int)
Parch = Number of parents/children abroad (int)
Ticket = Ticket number
Fare = Passenger fare (GBP) (int)
Cabin = Cabin
* Embarked = Port of embarkation (C=Cherbourg; Q=Queenstown; S=Southampton) (String)

* = Categorical feature
'''
#================================================
#Loading the dataset
#================================================

def loadData(path, fileName):
    csv_path = os.path.join(path, fileName)
    return pd.read_csv(csv_path)

def showHist(data):
    data.hist(bins=10, figsize=(10,10))
    plt.show()
    
def showScatter(X, Y, S, C, data):
    data.plot(kind='scatter', x=X, y=Y, alpha=0.1,
            s=data[S], label=S, figsize=(10,7), 
            c=data[C], cmap=plt.get_cmap("jet"), colorbar="True",
            )
    plt.legend()
    
def showMatrixPlot(attributes, data):
    scatter_matrix(data[attributes], figsize=(12,8))
    

train = loadData(path, trainFile)

test = loadData(path, testFile)
testResults = loadData(path, testResultsFile)
sel_testResults = testResults.iloc[:, 1:]

test_combined = pd.concat([test, sel_testResults], axis=1)

train = train[[c for c in train if c not in ['Survived']] + ['Survived']]

full_dataset = pd.concat([train, test_combined])

#================================================
#Preprocessing the datasets
#================================================

#>> REMOVAL OF 'UNRELATED' FEATURES

d = full_dataset.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

#>> IMPUTATION

simpleImputer = SimpleImputer(strategy="most_frequent")
simpleImputer.fit(d)
d_t = simpleImputer.transform(d)

#>> TEXTUAL/CATEGORICAL DATA HANDLING

#OneHotEncode the 2nd and 7th column (Sex, Embarkation) Independent variables
transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 6])], 
                                remainder='passthrough'
                                )
d_te = pd.DataFrame(transformer.fit_transform(d_t))

#================================================
#Splitting the data into training and test sets
#================================================

X = d_te.iloc[:, :10]
y = d_te.iloc[:, 10]

y = y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state = 1
                                                    )

#================================================
#Feature Scaling for the Independant variables
#================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#================================================
#Training the ML models
#================================================

def __plot_pr_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper right")
    plt.ylim([0, 1])
    
def __plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.axis([0, 1, 0, 1])
    
def precisionThresholdCurve(model, X_train, y_train, y_scores):
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
    __plot_pr_vs_threshold(precisions, recalls, thresholds)
    plt.show()

def precisionRecallCurve(model, X_train, y_train, y_scores):
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
    __plot_precision_vs_recall(precisions, recalls)
    plt.show()
    
def showROC(model, X_train, y_train, y_scores):
    fpr, tpr, thresholds = roc_curve(y_train, y_scores)
    plt.plot(fpr, tpr, linewidth=2, label=None)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1.01])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.show()
    print("AUC score: ",roc_auc_score(y_train, y_scores))
    

#Each model function returns a tuple containing:
#   (confusion matrix, [accuracy, precision, recall, f1])

#>> LOGISTIC REGRESSION CLASSIFIER
def logReg(X_train, y_train):
    log_reg = LogisticRegression(random_state=0)
    log_reg.fit(X_train, y_train)
    
    y_pred = log_reg.predict(X_test)
    
    conf_mat = confusion_matrix(y_test, y_pred)
    
    #Evaluation metrics
    score = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    y_scores = cross_val_predict(log_reg, X_train, y_train, cv=5, method='decision_function')
    
    precisionRecallCurve(log_reg, X_train, y_train, y_scores)
    
    precisionThresholdCurve(log_reg, X_train, y_train, y_scores)
    
    showROC(log_reg, X_train, y_train, y_scores)
    
    return (conf_mat, [score, precision, recall, f1])
#c, a = logReg(X_train, y_train)


#>> GAUSSIAN NAIVE BAYES CLASSIFIER
def gaussianNB(X_train, y_train):
    gaussNB = GaussianNB()
    gaussNB.fit(X_train, y_train)
    
    y_pred = gaussNB.predict(X_test)
    
    conf_mat = confusion_matrix(y_test, y_pred)
    
    #Evaluation metrics
    score = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    y_scores = cross_val_predict(gaussNB, X_train, y_train, cv=5, method='predict_proba')
    y_scores = y_scores[:, 1] #<- make sure to only keep the proba of positive class (second column)
    
    precisionRecallCurve(gaussNB, X_train, y_train, y_scores)
    
    precisionThresholdCurve(gaussNB, X_train, y_train, y_scores)
    
    showROC(gaussNB, X_train, y_train, y_scores)
    
    return (conf_mat, [score, precision, recall, f1])

#c2, a2 = gaussianNB(X_train, y_train)
    
#>> PERCEPTRON CLASSIFIER
def perceptron(X_train, y_train):
    sgd = SGDClassifier(max_iter=5, 
                        tol=None, 
                        random_state=42, 
                        loss='perceptron', 
                        eta0=1, 
                        learning_rate='constant', 
                        penalty=None
                        )
    sgd.fit(X_train, y_train)
    
    y_pred = sgd.predict(X_test)
    
    conf_mat = confusion_matrix(y_test, y_pred)
    
    #Evaluation metrics
    score = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    y_scores = cross_val_predict(sgd, X_train, y_train, cv=5, method='decision_function')

    
    precisionRecallCurve(sgd, X_train, y_train, y_scores)
    
    precisionThresholdCurve(sgd, X_train, y_train, y_scores)
    
    showROC(sgd, X_train, y_train, y_scores)
    
    return (conf_mat, [score, precision, recall, f1])

#>> UNCOMMENT THE BELOW LINES TO RUN THE RESPECTIVE CLASSIFIER MODEL
#>> (cn, an) is the confusion matrix and evaluation scores tuple returned for the nth model

#c3, a3 = perceptron(X_train, y_train)
#c2, a2 = gaussianNB(X_train, y_train)
#c1, a1 = logReg(X_train, y_train)




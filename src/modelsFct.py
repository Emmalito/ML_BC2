"""
    Source code of the different functions
    used in the notebook 3_Models
"""
#Classical Libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Importation of the model libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# Tools'libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump


def getLogisticModel(X_train, Inc_train, Acc_train):
    """Search the best hyperparameter for a logistic model"""

    #Hyper-parameters to optimise
    paramGrid = {'penalty': ['none', 'l1', 'l2'], #, #Regularisation
                'C': [0.01, 0.1, 0.7, 1, 5, 10, 15],  #inverse of regularization strength
                'fit_intercept': [False,True] }  #Intercept

    #IncomeInvestment model
    randomIncLogMod = GridSearchCV(estimator=LogisticRegression(solver='saga', n_jobs=-1, class_weight={0:0.6, 1:0.4}), cv=15,
                                    param_grid=paramGrid, n_jobs=-1).fit(X_train, Inc_train)
    logIncMod = randomIncLogMod.best_estimator_

    #AccumulationInvestment model
    randomAccLogMod = GridSearchCV(estimator=LogisticRegression(solver='saga', n_jobs=-1), cv=15,
                                    param_grid=paramGrid, n_jobs=-1).fit(X_train, Acc_train)
    logAccMod = randomAccLogMod.best_estimator_

    return logIncMod, logAccMod


def getRandomForestModel(X_train, Inc_train, Acc_train):
    """Search the best hyperparameter for a random forest model"""

    #Hyper-parameters to optimise
    paramGrid = {'n_estimators': [50, 100, 300], #Number of trees
                'criterion': ['gini', 'entropy'],  #Information gain measure
                'max_depth': [None, 5, 10, 50],  #Maximum depth of the tree
                'max_features': ['sqrt','auto', 'log2'],  #Number of features for the best split
                'bootstrap': [True, False]}  #Bootstrap samples are used

    #IncomeInvestment model
    rf_randomInc = RandomizedSearchCV(estimator = RandomForestClassifier(n_jobs=-1, class_weight={0:0.6, 1:0.4}), n_iter=144, cv=15, param_distributions = paramGrid, n_jobs = -1)
    rf_randomInc.fit(X_train, Inc_train)
    rfModInc = rf_randomInc.best_estimator_

    #AccumulationInvestment model
    rf_randomAcc = RandomizedSearchCV(estimator = RandomForestClassifier(n_jobs=-1), n_iter=144, cv=15, param_distributions = paramGrid, n_jobs = -1)
    rf_randomAcc.fit(X_train, Acc_train)
    rfModAcc = rf_randomAcc.best_estimator_

    #Save the models
    dump(rfModInc, 'RandomForestInc.joblib')
    dump(rfModAcc, 'RandomForestAcc.joblib')


def getANNModel(X_train, Inc_train, Acc_train):
    """Compute an Artificial Neural Network model"""

    #Parameters
    dim, batch = len(X_train[0]), len(X_train)//100
    nnIncMod, nnAccMod = Sequential(), Sequential()

    #IncomeInvestment model
    nnIncMod.add(Dense(units=512, activation='relu', input_dim=dim))
    nnIncMod.add(Dense(units=1, activation='sigmoid'))#output layer
    nnIncMod.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['Recall', 'AUC', 'accuracy'])
    nnIncMod.fit(X_train, Inc_train, batch_size=batch, epochs=60, verbose=1, class_weight={0:0.6, 1:0.4}, use_multiprocessing=True) #, validation_data=(X_test, Inc_test))

    #AccumulationInvestment model
    nnAccMod.add(Dense(units=512, activation='relu', input_dim=dim))
    nnAccMod.add(Dense(units=1, activation='sigmoid'))#output layer
    nnAccMod.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['Recall', 'AUC', 'accuracy'])
    nnAccMod.fit(X_train, Acc_train, batch_size=batch, epochs=100, verbose=1, use_multiprocessing=True) #, validation_data=(X_test, Acc_test))

    return nnIncMod, nnAccMod


def metrics(cm, isInc):
    """Return on the output standard, the metrics of the predictions"""

    TN, FP = cm[0][0], cm[0][1]
    FN, TP = cm[1][0], cm[1][1]
    TPR = TP/(TP+FN) ; P = TP/(TP+FP) 
    TNR = (TN/(TN+FP)) ; F1 = (2*P*TPR)/(P+TPR)
    acc = (TP+TN)/(TP+TN+FP+FN)
    Z = 0.6*TPR + 0.4*acc  #Business metric => 60% for the TPR, 40% for the Acc
    if isInc:
        print("Measures for IncomeInvestment:")
    else:
        print("Measures for AccumulationInvestment")
    print(f'Sensitivity = {TPR:2.2%} ;  Specificity = {TNR:2.2%} ; Precision = {P:2.2%} ; F1 score = {F1:2.2%} ; Accuracy = {acc:2.2%} ; Zmetric = {Z:2.2%}', "\n")


def confusionMat(predInc, predAcc, Inc_test, Acc_test):
    """Print the confusion matrix of a prediction"""

    cm1 = confusion_matrix(Inc_test, predInc)
    cm2 = confusion_matrix(Acc_test, predAcc)

    #Plot
    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts1 = ["{0:0.0f}".format(value) for value in cm1.flatten()]
    group_counts2 = ["{0:0.0f}".format(value) for value in cm2.flatten()]
    group_percentages1 = ["{0:.2%}".format(value) for value in cm1.flatten()/np.sum(cm1)]
    group_percentages2 = ["{0:.2%}".format(value) for value in cm2.flatten()/np.sum(cm2)]
    labels1 = np.asarray([f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts1,group_percentages1)]).reshape(2,2)
    labels2 = np.asarray([f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts2,group_percentages2)]).reshape(2,2)
    sns.heatmap(cm1, annot=labels1, fmt="", cmap='Blues', ax=axes[0])
    sns.heatmap(cm2, annot=labels2, fmt="", cmap='Blues', ax=axes[1])
    axes[0].set_title("IncomeInvestment")
    axes[1].set_title("AccumulationInvestment")

    #Metrics
    metrics(cm1, 1)
    metrics(cm2, 0)
    return

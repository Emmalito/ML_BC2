"""
    Source code of the function for the annexe notebook
"""

#Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix



def pcaPlot(X_train):
    """Apply a PCA transformation and plot the
    explained variance in function of the PCA"""

    pca = PCA()
    pca.fit(X_train) # we fit the dataset 
    print('first principal component direction:', pca.components_[:,0]) # we extract the principal components directions

    explained = pca.explained_variance_ratio_  # and the corresponding explained variance
    print('explained variance:', explained)

    T = pca.transform(X_train)  #transform into PC space

    explained_variance = np.cumsum(explained) / sum(explained)
    explained_variance = np.insert(explained_variance, 0, 0.)

    plt.figure(figsize=(12,5))
    plt.plot(range(len(explained_variance)), explained_variance)
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    plt.xticks(range(len(explained_variance)))
    plt.yticks(np.arange(0., 1.1, 0.1))
    plt.show()


def pcaDataPlot(T, Acc_train, Inc_train):
    need, no_need = T[Acc_train==1], T[Acc_train==0]  #Accumulation need and no need
    need_inc, no_need_inc = T[Inc_train==1], T[Inc_train==0]  #Income need and no need
    fig, axis = plt.subplots(2, 2, figsize=(15,10))

    axis[0,0].scatter(need[:, 0], need[:, 1], label='Acc need')
    axis[0,0].scatter(no_need[:, 0], no_need[:, 1], label='Acc no need', marker='x')
    axis[0,0].set(xlabel='pc1', ylabel='pc2')
    axis[0,0].grid()
    axis[0,0].legend()

    axis[0,1].scatter(need[:, 2], need[:, 3], label='Acc need')
    axis[0,1].scatter(no_need[:, 0], no_need[:, 2], label='Acc no need', marker='x')
    axis[0,1].set(xlabel='pc3', ylabel='pc4')
    axis[0,1].grid()
    axis[0,1].legend()

    axis[1,0].scatter(need_inc[:, 0], need_inc[:, 1], label='Inc need')
    axis[1,0].scatter(no_need_inc[:, 0], no_need_inc[:, 1], label='Inc no need', marker='x')
    axis[1,0].set(xlabel='pc1', ylabel='pc2')
    axis[1,0].grid()
    axis[1,0].legend()

    axis[1,1].scatter(need_inc[:, 2], need_inc[:, 3], label='Inc need')
    axis[1,1].scatter(no_need_inc[:, 2], no_need_inc[:, 3], label='Inc no need', marker='x')
    axis[1,1].set(xlabel='pc3', ylabel='pc4')
    axis[1,1].grid()
    axis[1,1].legend()

    plt.show()


def metrics(cm, isInc):
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0] 
    TP = cm[1][1]
    TPR = TP/(TP+FN) ; P = TP/(TP+FP) 
    TNR = (TN/(TN+FP)) ; F1 = (2*P*TPR)/(P+TPR)
    acc = (TP+TN)/(TP+TN+FP+FN)
    if isInc:
        print("Measures for IncomeInvestment:")
    else:
        print("Measures for AccumulationInvestment")
    print(f'Sensitivity = {TPR:2.2%} ;  Specificity = {TNR:2.2%} ; Precision = {P:2.2%} ; F1 score = {F1:2.2%} ; Accuracy = {acc:2.2%}', "\n")


def confusionMat(predInc, predAcc, Inc_test, Acc_test):
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

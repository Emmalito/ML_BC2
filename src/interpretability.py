"""
    Source code for the differents functions used
    in the notebook Interpretability
"""

#Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap

from sklearn_extra.cluster import KMedoids as KM



def dependenceShap(shapValues, X_test, columns):
    """Plot the dependance plot for the dataset features"""

    figure, axes = plt.subplots(2,2, figsize=(20, 8))
    figure.suptitle("Title")
    shap.dependence_plot(0, shapValues, X_test, ax=axes[0,0], feature_names=columns, show=False)
    shap.dependence_plot(1, shapValues, X_test, ax=axes[0,1], feature_names=columns, show=False)
    shap.dependence_plot(2, shapValues, X_test, ax=axes[1,0], feature_names=columns, show=False)
    shap.dependence_plot(3, shapValues, X_test, ax=axes[1,1], feature_names=columns, show=False)
    plt.show()


def plotInertia(Acc_true, Inc_true):
    """Plot the inertia elbow"""

    figure, axes = plt.subplots(1, 2, sharex=False, figsize=(15,5))
    figure.suptitle("Accumulation : Inertia : Income")
    numberCluster = range(1, 15)
    inertia_a = [KM(i).fit(Acc_true).inertia_ for i in numberCluster]
    inertia_i = [KM(i).fit(Inc_true).inertia_ for i in numberCluster]
    axes[0].plot(numberCluster, inertia_a, '-bo')
    axes[1].plot(numberCluster, inertia_i, '-ro')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia (within-cluster sum of squares)')
    plt.show()


def medoids(nb_cluster, train, data_std, data_mean):
    """Compute Kmedoids on the given number of cluster and look 
    how samples in the train set are distributed in the cluster
    Return the classes of samples"""

    medoids = KM(nb_cluster).fit(train)
    labels = medoids.predict(train)
    clusters = medoids.cluster_centers_*data_std[[0,2,3]] + data_mean[[0,2,3]]
    print("cluster centers are :")
    for clust in clusters:
        print("Age:", clust[0], "; Income:", clust[1], "; Wealth:", clust[2])
    a, count = np.unique(labels, return_counts=True)
    print("classes",a,"distributed as :", count)
    return labels

def repartition(labels, num_cluster, train, data_std, data_mean):
    """Given the distribution of samples in the cluster (labels), 
    return a dataframe of the sample of cluster num_cluster where feature are not scale"""
    first = train[labels[:]==num_cluster]
    distribution = pd.DataFrame(first*data_std[[0,2,3]] + data_mean[[0,2,3]], columns=["Age","logIncome", "logWealth"])
    return distribution


def boxplot_cluster(clusters):
    """Plot boxplot for our three features"""

    figure, axes = plt.subplots(len(clusters), 3, sharex=False, figsize=(30,8))
    figure.suptitle("boxplot of our feature")
    for i in range(len(clusters)):
        sns.boxplot(x=clusters[i]["Age"], ax=axes[i,0])
        sns.boxplot(x=clusters[i]["logIncome"],ax=axes[i,1])
        sns.boxplot(x=clusters[i]["logWealth"],ax=axes[i,2])

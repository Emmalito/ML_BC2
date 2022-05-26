"""
    Source code of the different functions that
    we use in the notebook 2_Preprocessing
"""

from tkinter import E
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as DT



def bacwardWrapper(X_train, X_test, t_train, t_test, nameTarget):
    """Backward wrapper method"""

    #feature order : {0:"Age", 1:"Gender", 2:"FamilyMembers", 3:"FinancialEducation", 4:"RiskPropensity", 5:"logIncome", 6: "logWealth"}
    classes = {0:"Age", 1:"Gender", 2:"FamilyMembers", 3:"FinancialEducation", 4:"RiskPropensity", 5:"logIncome", 6: "logWealth"}
    nb_classes = len(classes)

    # on accumulation investment :
    mod =DT()
    mod.fit(X_train, t_train) #Model with all the feature
    print("full accuracy for", nameTarget, ": ", mod.score(X_test, t_test))
    for i in range(nb_classes): #For each feature
        list = [j for j in range(nb_classes) if j!=i]
        mod_i = DT()
        mod_i.fit(X_train[:,list], t_train) #We train a model without it
        print("without ", classes[i], "and with the rest", mod_i.score(X_test[:,list], t_test))


def forwardFeatureSelect(features, X_train, X_test, t_train, t_test, isInc=1):
    """Forward Stepwise Selection function"""

    datas, datasTest = [], []
    indexFeat, accTotal = [], []
    for step_i in range(len(features)):
        accMax, ind = -1, features[0]
        for index in features:
            feat, featTest = X_train[:,index], X_test[:,index]
            train = np.array((datas + [feat]))
            test = np.array((datasTest + [featTest]))
            mod = None
            if isInc :
                mod = DT(class_weight={0:0.6, 1:0.4}) #Because the target is not balanced
            else:
                mod = DT()
            mod.fit(train.transpose(), t_train)
            acc = mod.score(test.transpose(), t_test)
            if accMax < acc:
                accMax = acc
                ind = index
        indexFeat.append(ind)
        accTotal.append(accMax)
        datas += [X_train[:,ind]]
        datasTest += [X_test[:,ind]]
        features.remove(ind)
    return indexFeat, accTotal


def runForward(X_train, X_test, t_train, t_test, isInc, isTest=0):
    """Run the forward wrapper """

    if isTest :
        withAll, withoutGender = [0, 1, 2, 3, 4, 5], [0, 2, 3, 4, 5]
        withoutFamMem, withoutBoth = [0, 1, 3, 4, 5], [0, 3, 4, 5]
    else :
        withAll, withoutGender = [0, 1, 2, 3, 4, 5, 6], [0, 2, 3, 4, 5, 6]
        withoutFamMem, withoutBoth = [0, 1, 3, 4, 5, 6], [0, 3, 4, 5, 6]

    print("with all:", forwardFeatureSelect(withAll, X_train, X_test, t_train, t_test, isInc))
    print("without Gender: ", forwardFeatureSelect(withoutGender, X_train, X_test, t_train, t_test, isInc))
    print("without FamMem: ", forwardFeatureSelect(withoutFamMem, X_train, X_test, t_train, t_test, isInc))
    print("without both: ", forwardFeatureSelect(withoutBoth, X_train, X_test, t_train, t_test, isInc))
    return 


def plot_feature_importance(importance, importanceOld, names, namesOld, modelName):
    """function to plot feature importance of a Random Forest"""

    #Create arrays from feature importance and feature names
    feature_importance, feature_importance_old = np.array(importance), np.array(importanceOld)
    feature_names, feature_names_old = np.array(names), np.array(namesOld)

    #Create a DataFrame using a Dictionary
    data={'FEATURE NAMES':feature_names,'FEATURE IMPORTANCE':feature_importance}
    fi_df = pd.DataFrame(data)

    dataOld={'FEATURE NAMES':feature_names_old,'FEATURE IMPORTANCE':feature_importance_old}
    fi_dfOld = pd.DataFrame(dataOld)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['FEATURE IMPORTANCE'], ascending=False,inplace=True)
    fi_dfOld.sort_values(by=['FEATURE IMPORTANCE'], ascending=False,inplace=True)

    #Define size of bar plot
    figure, axes = plt.subplots(1, 2, sharex=True, figsize=(23, 5))

    #Plot Searborn bar chart
    sns.barplot(x=fi_df['FEATURE IMPORTANCE'], y=fi_df['FEATURE NAMES'], ax=axes[0])
    sns.barplot(x=fi_dfOld['FEATURE IMPORTANCE'], y=fi_dfOld['FEATURE NAMES'], ax=axes[1])

    #Add chart labels
    figure.suptitle(modelName)

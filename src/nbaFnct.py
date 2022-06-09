"""
    Source code of the different functions that
    we use in the notebook 4_NextBestAction
"""

# Library
from src.modelsFct import predictionEnsemble



def incomePred(mean, std, model, data):
    """Predict an income need thanks to the income
    model that we build. Return the class prediction"""

    data = (data-mean)/std #We normalize the data
    return model.predict(data.reshape(1, -1))


def accumulationPred(mean, std, model, data):
    """Predict an income need thanks to the income
    model that we build. Return the class prediction"""

    data = (data-mean)/std #We normalize the data
    return predictionEnsemble(model, data.reshape(1, -1)) > 0.28


def NBA(Rc, productType):
    """"""

    index=0
    D_lessrisky, P_lessrisky = [], []
    D_morerisky, P_morerisky = [], []

    for index in range(len(productType)):
        if Rc-productType["Risk"].values[index] >= 0:
            D_lessrisky.append((Rc-productType["Risk"].values[index]))
            P_lessrisky.append(productType.index.values[index])
        if Rc-productType["Risk"].values[index] < 0:
            D_morerisky.append((productType["Risk"].values[index]-Rc))
            P_morerisky.append(productType.index.values[index])
            
    if len(P_morerisky)>0 and len(P_lessrisky)>1:
        if D_lessrisky[1] < D_morerisky[-1]:
            D= [[1-D_lessrisky[0]/(D_lessrisky[0]+D_lessrisky[1]), 1-D_lessrisky[1]/(D_lessrisky[0]+D_lessrisky[1])],[0,0]]
            P= [[P_lessrisky[0], P_lessrisky[1]],[0,0]]
        else:
            D= [[(1-D_lessrisky[0]/(D_lessrisky[0]+D_morerisky[-1])),0],[(1-D_morerisky[-1]/(D_lessrisky[0]+D_morerisky[-1])),0]]
            P= [[P_lessrisky[0],0], [P_morerisky[-1],0]]
    elif len(P_lessrisky)==1 :
            D= [[1-D_lessrisky[0]/(D_lessrisky[0]+D_morerisky[-1]),0],[1-D_morerisky[-1]/(D_lessrisky[0]+D_morerisky[-1]),0]]
            P= [[P_lessrisky[0],0], [P_morerisky[-1],0]]
    elif len(P_lessrisky)==0 :
            D= [[0,0],[1-D_morerisky[-1]/(D_morerisky[-1]+D_morerisky[-2]), 1-D_morerisky[-2]/(D_morerisky[-1]+D_morerisky[-2])]]
            P= [[0,0],[P_morerisky[-1], P_morerisky[-2]]]
    elif len(P_morerisky)==0 :
            D= [[1-D_lessrisky[0]/(D_lessrisky[0]+D_lessrisky[1]), 1-D_lessrisky[1]/(D_lessrisky[0]+D_lessrisky[1])],[0,0]]
            P= [[P_lessrisky[0], P_lessrisky[1]],[0,0]]   
    return([D,P])


def recommandationSys(ClientRisk, incomeProd, accumulationProd, mean, std, modelInc, modelAcc, data):
    """Recommande products in case of need"""

    # Verify the need
    incomeNeed = incomePred(mean, std, modelInc, data)[0]
    accumulationNeed = accumulationPred(mean, std, modelAcc, data)[0]

    # Income product recommandation
    if incomeNeed:
        probas, products = NBA(ClientRisk, incomeProd)
        print("We predict you an income need.")
        # More risky products
        for index in range(2):
            if probas[0][index]:
                print("We recommande you the product more risky n°", products[0][index], "with a probability of", probas[0][index])
        # Less risky products
        for index in range(2):
            if probas[1][index]:
                print("We recommande you the product more risky n°", products[1][index], "with a probability of", probas[1][index])
    else:
        print("We do not predict an income need.")
    print()

    # Accumulation product recommandation
    if accumulationNeed:
        probas, products = NBA(ClientRisk, accumulationProd)
        print("We predict you an accumulation need.")
        # More risky products
        for index in range(2):
            if probas[0][index]:
                print("We recommande you the product more risky n°", products[0][index], "with a probability of", probas[0][index])
        # Less risky products
        for index in range(2):
            if probas[1][index]:
                print("We recommande you the product more risky n°", products[1][index], "with a probability of", probas[1][index])
    else:
        print("We do not predict an accumulation need.")

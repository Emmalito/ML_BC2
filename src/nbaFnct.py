"""
    Source code of the different functions that
    we use in the notebook 4_NextBestAction
"""

# Classical libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def NBA(Rc, productType):
    index=0
    D_lessrisky=[]
    P_lessrisky=[]
    D_morerisky=[]
    P_morerisky=[]
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

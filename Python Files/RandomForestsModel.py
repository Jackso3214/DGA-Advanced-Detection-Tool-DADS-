
#######
#######
####### CURRENTLY UNUSED
#######
#######

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

def logRegModel(dataset):
    DGAList = dataset
    # Replace infinite updated data with nan
    DGAList.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN
    DGAList.dropna(inplace=True)

    #Drop columns
    #Create a list of feature
    X = DGAList.drop(['Domain','Verdict', 'VN', "Unnamed: 0"], axis=1)
    #create a list of verdicts
    y = DGAList['VN']
    #print(X)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)

    #predict
    y_pred = model.predict(X_test)

    #evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)


    print("accuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1: ", f1)

    return model
    
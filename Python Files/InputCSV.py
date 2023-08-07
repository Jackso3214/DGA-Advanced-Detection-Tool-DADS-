import pandas as pd
import numpy as np
import DataPrep as prep
import LogisticRegressionModel as logreg
import UnsupervisedModels as unsup
import saveModel as sm
import csv
import datetime
from main import LoadModelOption


def bulkTestCSV(path):
        
    file = open(path)
    reader = csv.reader(file)
    print("Loading the currently trained model\n")
    trainedModel = LoadModelOption()

    rows = []

    #grab rows and put them in list of list
    for row in reader:
        rows.append(row)
    
    for row in rows:
        domain = row[0]

        #prepare the domain
        testing = prep.prepDataTest(domain)

        #get the verdict
        result = trainedModel.predict(testing)

        #calculate the probability of the domain
        resultProbility = trainedModel.predict_proba(testing)
        #index 1: probability of domain being legit
        #convert result to an numpy array
        resultProbility = np.array(resultProbility)
        #extract the values of the resulting array
        resultProbility = [resultProbility[-1][0], resultProbility[-1][1]]

        resultProbility[0] = round(float(resultProbility[0]), 5)
        resultProbility[1] = round(float(resultProbility[1]), 5)

        #append the row to the result
        row.append(result)
        row.append(resultProbility[0])
        row.append(resultProbility[1])

    return rows

    

def bulkTestOutputCSV(rows):
    filename = './bulkTestOutput'
    current_time = datetime.datetime.now()
    dateTime = ""
    dateTime += current_time.strftime("%Y%m%d")
    dateTime += "_" + current_time.strftime("%H%M%S")

    filename += dateTime + '.csv'

    field_names=['Domain','Verdict','% of legit','% of DGA']
    outputfile = open(filename, 'w')
    writer=csv.writer(outputfile)
    writer.writerow(field_names)
    writer.writerows(rows)

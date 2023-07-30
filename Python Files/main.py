
import pandas as pd
import numpy as np
import DataPrep as prep
import LogisticRegressionModel as logreg
import saveModel as sm
import os.path
import sys

#prepare the data and output it to a csv called outputTable750k.csv
#this is the main script for the program, run this if you want to test and run the program

def Help():
    print(
    " ===================================================================\n",
    "To see this help output again, use: \"help\" option\n",
    "To prepare a dataset for training the model, use: \"prep\" option\n",
    "To train the model with the current prepared dataset, use: \"train\" option\n",
    "To test the currently trained model, use: \"test [domain]\" option\n",
    "===================================================================\n",
    )

def PrepDataOption():
    prep.prepData()

def trainModelOption():
    dataset = pd.read_csv("CSVs\output\outputTable10k_set1.csv")
    trainedModel = logreg.logRegModel(dataset)
    sm.savePickle(trainedModel)

def LoadModelOption():
    trainedModel = 0
    if os.path.isfile("saveModel\\saved_object.pkl"):
        trainedModel = sm.loadPickle()
        
    return trainedModel
        

def TestDomain(domain):
    print("Loading the currently trained model:\n")
    trainedModel = LoadModelOption()
    print("Extracting features of domain: ", domain, "\n")
    testing = prep.prepDataTest(domain)

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

    print("===============================================================\n")
    if result == 1:
        print("The domain ", domain, "is predicted to be DGA Generated.\n\n")
    else:
        print("The domain ", domain, "is predicted to be legit.\n\n")
    
    print("===============================================================\n")
    print("The model predicts that ", domain, "has a ", resultProbility[0]*100, "% of being legit.\n")
    print("The model predicts that ", domain, "has a ", resultProbility[1]*100, "% of being DGA Generated.\n")
    print("===============================================================\n")


#command line interface
if __name__ == '__main__':
    args = sys.argv[1:]
    if args:
        function_name = args[0]
        if function_name == "help":
            Help()
        if function_name == "prep":
            PrepDataOption()
        if function_name == "train":
            trainModelOption()
        if function_name == "test":
            #check if arguments were given
            if not len(args) == 2:
                print("You did not provide a domain.\n")
            else:
                TestDomain(args[1])
    else:
        print("No arguments was provided. use option \"help\" to see list of possible arguments.")
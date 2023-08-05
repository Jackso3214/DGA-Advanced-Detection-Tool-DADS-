
import pandas as pd
import numpy as np
import DataPrep as prep
import LogisticRegressionModel as logreg
import saveModel as sm
import os.path
import sys
from argparse import ArgumentParser, RawTextHelpFormatter




#prepare the data and output it to a csv called outputTable750k.csv
#this is the main script for the program, run this if you want to test and run the program

def PrepDataOption(input="CSVs\\trainingDataset\\dga_dataset_10k_set4.csv", output="CSVs\output\outputDataset.csv"):
    prep.prepData(input, output)

def trainModelOption(input="CSVs\\output\\outputDataset.csv"):
    print("using location modified dataset to train model: ", input)
    dataset = pd.read_csv(input)
    trainedModel = logreg.logRegModel(dataset)
    sm.savePickle(trainedModel)

def LoadModelOption():
    trainedModel = 0
    if os.path.isfile("saveModel\\saved_object.pkl"):
        trainedModel = sm.loadPickle()
        
    return trainedModel
        

def TestDomain(domain):
    print("Loading the currently trained model\n")
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


#main menu for the command line interface
parser = ArgumentParser(description='POC of detecting DGA domains using machine learning.', 
                        formatter_class=RawTextHelpFormatter)

#if no arguments were given
if len(sys.argv) == 1:
    print("Use the -h or --help flag to see help.")
    exit()

#sub parsers
subparser = parser.add_subparsers()

#create a parser for the prep command
parser_prep = subparser.add_parser('prep', help="prep: Prepares the dataset")
parser_prep.add_argument('--dataset', type=str, 
                        help="used to set the file location of the dataset to train the model")
                        
#create a parser for the prep command
parser_train = subparser.add_parser('train', help="train: Trains the model with the current dataset")
parser_train.add_argument('--model', type=str, 
                         help="used to select the ML model to train")

#create a parser for the prep command
parser_test = subparser.add_parser('test', help="test: tests the currently trained model")
parser_test.add_argument('--domain', type=str, 
                         help="used to test a domain")

#take in the arguments
args = parser.parse_args()

if (sys.argv[1] == 'prep'):
    if (args.dataset):
        print("prepping the dataset to use.\n")
        PrepDataOption(args.dataset)
    else:
        print("you need to specify the location of the .csv file to train the model using the --dataset option.\nUsing the default dataset.\n")
        PrepDataOption()
elif (sys.argv[1] == 'train'):
    print("Training the model using the current dataset.\n")
    trainModelOption()
elif (sys.argv[1] == 'test'):
    if args.domain:
        print("testing the domain: ", args.domain)
        TestDomain(args.domain)
    else:
        print("you need to specify the domain to test using the --domain option.")



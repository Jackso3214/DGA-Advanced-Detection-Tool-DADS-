import pandas as pd
import DataPrep as prep
import LogisticRegressionModel as logreg
import saveModel as sm

#prepare the data and output it to a csv called outputTable750k.csv
#this is the main script for the program, run this if you want to test and run the program

prep.prepData()

dataset = pd.read_csv("CSVs\output\outputTable10k_set1.csv")
trainedModel = logreg.logRegModel(dataset)
sm.savePickle(trainedModel)

testing = prep.prepDataTest("microsoft.com")
result = trainedModel.predict(testing)
resultProbility = trainedModel.predict_proba(testing)
print(testing)
print(result)
print(resultProbility)



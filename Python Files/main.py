import pandas as pd
import DataPrep as prep
import LogisticRegressionModel as logreg

#prepare the data and output it to a csv called outputTable750k.csv

prep.prepData()

dataset = pd.read_csv("CSVs\output\outputTable10k.csv")
trainedModel = logreg.logRegModel(dataset)

testing = prep.prepDataTest("doh123145.test")
result = trainedModel.predict(testing)
print(testing)
print(result)



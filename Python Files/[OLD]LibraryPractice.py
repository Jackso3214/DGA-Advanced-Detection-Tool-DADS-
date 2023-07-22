import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
import pickle

#custom imports
import EntropyCalculator as ent
import VowelConsonantsCalculator as vowcon
import whoisExtractor as whois
import FeatureEngineering as feat

#load Dataset function
def load_DGA_data():
    return pd.read_csv("CSVs\\Testing_Set_Take_1.csv")

#load the dataset
DGAList = load_DGA_data()
#calculate size of dataset
#print(DGAList.head())
#print(DGAList.shape)

#create temporary lists
temp = []
vowelTemp = []
consTemp = []

#calculate and create an Entropy column
for ind in DGAList.index:
    temp.append(ent.Entropy(DGAList['Domain'][ind]))
DGAList['Entropy'] = temp

#calculate and create an vowerl and consonants column
temp.clear()
for ind in DGAList.index:
    temp.append(vowcon.VowelConsonants(DGAList['Domain'][ind]))
for val in temp:
    vowelTemp.append(val['vowel'])
    consTemp.append(val['consonant'])

DGAList['Vowel'] = vowelTemp
DGAList['Consonant'] = consTemp

#calculate and create an Length column
temp.clear()
for ind in DGAList.index:
    temp.append(len(DGAList['Domain'][ind]))
DGAList['Length'] = temp

#calculate and create an whoIs column Might scrap this idea
temp.clear()
#temp.append(whois.whoIs(DGAList['Domain'][1]))
#DGAList['whoIs'] = temp

#print(DGAList)

DGAList = DGAList.sample(frac=1)
print(DGAList)

DGAList['VN'] = DGAList['Verdict'].apply(lambda x: feat.DGAtoNum(x))
DGAList = DGAList.assign(VowelConsRatio=lambda x: round(x['Vowel'] / x["Consonant"], 5))
DGAList = DGAList.assign(EntropyLength=lambda x: round(x['Entropy'] / x["Length"], 5))
print(DGAList)

#df['Entropy'].hist()
#plt.scatter(DGAList['Entropy'], DGAList['VerdictNum'])

#plt.xlabel('Length')
#plt.ylabel('VerdictNum')
#plt.show()


#df_subset = DGAList.head(100000)
#sns.lmplot(x='Entropy', y='VerdictNum', data=df_subset, logistic=True)
#plt.show()


DGAList.to_csv("CSVs\outputTable750k.csv")


#Drop columns
#Create a list of feature
X = DGAList.drop(['Domain','Verdict', 'VN', ], axis=1)
#create a list of verdicts
y = DGAList['VN']

print(X)
print(y)

 
#split the data set 70% training, and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

#Train the model
model = LogisticRegression()
#model.fit(X_train, y_train)


#save the model
#s = pickle.dumps(model)
#
#coefficients = model.coef_
#
#avg_importance = np.mean(np.abs(coefficients), axis=0)
#feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': avg_importance})
#feature_importance = feature_importance.sort_values('Importance', ascending=True)
#feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))

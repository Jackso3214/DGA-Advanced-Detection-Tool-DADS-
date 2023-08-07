import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

def savePickle(trainedModel):
    
    data = trainedModel
    path = "saveModel\\supervisedTrained.pkl"
    with open(path, 'wb') as file:
        pickle.dump(data, file)
        file.close()

def loadPickle():
    path = "saveModel\\supervisedTrained.pkl"
    with open(path, 'rb') as file:
        savedPickle = pickle.load(file)
        file.close()
        return savedPickle
    

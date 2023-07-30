import pickle

def savePickle(trainedModel):
    
    data = trainedModel
    path = "saveModel\\saved_object.pkl"
    file_name = 'saved_object.pkl'
    with open(path, 'wb') as file:
        pickle.dump(data, file)
        file.close()

def savePickle(trainedModel):
    
    data = trainedModel
    path = "saveModel\\saved_object.pkl"
    file_name = 'saved_object.pkl'
    with open(path, 'wb') as file:
        pickle.dump(data, file)
        file.close()
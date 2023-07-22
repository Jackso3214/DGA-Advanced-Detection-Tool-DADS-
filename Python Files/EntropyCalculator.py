import math

def Entropy(string,base = 2.0):

    string = string.lower()
    #convert string into a dict for no repeats of characters, and to count the frequency
    characterFrequency = dict.fromkeys(list(string), 0)
    characterProbability = characterFrequency
    
    #calculate frequencies
    for character in string:
        if character in string:
            characterFrequency[character] += 1

    #calculate probabilities
    for character in characterFrequency:
        characterProbability[character] = characterFrequency[character] / len(string)
    #print(characterProbability)

    #calculate Entropy
    sumOfEntropy = 0
    for probabilities in characterProbability:
        probability = characterProbability[probabilities]
        sumOfEntropy += probability * math.log(1/probability, base)
    sumOfEntropy = round(sumOfEntropy, 5)
    #print(sumOfEntropy)
    
    return sumOfEntropy
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
import re
import pickle

#custom imports
import EntropyCalculator as ent
import VowelConsonantsCalculator as vowcon
import FeatureEngineering as feat
import DomainNgrams as dngrams
import NumCountCalculator as numCount

#load Dataset function
def load_DGA_data():
    return pd.read_csv("CSVs\\trainingDataset\\dga_dataset_10k_set1.csv")

def prepData():
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

    #create cosonant and vowel count columns
    DGAList['Vowel'] = vowelTemp
    DGAList['Consonant'] = consTemp

    #calculate and create an Length column
    temp.clear()
    for ind in DGAList.index:
        temp.append(len(DGAList['Domain'][ind]))
    DGAList['Length'] = temp

    DGAList = DGAList.sample(frac=1)
    #print(DGAList)

    #convert verdict to a number column
    DGAList['VN'] = DGAList['Verdict'].apply(lambda x: feat.DGAtoNum(x))
    DGAList = DGAList.assign(VowelConsRatio=lambda x: round(x['Vowel'] / x["Consonant"], 5))
    DGAList = DGAList.assign(EntropyLength=lambda x: round(x['Entropy'] / x["Length"], 5))
    #print(DGAList)

    #calculate numcount and create a column
    temp.clear()
    for ind in DGAList.index:
        temp.append(numCount.numCount(DGAList['Domain'][ind]))
    DGAList['CountNum'] = temp

    #calculate and create a ngram score column FOR LEGIT Ngrams
    #variables for the ngram scores
    temp.clear()
    totalScore = []
    score = [[],[],[],[]]
    tempTotalScore = 0
    tempScore = 0
    top2String, top3String, top4String, top5String = dngrams.topNgramsCalculation()
    #go through each domain
    for ind in DGAList.index:
        #calculate each score of each ngram
        testDomain = DGAList['Domain'][ind]
        testDomainSplit = re.split('.[A-Za-z]*\.*$', testDomain)[0]

        #check if one of the top 50 ngrams exists in domain and accumulate one if it is
        for ngram in top2String:
            if ngram[0] in testDomainSplit:
                tempTotalScore += 1
                tempScore += 1
        score[0].append(tempScore)
        tempScore = 0

        for ngram in top3String:
            if ngram[0] in testDomainSplit:
                tempTotalScore += 1
                tempScore += 1
        score[1].append(tempScore)
        tempScore = 0

        for ngram in top4String:
            if ngram[0] in testDomainSplit:
                tempTotalScore += 1
                tempScore += 1
        score[2].append(tempScore)
        tempScore = 0

        for ngram in top5String:
            if ngram[0] in testDomainSplit:
                tempTotalScore += 1
                tempScore += 1
        score[3].append(tempScore)
        totalScore.append(tempTotalScore)
        tempScore = 0
        tempTotalScore = 0
    
    #add new column
    DGAList['ngram2ScoreLegit'] = score[0]
    DGAList['ngram3ScoreLegit'] = score[1]
    DGAList['ngram4ScoreLegit'] = score[2]
    DGAList['ngram5ScoreLegit'] = score[3]
    DGAList['ngramTotalScoreLegit'] = totalScore



    #calculate and create a ngram score column FOR DGA Ngrams
    #variables for the ngram scores
    temp.clear()
    totalScore = []
    score = [[],[],[],[]]
    tempTotalScore = 0
    tempScore = 0
    top2String, top3String, top4String, top5String = dngrams.topNgramsCalculationDGA()
    #go through each domain
    for ind in DGAList.index:
        #calculate each score of each ngram
        testDomain = DGAList['Domain'][ind]
        testDomainSplit = re.split('.[A-Za-z]*\.*$', testDomain)[0]

        #check if one of the top 50 ngrams exists in domain and accumulate one if it is
        for ngram in top2String:
            if ngram[0] in testDomainSplit:
                tempTotalScore += 1
                tempScore += 1
        score[0].append(tempScore)
        tempScore = 0

        for ngram in top3String:
            if ngram[0] in testDomainSplit:
                tempTotalScore += 1
                tempScore += 1
        score[1].append(tempScore)
        tempScore = 0

        for ngram in top4String:
            if ngram[0] in testDomainSplit:
                tempTotalScore += 1
                tempScore += 1
        score[2].append(tempScore)
        tempScore = 0

        for ngram in top5String:
            if ngram[0] in testDomainSplit:
                tempTotalScore += 1
                tempScore += 1
        score[3].append(tempScore)
        totalScore.append(tempTotalScore)
        tempScore = 0
        tempTotalScore = 0
    
    #add new column
    DGAList['ngram2ScoreDGA'] = score[0]
    DGAList['ngram3ScoreDGA'] = score[1]
    DGAList['ngram4ScoreDGA'] = score[2]
    DGAList['ngram5ScoreDGA'] = score[3]
    DGAList['ngramTotalScoreDGA'] = totalScore

    

    
    #output to a file
    DGAList.to_csv("CSVs\output\outputTable10k.csv")




def prepDataTest(string):
    #assign string
    domain = string

    #create temporary lists
    

    #calculate and create an Entropy column
    entropy = ent.Entropy(domain)
    

    #calculate and create an vowerl and consonants column
    count = vowcon.VowelConsonants(domain)
    vowel = count['vowel']
    consonant = count['consonant']
    #calculate and create an Length column
    
    length = len(domain)

    #convert verdict to a number column
    vowelConsRatio = round(vowel / consonant, 5)
    entropyLenRatio = round(entropy / length, 5)

    #calculate numcount and create a column
    countNum = numCount.numCount(domain)

    #calculate and create a ngram score column FOR LEGIT NGRAMS
    
    top2String, top3String, top4String, top5String = dngrams.topNgramsCalculation()

    testDomain = re.split('.[A-Za-z]*\.*$', domain)[0]
    
    totalScore = 0
    score = [0,0,0,0]
    for ngram in top2String:
        if ngram[0] in testDomain:
            totalScore += 1
            score[0] += 1

    for ngram in top3String:
        if ngram[0] in testDomain:
            totalScore += 1
            score[1] += 1

    for ngram in top4String:
        if ngram[0] in testDomain:
            totalScore += 1
            score[2] += 1

    for ngram in top5String:
        if ngram[0] in testDomain:
            totalScore += 1
            score[3] += 1

    #calculate and create a ngram score column FOR DGA NGRAMS
    
    top2String, top3String, top4String, top5String = dngrams.topNgramsCalculationDGA()

    testDomain = re.split('.[A-Za-z]*\.*$', domain)[0]
    
    totalScoreDGA = 0
    scoreDGA = [0,0,0,0]
    for ngram in top2String:
        if ngram[0] in testDomain:
            totalScoreDGA += 1
            scoreDGA[0] += 1

    for ngram in top3String:
        if ngram[0] in testDomain:
            totalScoreDGA += 1
            scoreDGA[1] += 1

    for ngram in top4String:
        if ngram[0] in testDomain:
            totalScoreDGA += 1
            scoreDGA[2] += 1

    for ngram in top5String:
        if ngram[0] in testDomain:
            totalScoreDGA += 1
            scoreDGA[3] += 1
    
    #note: Domain,Verdict,Entropy,Vowel,Consonant,Length,VN,VowelConsRatio,EntropyLength,ngram2Score,ngram3Score,ngram4Score,ngram5Score,ngramTotalScore
    data = [[entropy, vowel, consonant, length, vowelConsRatio, entropyLenRatio, countNum ,score[0], score[1], score[2], score[3], totalScore, scoreDGA[0], scoreDGA[1], scoreDGA[2], scoreDGA[3], totalScoreDGA]]
    df = pd.DataFrame(data, columns=["Entropy", "Vowel", "Consonant", "Length", "VowelConsRatio", "EntropyLength", "CountNum" ,"ngram2ScoreLegit", "ngram3ScoreLegit", "ngram4ScoreLegit", "ngram5ScoreLegit", "ngramTotalScoreLegit", "ngram2ScoreDGA", "ngram3ScoreDGA", "ngram4ScoreDGA", "ngram5ScoreDGA", "ngramTotalScoreDGA"])
    return df
    #lmao
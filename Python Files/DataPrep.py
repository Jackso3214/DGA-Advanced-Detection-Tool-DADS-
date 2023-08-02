import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
import re
import os.path
import csv
import pickle

#custom imports
import EntropyCalculator as ent
import VowelConsonantsCalculator as vowcon
import FeatureEngineering as feat
import DomainNgrams as dngrams
import NumCountCalculator as numCount

#load Dataset function
def load_DGA_data(path):
    return pd.read_csv(path)

def Export_DGA_data(dataframe, location):
    print("outputted file to location: ", location)
    dataframe.to_csv(location)

def readNgram(path):
    temp = []
    with open(path) as file:
                    reader = csv.reader(file)
                    for row in reader:
                        temp.append(row)
                    file.close()
    return temp
    

def prepData(path, outputPath):
    #load the dataset
    DGAList = load_DGA_data(path)
    unmodifiedDGAList = DGAList
    #calculate size of dataset
    #print(DGAList.head())
    #print(DGAList.shape)

    #create temporary lists
    temp = []
    vowelTemp = []
    consTemp = []

    #calculate and create an Entropy column
    for ind in DGAList.index:
        testDomainSplit = re.split('.[A-Za-z]*\.*$', DGAList['Domain'][ind])[0]
        temp.append(ent.Entropy(testDomainSplit))
        #temp.append(ent.Entropy(DGAList['Domain'][ind]))
    DGAList['Entropy'] = temp
    

    #calculate and create an vowerl and consonants column
    temp.clear()
    for ind in DGAList.index:
        testDomainSplit = re.split('.[A-Za-z]*\.*$', DGAList['Domain'][ind])[0]
        temp.append(vowcon.VowelConsonants(testDomainSplit))
        #temp.append(vowcon.VowelConsonants(DGAList['Domain'][ind]))
    for val in temp:
        vowelTemp.append(val['vowel'])
        consTemp.append(val['consonant'])

    #create cosonant and vowel count columns
    DGAList['Vowel'] = vowelTemp
    DGAList['Consonant'] = consTemp

    #calculate and create an Length column
    temp.clear()
    for ind in DGAList.index:
        testDomainSplit = re.split('.[A-Za-z]*\.*$', DGAList['Domain'][ind])[0]
        temp.append(len(testDomainSplit))
        #temp.append(len(DGAList['Domain'][ind]))
    DGAList['Length'] = temp

    DGAList = DGAList.sample(frac=1)
    #print(DGAList)

    #convert verdict to a number column

    VowelConsRatioTemp = []

    #calculate ratio of vowel and consonant
    for index, row in DGAList.iterrows():
        if row['Consonant'] == 0 or row['Vowel'] == 0:
            VowelConsRatioTemp.append(0)
            continue
        else:
            VowelConsRatioTemp.append(round(row['Vowel'] / row["Consonant"], 5))
    

    DGAList['VN'] = DGAList['Verdict'].apply(lambda x: feat.DGAtoNum(x))
    DGAList['VowelConsRatio'] = VowelConsRatioTemp
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

    top2String = []
    top3String = []
    top4String = []
    top5String = []

   
    
    #if topNgrams exist in folder, read those otherwise generate it
    #generate top NGrams for legit
    #if file exists in the specific folder
    #top 2 ngrams
    #if folder exists
    if os.path.isdir("CSVs"):
        #if the csv file exists
        if os.path.isfile("CSVs\\topNgrams\\top2String.csv"):
            #read the csv file
            top2String = readNgram("CSVs\\topNgrams\\top2String.csv")
            top3String = readNgram("CSVs\\topNgrams\\top3String.csv")
            top4String = readNgram("CSVs\\topNgrams\\top4String.csv")
            top5String = readNgram("CSVs\\topNgrams\\top5String.csv")
        else:
            top2String, top3String, top4String, top5String = dngrams.topNgramsCalculation(unmodifiedDGAList)
            
    #if the folder doesnt exist
    else:
        #check to see if the csv file is in the same folder as python file
        if os.path.isfile("top2String.csv"):
            #read the file if it is
            top2String = readNgram("top2String.csv")
            top3String = readNgram("top3String.csv")
            top4String = readNgram("top4String.csv")
            top5String = readNgram("top5String.csv")
        else:
            top2String, top3String, top4String, top5String = dngrams.topNgramsCalculation(unmodifiedDGAList)
    
    
            
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

    #top 2 ngrams
    #if folder exists
    if os.path.isdir("CSVs"):
        #if the csv file exists
        if os.path.isfile("CSVs\\topNgrams\\top2StringDGA.csv"):
            #read the csv file
            top2String = readNgram("CSVs\\topNgrams\\top2StringDGA.csv")
            top3String = readNgram("CSVs\\topNgrams\\top3StringDGA.csv")
            top4String = readNgram("CSVs\\topNgrams\\top4StringDGA.csv")
            top5String = readNgram("CSVs\\topNgrams\\top5StringDGA.csv")
        else:
            top2String, top3String, top4String, top5String = dngrams.topNgramsCalculationDGA(unmodifiedDGAList)
            
    #if the folder doesnt exist
    else:
        #check to see if the csv file is in the same folder as python file
        if os.path.isfile("top2StringDGA.csv"):
            #read the file if it is
            top2String = readNgram("top2StringDGA.csv")
            top3String = readNgram("top3StringDGA.csv")
            top4String = readNgram("top4StringDGA.csv")
            top5String = readNgram("top5StringDGA.csv")
        else:
            top2String, top3String, top4String, top5String = dngrams.topNgramsCalculationDGA(unmodifiedDGAList)


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
    Export_DGA_data(DGAList, outputPath)
    



#function for preparing individual domains 
def prepDataTest(string):
    #assign string
    domain = re.split('.[A-Za-z]*\.*$', string)[0]
    #domain = string

    #create temporary lists
    top2String = []
    top3String = []
    top4String = []
    top5String = []

    #calculate and create an Entropy column
    entropy = ent.Entropy(domain)
    

    #calculate and create an vowerl and consonants column
    count = vowcon.VowelConsonants(domain)
    vowel = count['vowel']
    consonant = count['consonant']
    #calculate and create an Length column
    
    length = len(domain)

    #convert verdict to a number column
    if consonant == 0 or vowel == 0:
        vowelConsRatio = 0
    else:
        vowelConsRatio = round(vowel / consonant, 5)
    entropyLenRatio = round(entropy / length, 5)

    #calculate numcount and create a column
    countNum = numCount.numCount(domain)

    #calculate and create a ngram score column FOR LEGIT NGRAMS
    #if topNgrams exist in folder, read those otherwise generate it
    #generate top NGrams for legit
    #if file exists in the specific folder
    #top 2 ngrams
    #if folder exists
    if os.path.isdir("CSVs"):
        #if the csv file exists
        if os.path.isfile("CSVs\\topNgrams\\top2String.csv"):
            #read the csv file
            top2String = readNgram("CSVs\\topNgrams\\top2String.csv")
            top3String = readNgram("CSVs\\topNgrams\\top3String.csv")
            top4String = readNgram("CSVs\\topNgrams\\top4String.csv")
            top5String = readNgram("CSVs\\topNgrams\\top5String.csv")
            
    #if the folder doesnt exist
    else:
        #check to see if the csv file is in the same folder as python file
        if os.path.isfile("top2String.csv"):
            #read the file if it is
            top2String = readNgram("Ctop2String.csv")
            top3String = readNgram("top3String.csv")
            top4String = readNgram("top4String.csv")
            top5String = readNgram("top5String.csv")

    #testDomain = re.split('.[A-Za-z]*\.*$', domain)[0]
    testDomain = domain

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
    
    #if topNgrams exist in folder, read those otherwise generate it
    #generate top NGrams for legit
    #if file exists in the specific folder
    #top 2 ngrams
    #if folder exists
    if os.path.isdir("CSVs"):
        #if the csv file exists
        if os.path.isfile("CSVs\\topNgrams\\top2StringDGA.csv"):
            #read the csv file
            top2String = readNgram("CSVs\\topNgrams\\top2StringDGA.csv")
            top3String = readNgram("CSVs\\topNgrams\\top3StringDGA.csv")
            top4String = readNgram("CSVs\\topNgrams\\top4StringDGA.csv")
            top5String = readNgram("CSVs\\topNgrams\\top5StringDGA.csv")
            
    #if the folder doesnt exist
    else:
        #check to see if the csv file is in the same folder as python file
        if os.path.isfile("top2String.csv"):
            #read the file if it is
            top2String = readNgram("top2StringDGA.csv")
            top3String = readNgram("top3StringDGA.csv")
            top4String = readNgram("top4StringDGA.csv")
            top5String = readNgram("top5StringDGA.csv")

    #testDomain = re.split('.[A-Za-z]*\.*$', domain)[0]
    testDomain = domain

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
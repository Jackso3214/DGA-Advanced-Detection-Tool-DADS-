import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np
import re

def readNgram(path):
    temp = []
    with open(path) as file:
                    reader = csv.reader(file)
                    for row in reader:
                        temp.append(row)
                    file.close()
    return temp

def prepDataTest(string):
    #assign string
    domain = string

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
   
        #check to see if the csv file is in the same folder as python file
            #read the file if it is
    top2String = readNgram("CSVs\\topNgrams\\top2String.csv")
    top3String = readNgram("CSVs\\topNgrams\\top3String.csv")
    top4String = readNgram("CSVs\\topNgrams\\top4String.csv")
    top5String = readNgram("CSVs\\topNgrams\\top5String.csv")

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
    
    #if topNgrams exist in folder, read those otherwise generate it
    #generate top NGrams for legit
    #if file exists in the specific folder
    #top 2 ngrams
    #if folder exists
  
        #check to see if the csv file is in the same folder as python file
            #read the file if it is
    top2String = readNgram("CSVs\\topNgrams\\top2StringDGA.csv")
    top3String = readNgram("CSVs\\topNgrams\\top3StringDGA.csv")
    top4String = readNgram("CSVs\\topNgrams\\top4StringDGA.csv")
    top5String = readNgram("CSVs\\topNgrams\\top5StringDGA.csv")

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
    data = [[string,"unknown",entropy, vowel, consonant, length, vowelConsRatio, entropyLenRatio, countNum ,score[0], score[1], score[2], score[3], totalScore, scoreDGA[0], scoreDGA[1], scoreDGA[2], scoreDGA[3], totalScoreDGA]]
    df = pd.DataFrame(data, columns=["Domain","Verdict","Entropy", "Vowel", "Consonant", "Length", "VowelConsRatio", "EntropyLength", "CountNum" ,"ngram2ScoreLegit", "ngram3ScoreLegit", "ngram4ScoreLegit", "ngram5ScoreLegit", "ngramTotalScoreLegit", "ngram2ScoreDGA", "ngram3ScoreDGA", "ngram4ScoreDGA", "ngram5ScoreDGA", "ngramTotalScoreDGA"])
    return df
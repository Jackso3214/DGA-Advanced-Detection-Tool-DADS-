import pandas as pd
import nltk
from nltk import ngrams
from collections import Counter
import re


def load_DGA_data():
    return pd.read_csv("CSVs\\trainingDataset\\dga_dataset_10k_set1.csv")
    #return pd.read_csv("CSVs\\Testing_Set_Take_1.csv")
    

#top ngrams in all legit domains
def topNgramsCalculation():

    #set variables
    
    temp = []
    top2 = []
    top3 = []
    top4 = []
    top5 = []

    top2String = []
    top3String = []
    top4String = []
    top5String = []
    

    #import dataset
    dataset = load_DGA_data()

    #regex to remove the TLD 
    for ind in dataset.index:
        domain = dataset['Domain'][ind]
        temp.append(re.split('.[A-Za-z]*\.*$', domain)[0])
    
    #assign new column with excluded TLD
    dataset['TLDExcluded'] = temp
    #print(dataset)

    #create a dataset with only legit domains
    legitDomains = dataset[dataset['Verdict'] == 'legit']


    #top 2 Grams

    #go through each domain and find out the most common trigram in legit domains
    ngram_counts = Counter()
    for ind in range(0, len(legitDomains.index), 1):
        tokens = list(legitDomains['TLDExcluded'][ind])
    #    #print(tokens)
        domain_ngrams = list(ngrams(tokens, 2))
        ngram_counts.update(domain_ngrams)

    ##get the top 50 N grams of occurences in legit domains
    for top in range(0,50):
        top2.append(ngram_counts.most_common()[top])

    #top 3 Grams

    #go through each domain and find out the most common trigram in legit domains
    ngram_counts = Counter()
    for ind in range(0, len(legitDomains.index), 1):
        tokens = list(legitDomains['TLDExcluded'][ind])
        domain_ngrams = list(ngrams(tokens, 3))
        ngram_counts.update(domain_ngrams)

    #get the top 50 N grams of occurences in legit domains
    for top in range(0,50):
        top3.append(ngram_counts.most_common()[top])

    #top 4 Grams

    #go through each domain and find out the most common trigram in legit domains
    ngram_counts = Counter()
    for ind in range(0, len(legitDomains.index), 1):
        tokens = list(legitDomains['TLDExcluded'][ind])
        domain_ngrams = list(ngrams(tokens, 4))
        ngram_counts.update(domain_ngrams)

    #get the top 10 N grams of occurences in legit domains
    for top in range(0,50):
        top4.append(ngram_counts.most_common()[top])

    #top 5 Grams

    #go through each domain and find out the most common trigram in legit domains
    ngram_counts = Counter()
    for ind in range(0, len(legitDomains.index), 1):
        tokens = list(legitDomains['TLDExcluded'][ind])
        domain_ngrams = list(ngrams(tokens, 5))
        ngram_counts.update(domain_ngrams)

    #get the top 10 N grams of occurences in legit domains
    for top in range(0,50):
        top5.append(ngram_counts.most_common()[top])


    #top2
    #put put each combination in one string and convert from tuple to list
    for item in top2:
        #temporary array for object
        temp = []
         #temporary string for combined characters
        string = ""
        for characters in item[0]:
            string += characters
        
        temp.append(string)
        temp.append(item[1])
        top2String.append(temp)

    #top3
    #put put each combination in one string and convert from tuple to list
    for item in top3:
        #temporary array for object
        temp = []
        #temporary string for combined characters
        string = ""
        for characters in item[0]:
            string += characters
        
        temp.append(string)
        temp.append(item[1])
        top3String.append(temp)

    #top4
    #put put each combination in one string and convert from tuple to list
    for item in top4:
        #temporary array for object
        temp = []
         #temporary string for combined characters
        string = ""
        for characters in item[0]:
            string += characters
        
        temp.append(string)
        temp.append(item[1])
        top4String.append(temp)
    
    #top5
    #put put each combination in one string and convert from tuple to list
    for item in top5:
        #temporary array for object
        temp = []
        #temporary string for combined characters
        string = ""
        for characters in item[0]:
            string += characters
        
        temp.append(string)
        temp.append(item[1])
        top5String.append(temp)
    
    #print the resulting top Ngrams
    #print(top2String, '\n\n', top3String, '\n\n', top4String, '\n\n', top5String, '\n\n')

    
    return top2String, top3String, top4String, top5String

#top ngrams in DGA domains
def topNgramsCalculationDGA():

    #set variables
    
    temp = []
    top2 = []
    top3 = []
    top4 = []
    top5 = []

    top2String = []
    top3String = []
    top4String = []
    top5String = []
    

    #import dataset
    dataset = load_DGA_data()

    #regex to remove the TLD 
    for ind in dataset.index:
        domain = dataset['Domain'][ind]
        temp.append(re.split('.[A-Za-z]*\.*$', domain)[0])
    
    #assign new column with excluded TLD
    dataset['TLDExcluded'] = temp
    #print(dataset)

    #create a dataset with only legit domains
    dgaDomains = dataset[dataset['Verdict'] == 'dga']
    dgaDomains = dgaDomains.reset_index(drop=True)

    #top 2 Grams

    #go through each domain and find out the most common trigram in legit domains
    ngram_counts = Counter()
    for ind in range(0, len(dgaDomains.index), 1):
        tokens = list(dgaDomains['TLDExcluded'][ind])
    #   #print(tokens)
        domain_ngrams = list(ngrams(tokens, 2))
        ngram_counts.update(domain_ngrams)

    ##get the top 50 N grams of occurences in legit domains
    for top in range(0,50):
        top2.append(ngram_counts.most_common()[top])

    #top 3 Grams

    #go through each domain and find out the most common trigram in legit domains
    ngram_counts = Counter()
    for ind in range(0, len(dgaDomains.index), 1):
        tokens = list(dgaDomains['TLDExcluded'][ind])
        domain_ngrams = list(ngrams(tokens, 3))
        ngram_counts.update(domain_ngrams)

    #get the top 50 N grams of occurences in legit domains
    for top in range(0,50):
        top3.append(ngram_counts.most_common()[top])

    #top 4 Grams

    #go through each domain and find out the most common trigram in legit domains
    ngram_counts = Counter()
    for ind in range(0, len(dgaDomains.index), 1):
        tokens = list(dgaDomains['TLDExcluded'][ind])
        domain_ngrams = list(ngrams(tokens, 4))
        ngram_counts.update(domain_ngrams)

    #get the top 10 N grams of occurences in legit domains
    for top in range(0,50):
        top4.append(ngram_counts.most_common()[top])

    #top 5 Grams

    #go through each domain and find out the most common trigram in legit domains
    ngram_counts = Counter()
    for ind in range(0, len(dgaDomains.index), 1):
        tokens = list(dgaDomains['TLDExcluded'][ind])
        domain_ngrams = list(ngrams(tokens, 5))
        ngram_counts.update(domain_ngrams)

    #get the top 50 N grams of occurences in legit domains
    for top in range(0,50):
        top5.append(ngram_counts.most_common()[top])


    #top2
    #put put each combination in one string and convert from tuple to list
    for item in top2:
        #temporary array for object
        temp = []
         #temporary string for combined characters
        string = ""
        for characters in item[0]:
            string += characters
        
        temp.append(string)
        temp.append(item[1])
        top2String.append(temp)

    #top3
    #put put each combination in one string and convert from tuple to list
    for item in top3:
        #temporary array for object
        temp = []
        #temporary string for combined characters
        string = ""
        for characters in item[0]:
            string += characters
        
        temp.append(string)
        temp.append(item[1])
        top3String.append(temp)

    #top4
    #put put each combination in one string and convert from tuple to list
    for item in top4:
        #temporary array for object
        temp = []
         #temporary string for combined characters
        string = ""
        for characters in item[0]:
            string += characters
        
        temp.append(string)
        temp.append(item[1])
        top4String.append(temp)
    
    #top5
    #put put each combination in one string and convert from tuple to list
    for item in top5:
        #temporary array for object
        temp = []
        #temporary string for combined characters
        string = ""
        for characters in item[0]:
            string += characters
        
        temp.append(string)
        temp.append(item[1])
        top5String.append(temp)
    
    #print the resulting top Ngrams DGA
    #print(top2String, '\n\n', top3String, '\n\n', top4String, '\n\n', top5String, '\n\n')

    
    return top2String, top3String, top4String, top5String

#for testing this module
#legit ngrams
#top2String, top3String, top4String, top5String = topNgramsCalculation()
#dga ngrams
#top2Stringdga, top3Stringdga, top4Stringdga, top5Stringdga = topNgramsCalculationDGA()
#legit domain tests
#testDomain = "microsoft.com"
#testDomain = "a21.z.akamai.net"
#testDomain = "onedscolprdcus06.centralus.cloudapp.azure.com"
#testDomain = "Google.ca"
#testDomain = "array615.prod.do.dsp.mp.microsoft.com"

#DGA domain tests
#testDomain = "ldcpnuvswiflqpcdp.com"
#testDomain = "ntnuytghqgaenhrg.eu"
#testDomain = "q8m6m0wnmfe8grox3vk4slu.ddns.net"
#testDomain = "relationaveragediscoverreplace.com"
#testDomain = "cracktowerproposereducebehave.com"


#print("testing")
#print(top3String[0][0])
#
#if top3String[0][0] in testDomain:
#    print("it works!")
#
#totalScore = 0
#score = [0,0,0,0]
#for ngram in top2String:
#    if ngram[0] in testDomain:
#        totalScore += 1
#        score[0] += 1#
#
#for ngram in top3String:
#    if ngram[0] in testDomain:
#        totalScore += 1
#        score[1] += 1
#
#for ngram in top4String:
#    if ngram[0] in testDomain:
#        totalScore += 1
#        score[2] += 1#

#for ngram in top5String:
#    if ngram[0] in testDomain:
#        totalScore += 1
#        score[3] += 1


#print("the domain ", testDomain, " has a TOTAL score of ", totalScore)
#print("the domain ", testDomain, " has a individual score of ", score)



import subprocess
import numpy as np
import pandas as pd

def whoIs(domainStr):
    #CommandToRun = "dir; cd Whois; dir; whois64.exe {url};".format(url = "www.google.com")
    #CommandToRun = ["Whois\whois64.exe", "www.google.com"]
    CommandToRun = ["Whois\whois64.exe", domainStr]

    #run command and write output to file
    with open('Whois\output.txt', 'w') as f:
        procObj = subprocess.run(CommandToRun, shell=True, stdout=f, text=True)
        f.close()
    print(procObj.returncode)

    whoisInfo = dict() #create empty dictionary

    with open('Whois\output.txt', 'r') as f:
        keyList = ["Registrar URL", "Updated Date", "Creation Date", "Registrar Abuse Contact Email", "Registrar Abuse Contact Phone"]
        for line in f.readlines():
            #list of words to find
            #check if any of the words exist in the keyList in the string
            if any(word in line for word in keyList):
                #if the word is in the line, get the info
                keyValuePair = line.split(": ")
                
                #replace any uneccesary characters
                keyValuePair[0] = keyValuePair[0].replace(":", "") #remove unecessary spacings
                keyValuePair[0] = keyValuePair[0].replace("   ", "") #remove unecessary spacings
                keyValuePair[0] = keyValuePair[0].replace('\n', '')  #remove newline character

                #if no value is found for a key
                if len(keyValuePair) == 2:
                    keyValuePair[1] = keyValuePair[1].replace('\n', '')  #remove newline character
                else:
                    keyValuePair.append(np.nan)

                #append key and value to dictionary
                whoisInfo[keyValuePair[0]]=keyValuePair[1]

        #if whoIsInfo is empty, then fill it up with nan's
        if not bool(whoisInfo):
            for key in keyList:
                whoisInfo[key] = np.nan
    
    return whoisInfo
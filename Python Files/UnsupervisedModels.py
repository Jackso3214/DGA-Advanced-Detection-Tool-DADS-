#custom imports
import EntropyCalculator as ent
import VowelConsonantsCalculator as vowcon
import FeatureEngineering as feat
import DomainNgrams as dngrams
import NumCountCalculator as numCount
import DataPrepUnsupervised as DPU

import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
import re


def VerdHomeData(VerdDataPath):
    columns_to_exclude = ['Domain', 'Verdict', 'VN','Unnamed: 0']
    home_data = pd.read_csv(VerdDataPath,usecols=lambda column: column not in columns_to_exclude)   
    verd_data = pd.read_csv(VerdDataPath,usecols=lambda column: column in columns_to_exclude)
    verd_data=verd_data.drop(columns='Unnamed: 0')
    verd_data=verd_data.drop(columns='VN')
    return home_data, verd_data

def kmean(indomain, VerdDataPath='CSVs\output\outputTable10k_set1.csv'):
    X=DPU.prepDataTest(indomain)
    X2 = X[['Domain', 'Verdict']]
    X3 = X.iloc[:, 2:]
    home_data, verd_data = VerdHomeData(VerdDataPath)
    
    hom_data = pd.concat([home_data, X3], ignore_index=True)
    ver_data = pd.concat([verd_data, X2], ignore_index=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(hom_data)
 
    X_normalized = normalize(X_scaled)
 
    X_normalized = pd.DataFrame(X_normalized)
 
    pca = PCA(n_components = 2)
    X_principal = pca.fit_transform(X_normalized)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']
    kmeans = KMeans(n_clusters = 4, random_state = 0, n_init='auto')
    kmeans.fit(X_principal)
    ver_data['labels']=kmeans.labels_
    value_counts = ver_data['labels'].value_counts()##
    grouped = ver_data.groupby(['labels', 'Verdict'])
    result = grouped.size()##
    value_test = ver_data.loc[ver_data['Verdict']== 'unknown']
    return result, value_counts, value_test


def spect(indomain, inaffinity,VerdDataPath='CSVs\output\outputTable10k_set1.csv',):
    X=DPU.prepDataTest(indomain)
    X2 = X[['Domain', 'Verdict']]
    X3 = X.iloc[:, 2:]
    home_data, verd_data = VerdHomeData(VerdDataPath)
    
    hom_data = pd.concat([home_data, X3], ignore_index=True)
    ver_data = pd.concat([verd_data, X2], ignore_index=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(hom_data) 

    X_normalized = normalize(X_scaled)

    X_normalized = pd.DataFrame(X_normalized)

    pca = PCA(n_components = 2)
    X_principal = pca.fit_transform(X_normalized)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']
    spectral_model = SpectralClustering(n_clusters = 4, affinity = inaffinity)
    labels = spectral_model.fit_predict(X_principal)
    
    ver_data['labels']=labels
    value_counts = ver_data['labels'].value_counts()##
    grouped = ver_data.groupby(['labels', 'Verdict'])
    result = grouped.size()##
    value_test = ver_data.loc[ver_data['Verdict']== 'unknown']
    return result, value_counts, value_test

#result, value_counts = kmean('iomvwijvaijacsaikj.com')
#print(result, "\n", value_counts)
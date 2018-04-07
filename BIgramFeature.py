import pickle
import collections
import numpy as np
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer



## const
PKL_PATH = "consPapers.pkl"
PKL_PATH1 = "deonPapers.pkl"
LEMMA_FILTER = ['NOUN', 'PROPN', 'VERB', 'ADJ', 'NUM'] # keep 'NUM' as the last, function lemma_dep_list() refers to this



def loadDataset(pklPath=PKL_PATH):
    with open(pklPath, "rb") as pklFile:
        return np.array(pickle.load(pklFile, encoding="utf-8"))

def loadDataset1(pklPath=PKL_PATH1):
    with open(pklPath, "rb") as pklFile:
        return np.array(pickle.load(pklFile, encoding="utf-8"))

c_data_set = loadDataset()
d_data_set = loadDataset1()

#splitting to pieces:
c_list = ""
for i in c_data_set:
	c_list = c_list + i

d_list=""
for i in d_data_set:
    d_list = d_list + i

c_data = []
div = len(c_list)//100
temp = ""
for j in range(len(c_list)):
	temp = temp + c_list[j]
	if j%div == 0:
		c_data.append(temp)
		temp = ""
c_data.append(temp)

d_data = []
div = len(d_list)//100
temp = ""
for j in range(len(d_list)):
	temp = temp + d_list[j]
	if j%div == 0:
		d_data.append(temp)
		temp = ""
d_data.append(temp)

print(len(c_data), len(d_data))

'''
d_data = []
div = len(d_data)//100
temp = []
for j in range(len(d_list)):
	temp.append(d_list[j])
	if j%div == 0:
		d_data.append(temp)
		temp = []
d_data.append(temp)

print(len(c_data), len(d_data))





#implementing n-gram

bigram_vectorizer = CountVectorizer(ngram_range=(1, 3),token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
#print(analyze(c_data_set[0]))
X_1 = bigram_vectorizer.fit_transform(c_data_set).toarray()
X_2 = bigram_vectorizer.fit_transform(d_data_set).toarray()
X_train = X_1.tolist() + X_2.tolist()

y = []
for i in X_1:
	y.append('cons')
for i in X_2:
	y.append('deon')

print(len(X_1),len(X_2),len(X_train))
print(y)
'''




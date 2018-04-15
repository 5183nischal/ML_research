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
from sklearn.naive_bayes import MultinomialNB
import sklearn.svm
import featureExtraction


## const
PKL_PATH = "consPapersNew.pkl"
PKL_PATH1 = "deonPapersNew.pkl"
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
    c_list = c_list + str(i)

d_list=""
for i in d_data_set:
    d_list = d_list + str(i)

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

final_data_set = c_data + d_data


#Using the tfidf feature extractor

train_set = []
n = 0
for item in c_data:
    case = {'text':item, 'label': "cons"}
    train_set.append(case)

for item in d_data:
    case = {'text':item, 'label': "deon"}
    train_set.append(case)

vocab = vocabBuild(train_set)

feature_set = addFeature(vocab, train_set)
#feature_set = addFeature(vocab, dev_set, feature_set)
addVector(feature_set, train_set)
#addVector(feature_set, dev_set)

vec_train = [item['vector'] for item in train_set]
class_train = [item['label'] for item in train_set]

#Train
X = vec_train
y = class_train

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 50)


clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
coef = clf.coef_[0].tolist()


n = 0
correct = 0
for i, j in zip(y_test, predicted):
    print('%r => %s' % (i, j))
    n = n + 1
    if i == j:
        correct = correct + 1
print(correct*100/n)

top = 100
predictors = []
for i in range(top):
    val = min(coef)
    index = coef.index(val)
    predictors.append([n_grams[index], val])
    coef.pop(index)

for i in predictors:
    print (i ,"\n")









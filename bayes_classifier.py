import pickle
import collections
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import sklearn.svm


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



#implementing n-gram

#bigram_vectorizer = CountVectorizer(ngram_range=(1, 3),token_pattern=r'\b\w+\b', min_df=1)
bigram_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
analyze = bigram_vectorizer.build_analyzer()
#print(analyze(c_data_set[0]))
X = bigram_vectorizer.fit_transform(final_data_set).toarray()
n_grams = bigram_vectorizer.get_feature_names()
y = []
for i in c_data:
    y.append('cons')
for i in d_data:
    y.append('deon')







#------------


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100)


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






print("SVM with Vector Featues")

clf = sklearn.svm.LinearSVC().fit(X_train, y_train)
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
    val = max(coef)
    index = coef.index(val)
    predictors.append([n_grams[index], val])
    coef.pop(index)

for i in predictors:
    print (i ,"\n")




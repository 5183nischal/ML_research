import numpy as np
import glob, os, pickle
import math

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB




def loadDataset(filePath):
    with open(filePath, "rb") as pklFile:
        return np.array(pickle.load(pklFile, encoding="utf-8"))


f1 = '/Users/liammeier/Dropbox/MoralReasoning/consPapers.pkl'
f2 = '/Users/liammeier/Dropbox/MoralReasoning/deonPapers.pkl'

consArray = loadDataset(f1)
deonArray = loadDataset(f2)

X = []
y = []
for i in consArray:
    X.append(i)
    y.append('cons')
for i in deonArray:
    X.append(i)
    y.append('deon')

'''
d = dict()
for i in consArray:
    d
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 50)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

predicted = clf.predict(X_test_tfidf)

for i, j in zip(y_test, predicted):
    print('%r => %s' % (i, j))


'''
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)


for i, j in zip(X_test_tfidf, y_test):
    print(clf.predict(i))
    print("Actual: ", j)
'''

'''
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),])
'''
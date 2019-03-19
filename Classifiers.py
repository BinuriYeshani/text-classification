#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 02:26:54 2019

@author: binuri
"""
import pandas as pd
import numpy as np
import codecs,re,nltk
from sklearn import svm
import fetchData
from sklearn import model_selection, preprocessing, naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc


dirName = '20_newsgroupSample'
listOfFiles = fetchData.main(dirName)
corpus = []

labels, texts = [], []
for f in listOfFiles:
    texts = codecs.open(f, "r", encoding ="ISO-8859-1").read()
    corpus.append(texts)
    label= f.replace('20_newsgroupSample/','')
    amp_index = label.index('/')
    label = label[0:amp_index]
    labels.append(label)
#print(corpus)


wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

normalize_corpus = np.vectorize(normalize_document)
norm_corpus = normalize_corpus(corpus)



corpus = np.array(norm_corpus)
corpus_df = pd.DataFrame({'Document': corpus,'Category': labels})
corpus_df = corpus_df[['Document', 'Category']]
print(corpus_df)


trainDF = pd.DataFrame()

trainDF['text'] = corpus_df.iloc[:,0]
trainDF['label'] = corpus_df.iloc[:,1]

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])
###label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',  max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)



##########Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(xtrain_tfidf_ngram, train_y)

from  sklearn.metrics  import accuracy_score
predicted = clf.predict(xvalid_tfidf_ngram)
print(predicted)
print(accuracy_score(valid_y,predicted))


###########SVM
#from sklearn.pipeline import Pipeline
#from sklearn.feature_extraction.text import CountVectorizer 
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.svm import SVC
#from sklearn.pipeline import Pipeline 
#
#text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(kernel='rbf'))])
#
##train model
#text_clf.fit(xtrain_tfidf_ngram, train_y)
#
##predict class form test data 
#predicted = text_clf.predict(xvalid_tfidf_ngram)
#print(predicted)



############SVM
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(xtrain_tfidf_ngram, train_y, test_size = 0.2, random_state=0)
clf = svm.SVC(kernel='rbf', C = 10.0, gamma=0.1)
clf.fit(xvalid_tfidf_ngram, valid_y)



######kmeans
from sklearn.cluster import KMeans
km = KMeans(n_clusters=20, random_state=0)
km.fit(xtrain_tfidf_ngram,train_y)
print(km.predict(xvalid_tfidf_ngram))
print(accuracy_score(valid_y,predicted))

# cross validation
#from sklearn.cross_validation import cross_val_score, cross_val_predict
#from sklearn import metrics
#from sklearn.cross_validation import train_test_split
#
## Perform 6-fold cross validation
#df = pd.DataFrame(xtrain_tfidf_ngram, train_y)
#print(pd)
#scores = cross_val_score(xtrain_tfidf_ngram, df, train_y, cv=10)
#print (scores)
#


from sklearn.model_selection import KFold
X = xtrain_tfidf_ngram
y = train_y
kf = KFold(n_splits=10)
kf.get_n_splits(X)

print(kf)  

for train_index, test_index in kf.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]


######precision,recall,F score and accuracy
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
actual = valid_y
predicted = predicted
results = confusion_matrix(actual, predicted) 
print ('Confusion Matrix :')
print(results) 
print (accuracy_score(actual, predicted) )
#
print (classification_report(actual, predicted))

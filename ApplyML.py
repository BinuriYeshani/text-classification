#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 18:14:34 2019

@author: binuri
"""

import pandas as pd
import numpy as np
import re,codecs
import nltk
import sklearn
from sklearn.cluster import KMeans
#create corpus fro whole news group and assign new topics tags
import fetchData

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


dirName = '20_newsgroup';
listOfFiles = fetchData.main(dirName)
corpus = list()

labels, texts = [], []
for f in listOfFiles:
    file = codecs.open(f, "r", encoding="ISO-8859-1").read()
    file = normalize_corpus(file)
    corpus.append(file)
    label= f.replace('20_newsgroup/','')
    amp_index = label.index('/')
    label = label[0:amp_index]
    labels.append(label)


corpus = np.array(corpus)
corpus_df = pd.DataFrame({'Document': corpus, 
                          'Category': labels})
corpus_df = corpus_df[['Document', 'Category']]
#print(corpus_df)

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')



normalize_corpus = np.vectorize(normalize_document)


norm_corpus = normalize_corpus(corpus)
print(norm_corpus)

#########create feature vector 
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(norm_corpus)
cv_matrix = cv_matrix.toarray()
print(cv_matrix.shape)
print(cv_matrix)

# get all unique words in the corpus
vocab = cv.get_feature_names()
# show document feature vectors
print(pd.DataFrame(cv_matrix, columns=vocab))



#####apply machine learning
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=20, max_iter=10000, random_state=0)
dt_matrix = lda.fit_transform(cv_matrix)
features = pd.DataFrame(dt_matrix)
print(features)

print("going to use algorithm")
km = KMeans(n_clusters=20, random_state=0)
km.fit_transform(features)
cluster_labels = km.labels_
cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])
pd.concat([corpus_df, cluster_labels], axis=1)

print(pd)


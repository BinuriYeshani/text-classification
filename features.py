import pandas as pd
import numpy as np
import re
import codecs
import nltk
from nltk import word_tokenize,sent_tokenize


import fetchData

dirName = '20_newsgroupSample'
listOfFiles = fetchData.main(dirName)
corpus = []

labels, texts = [], []
for f in listOfFiles:
    file = codecs.open(f, "r", encoding="ISO-8859-1").read()
    corpus.append(file)
    label= f.replace('20_newsgroupSample/','')
    amp_index = label.index('/')
    label = label[0:amp_index]
    labels.append(label)
print(corpus)

#corpus = np.array(corpus)
corpus_df = pd.DataFrame({'Document': corpus,'Category': labels})
corpus_df = corpus_df[['Document', 'Category']]
print(corpus_df)



wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

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

normalize_corpus = np.vectorize(normalize_document)
norm_corpus = normalize_corpus(corpus)
#print(norm_corpus)







###bag of Ngram 
bv = CountVectorizer(ngram_range=(2,2))
bv_matrix = bv.fit_transform(norm_corpus)
bv_matrix = bv_matrix.toarray()
vocab = bv.get_feature_names()
print("N-gram calculation")
print(pd.DataFrame(bv_matrix, columns=vocab))



#####tfidf model
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
tv_matrix = tv.fit_transform(norm_corpus)
tv_matrix = tv_matrix.toarray()
vocab = tv.get_feature_names()
print("tfidf calculation")
print(pd.DataFrame(np.round(tv_matrix, 20), columns=vocab))




#####cosin similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(tv_matrix)
similarity_df = pd.DataFrame(similarity_matrix)
print("cosin similarity calculation")
print(similarity_df)




####Clustering documents using similarity features
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(similarity_matrix, 'ward')
print(pd.DataFrame(Z, columns=['Document\Cluster 1','Document\Cluster 2',
                               'Distance','Cluster Size'],dtype='object'))




#####topic modeling feature
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_topics=20, max_iter=1000, random_state=0)
dt_matrix = lda.fit_transform(cv_matrix)
features = pd.DataFrame(dt_matrix,)

print(features)



import codecs
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

def tokenize(text):
    tokens = word_tokenize(text)
    stems = []
    for item in tokens: stems.append(PorterStemmer().stem(item))
    return stems

# your corpus
import fetchData
dirName = '/home/binuri/work_Msc/IR_Assignment3/20_newsgroup';
listOfFiles = fetchData.main(dirName)
fileList = list()
for f in listOfFiles:
    fileList.append(codecs.open(f, "r", encoding="ISO-8859-1").read())

#text = ["This is your first text book", "This is the third text for analysis", "This is another text"]
# word tokenize and stem
text = fileList
text = [" ".join(tokenize(txt.lower())) for txt in text]
vectorizer = TfidfVectorizer()

matrix = vectorizer.fit_transform(text).todense()
#print (matrix)
# transform the matrix to a pandas df
matrix = pd.DataFrame(matrix, columns=vectorizer.get_feature_names())
# sum over each document (axis=0)
print(matrix)
top_words = matrix.sum(axis=0).sort_values(ascending=False)

#print(top_words)




import re
from math import log

import numpy as np
import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open('article1.txt', encoding="utf8") as f:
    article1 = f.read()
with open('article2.txt', encoding="utf8") as f:
    article2 = f.read()
with open('article3.txt', encoding="utf8") as f:
    article3 = f.read()
tokenized_article1 = word_tokenize(article1)
tokenized_article2 = word_tokenize(article2)
tokenized_article3 = word_tokenize(article3)
n = [len(tokenized_article1), len(tokenized_article2), len(tokenized_article3)]


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        regex_num_punctuation = r'(\d+)|([^\w\s])'
        regex_little_words = r'(\b\w{1,2}\b)'
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)
                if not re.search(regex_num_punctuation, t) and not re.search(regex_little_words, t)]


count_vect = CountVectorizer(stop_words='english', tokenizer=LemmaTokenizer())
count_matrix = count_vect.fit_transform([article1, article2, article3])
count_array = count_matrix.toarray()
df = pd.DataFrame(data=count_array, columns=count_vect.get_feature_names_out())
DTM = df.copy()
DTM.transpose().to_csv("DTM.csv")
print("\nDTM:\n", DTM)

for i in range(3):
    df.iloc[i] = df.iloc[i].apply(lambda x: round(x/n[i], 6))
TF = df.copy()
TF.transpose().to_csv("TF.csv")
print("\nTF:\n", TF)

count_vect = CountVectorizer(stop_words='english', tokenizer=LemmaTokenizer(), binary=True)
count_matrix = count_vect.fit_transform([article1, article2, article3])
count_array = count_matrix.toarray()
df = pd.DataFrame(data=count_array, columns=count_vect.get_feature_names_out())
words = df.columns.values
IDF_values = []
for i in words:
    IDF_values.append(round(log(3/df[i].sum()), 6))
IDF = pd.DataFrame(data=[IDF_values], columns=words)
IDF.transpose().to_csv("IDF.csv")
print("\nIDF:\n", IDF)

TFIDF = TF.copy()
for i in words:
    TFIDF[i] = TFIDF[i].apply(lambda x: round(x*IDF[i][0], 6))
TFIDF.transpose().to_csv("TFIDF.csv")
print("\nTFIDF:\n", TFIDF)

print("\ncosine similarity matrix:\n", cosine_similarity(TF.values))



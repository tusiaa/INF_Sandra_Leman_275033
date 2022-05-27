import re
from math import log
import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
stop_words = stopwords.words("english")

filtered_article1 = []
filtered_article2 = []
filtered_article3 = []
for w in tokenized_article1:
    if w not in stop_words:
        filtered_article1.append(w)
for w in tokenized_article2:
    if w not in stop_words:
        filtered_article2.append(w)
for w in tokenized_article3:
    if w not in stop_words:
        filtered_article3.append(w)

n = [len(filtered_article1), len(filtered_article2), len(filtered_article3)]
print(n)


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        regex_num_punctuation = r'(\d+)|([^\w\s])'
        regex_little_words = r'(\b\w{1,2}\b)'
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)
                if not re.search(regex_num_punctuation, t) and not re.search(regex_little_words, t)]


count_vect = CountVectorizer(stop_words=stop_words, tokenizer=LemmaTokenizer())
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

count_vect = CountVectorizer(stop_words=stop_words, tokenizer=LemmaTokenizer(), binary=True)
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


tf_vect = TfidfVectorizer(use_idf=False, stop_words=stop_words, tokenizer=LemmaTokenizer())
tf_matrix = tf_vect.fit_transform([article1, article2, article3])
tf_array = tf_matrix.toarray()
df = pd.DataFrame(data=tf_array, columns=tf_vect.get_feature_names_out())
print("\nTF with sklearn\n", df)
TF2 = df.copy()

tf_vect = TfidfVectorizer(stop_words=stop_words, tokenizer=LemmaTokenizer())
tf_matrix = tf_vect.fit_transform([article1, article2, article3])
tf_array = tf_matrix.toarray()
df = pd.DataFrame(data=tf_array, columns=tf_vect.get_feature_names_out())
print("\nIDF with sklearn\n", tf_vect.idf_)
print("\nTFIDF with sklearn\n", df)

print("\ncosine similarity matrix for sklearn:\n", cosine_similarity(TF2.values))

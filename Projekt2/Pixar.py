import re

import pandas as pd
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize, WordNetLemmatizer, FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

tweets = pd.read_csv("tweets/pixar.csv")
print(tweets)

pixar_content = pd.DataFrame(tweets, columns=['content']).to_numpy().flatten()
print(pixar_content)
pixar_content = " ".join(pixar_content).lower()
pixar_content = re.sub(r'[^a-zA-Z0-9 ]', '', pixar_content)
pixar_tokenized = word_tokenize(pixar_content)
stop_words = stopwords.words("english")
stop_words.extend(['im'])

pixar_filtered = []
for w in pixar_tokenized:
    if w not in stop_words:
        pixar_filtered.append(w)

ps = PorterStemmer()
pixar_stemmed = []
for w in pixar_filtered:
    pixar_stemmed.append(ps.stem(w))

lem = WordNetLemmatizer()
pixar_lem = []
for w in pixar_filtered:
    pixar_lem.append(lem.lemmatize(w))

print("\nTokenized:")
print("Liczba slow:", len(pixar_tokenized))

fdist = FreqDist(pixar_tokenized)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nFiltered:")
print("Liczba slow:", len(pixar_filtered))

fdist = FreqDist(pixar_filtered)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nStemmed:")
print("Liczba slow:", len(pixar_stemmed))

fdist = FreqDist(pixar_stemmed)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nLemmatized:")
print("Liczba slow:", len(pixar_lem))

fdist = FreqDist(pixar_lem)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

fdist = FreqDist(pixar_lem).most_common(10)
words = []
frequency = []
for i in fdist:
    words.append(i[0])
    frequency.append(i[1])
words.reverse()
frequency.reverse()

plt.barh(words, frequency)
plt.savefig("charts/PixarBarPlot")
plt.show()

wordcloud = WordCloud(stopwords=stop_words, max_words=100, background_color="white").generate(pixar_content)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("charts/PixarWordCloud.png")

sid = SentimentIntensityAnalyzer()
ss = sid.polarity_scores(pixar_content)
for k in sorted(ss):
    print('{0}: {1}, \n'.format(k, ss[k]), end='')

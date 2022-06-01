import re
import pandas as pd
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize, WordNetLemmatizer, FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

tweets = pd.read_csv("tweets/dreamworks.csv")
print(tweets)

dreamworks_content = pd.DataFrame(tweets, columns=['content']).to_numpy().flatten()
print(dreamworks_content)
dreamworks_content = " ".join(dreamworks_content).lower()
dreamworks_content = re.sub(r'[^a-zA-Z0-9 ]', '', dreamworks_content)
dreamworks_tokenized = word_tokenize(dreamworks_content)
stop_words = stopwords.words("english")
# stop_words.extend(['dreamworks'])

dreamworks_filtered = []
for w in dreamworks_tokenized:
    if w not in stop_words:
        dreamworks_filtered.append(w)

ps = PorterStemmer()
dreamworks_stemmed = []
for w in dreamworks_filtered:
    dreamworks_stemmed.append(ps.stem(w))

lem = WordNetLemmatizer()
dreamworks_lem = []
for w in dreamworks_filtered:
    dreamworks_lem.append(lem.lemmatize(w))

print("\nTokenized:")
print("Liczba slow:", len(dreamworks_tokenized))

fdist = FreqDist(dreamworks_tokenized)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nFiltered:")
print("Liczba slow:", len(dreamworks_filtered))

fdist = FreqDist(dreamworks_filtered)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nStemmed:")
print("Liczba slow:", len(dreamworks_stemmed))

fdist = FreqDist(dreamworks_stemmed)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nLemmatized:")
print("Liczba slow:", len(dreamworks_lem))

fdist = FreqDist(dreamworks_lem)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

fdist = FreqDist(dreamworks_lem).most_common(10)
words = []
frequency = []
for i in fdist:
    words.append(i[0])
    frequency.append(i[1])
words.reverse()
frequency.reverse()

plt.barh(words, frequency)
plt.savefig("charts/DreamworksBarPlot")
plt.show()

wordcloud = WordCloud(stopwords=stop_words, max_words=100, background_color="white").generate(dreamworks_content)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("charts/DreamworksWordCloud.png")

sid = SentimentIntensityAnalyzer()
ss = sid.polarity_scores(dreamworks_content)
for k in sorted(ss):
    print('{0}: {1}, \n'.format(k, ss[k]), end='')

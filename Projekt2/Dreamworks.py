import pandas as pd
import itertools
from wordcloud import WordCloud
import snscrape.modules.twitter as sntwitter
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize, WordNetLemmatizer, FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

search = '"#dreamworks"'

tweets = itertools.islice(sntwitter.TwitterSearchScraper(search).get_items(), 10000)

df = pd.DataFrame(tweets)[['date', 'content']]
df.to_csv("tweets/dreamworks.csv")
print(df)

dreamworks_content = pd.DataFrame(tweets)[['content']]
dreamworks_tokenized = word_tokenize(dreamworks_content)
stop_words = stopwords.words("english")

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

print("\nTokenized:", dreamworks_tokenized)
print("Liczba slow:", len(dreamworks_tokenized))

fdist = FreqDist(dreamworks_tokenized)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nFiltered:", dreamworks_filtered)
print("Liczba slow:", len(dreamworks_filtered))

fdist = FreqDist(dreamworks_filtered)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nStemmed:", dreamworks_stemmed)
print("Liczba slow:", len(dreamworks_stemmed))

fdist = FreqDist(dreamworks_stemmed)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nLemmatized:", dreamworks_lem)
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

wordcloud = WordCloud(stopwords=stop_words).generate(dreamworks_content)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("charts/DreamworksWordCloud.png")

sid = SentimentIntensityAnalyzer()
ss = sid.polarity_scores(dreamworks_content)
for k in sorted(ss):
    print('{0}: {1}, \n'.format(k, ss[k]), end='')

import pandas as pd
import itertools
from wordcloud import WordCloud
import snscrape.modules.twitter as sntwitter
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize, WordNetLemmatizer, FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

search = '"#pixar"'

tweets = itertools.islice(sntwitter.TwitterSearchScraper(search).get_items(), 10000)

df = pd.DataFrame(tweets)[['date', 'content']]
df.to_csv("tweets/pixar.csv")
print(df)

pixar_content = pd.DataFrame(tweets)[['content']]
pixar_tokenized = word_tokenize(pixar_content)
stop_words = stopwords.words("english")

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

print("\nTokenized:", pixar_tokenized)
print("Liczba slow:", len(pixar_tokenized))

fdist = FreqDist(pixar_tokenized)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nFiltered:", pixar_filtered)
print("Liczba slow:", len(pixar_filtered))

fdist = FreqDist(pixar_filtered)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nStemmed:", pixar_stemmed)
print("Liczba slow:", len(pixar_stemmed))

fdist = FreqDist(pixar_stemmed)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nLemmatized:", pixar_lem)
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

wordcloud = WordCloud(stopwords=stop_words).generate(pixar_content)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("charts/PixarWordCloud.png")

sid = SentimentIntensityAnalyzer()
ss = sid.polarity_scores(pixar_content)
for k in sorted(ss):
    print('{0}: {1}, \n'.format(k, ss[k]), end='')

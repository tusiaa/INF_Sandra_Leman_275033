import pandas as pd
import itertools
from wordcloud import WordCloud
import snscrape.modules.twitter as sntwitter
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize, WordNetLemmatizer, FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

search = '"#disney"'

tweets = itertools.islice(sntwitter.TwitterSearchScraper(search).get_items(), 10000)

df = pd.DataFrame(tweets)[['date', 'content']]
df.to_csv("tweets/disney.csv")
print(df)

disney_content = pd.DataFrame(tweets)[['content']]
disney_tokenized = word_tokenize(disney_content)
stop_words = stopwords.words("english")

disney_filtered = []
for w in disney_tokenized:
    if w not in stop_words:
        disney_filtered.append(w)

ps = PorterStemmer()
disney_stemmed = []
for w in disney_filtered:
    disney_stemmed.append(ps.stem(w))

lem = WordNetLemmatizer()
disney_lem = []
for w in disney_filtered:
    disney_lem.append(lem.lemmatize(w))


print("\nTokenized:", disney_tokenized)
print("Liczba slow:", len(disney_tokenized))

fdist = FreqDist(disney_tokenized)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nFiltered:", disney_filtered)
print("Liczba slow:", len(disney_filtered))

fdist = FreqDist(disney_filtered)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nStemmed:", disney_stemmed)
print("Liczba slow:", len(disney_stemmed))

fdist = FreqDist(disney_stemmed)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nLemmatized:", disney_lem)
print("Liczba slow:", len(disney_lem))

fdist = FreqDist(disney_lem)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

fdist = FreqDist(disney_lem).most_common(10)
words = []
frequency = []
for i in fdist:
    words.append(i[0])
    frequency.append(i[1])
words.reverse()
frequency.reverse()

plt.barh(words, frequency)
plt.savefig("charts/DisneyBarPlot")
plt.show()

wordcloud = WordCloud(stopwords=stop_words).generate(disney_content)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("charts/DisneyWordCloud.png")

sid = SentimentIntensityAnalyzer()
ss = sid.polarity_scores(disney_content)
for k in sorted(ss):
    print('{0}: {1}, \n'.format(k, ss[k]), end='')

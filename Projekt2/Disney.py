import re
import pandas as pd
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize, WordNetLemmatizer, FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

tweets = pd.read_csv("tweets/disney.csv")
print(tweets)

disney_content = pd.DataFrame(tweets, columns=['content']).to_numpy().flatten()
print(disney_content)
disney_content = " ".join(disney_content).lower()
disney_content = re.sub(r'[^a-zA-Z0-9 ]', '', disney_content)
disney_tokenized = word_tokenize(disney_content)
stop_words = stopwords.words("english")
stop_words.extend(['im'])

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


print("\nTokenized:")
print("Liczba slow:", len(disney_tokenized))

fdist = FreqDist(disney_tokenized)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nFiltered:")
print("Liczba slow:", len(disney_filtered))

fdist = FreqDist(disney_filtered)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nStemmed:")
print("Liczba slow:", len(disney_stemmed))

fdist = FreqDist(disney_stemmed)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nLemmatized:")
print("Liczba slow:", len(disney_lem))

fdist = FreqDist(disney_lem)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist), "\n")

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

wordcloud = WordCloud(stopwords=stop_words, max_words=100, background_color="white").generate(disney_content)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("charts/DisneyWordCloud.png")

sid = SentimentIntensityAnalyzer()
ss = sid.polarity_scores(disney_content)
for k in sorted(ss):
    print('{0}: {1}, \n'.format(k, ss[k]), end='')

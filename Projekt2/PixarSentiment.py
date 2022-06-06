import re
import pandas as pd
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk import word_tokenize, WordNetLemmatizer, FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

tweets = pd.read_csv("tweets/pixar.csv")

pixar_content_array = pd.DataFrame(tweets, columns=['content']).to_numpy().flatten()

sid = SentimentIntensityAnalyzer()
lem = WordNetLemmatizer()
stop_words = stopwords.words("english")
stop_words.extend(['pixar', 'im', 'show', 'people', 'dont', 'didnt', 'movie', 'disney', 'pixars', 'one', 'disneypixar',
                   'know', 'cant', 'get', 'got', 'make', 'made', 'think', 'going', 'also', 'would', 'film', 'guy', '2',
                   'thing', 'go', 'see', 'say', 'said', 'thats', 'movies', 'us', 'animation', 'look', 'amp'])

neg = []
pos = []
neu = []
for i in pixar_content_array:
    if sid.polarity_scores(i)["compound"] > 0:
        pos.append(i)
    elif sid.polarity_scores(i)["compound"] == 0:
        neu.append(i)
    else:
        neg.append(i)

print("neg: ", len(neg))
print("neu: ", len(neu))
print("pos: ", len(pos))

neg = " ".join(neg).lower()
neg = re.sub(r'[^a-zA-Z0-9 ]', '', neg)
pixar_tokenized = word_tokenize(neg)

pixar_filtered = []
for w in pixar_tokenized:
    if w not in stop_words:
        pixar_filtered.append(w)

pixar_lem = []
for w in pixar_filtered:
    pixar_lem.append(lem.lemmatize(w))

fdist = FreqDist(pixar_lem).most_common(10)
words = []
frequency = []
for i in fdist:
    words.append(i[0])
    frequency.append(i[1])
words.reverse()
frequency.reverse()

plt.barh(words, frequency)
plt.savefig("charts/pixarNegBarPlot")
plt.show()

wordcloud = WordCloud(stopwords=stop_words, max_words=30, background_color="white").generate(neg)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("charts/PixarNegWordCloud.png")

pos = " ".join(pos).lower()
pos = re.sub(r'[^a-zA-Z0-9 ]', '', pos)
pixar_tokenized = word_tokenize(pos)

pixar_filtered = []
for w in pixar_tokenized:
    if w not in stop_words:
        pixar_filtered.append(w)

pixar_lem = []
for w in pixar_filtered:
    pixar_lem.append(lem.lemmatize(w))

fdist = FreqDist(pixar_lem).most_common(10)
words = []
frequency = []
for i in fdist:
    words.append(i[0])
    frequency.append(i[1])
words.reverse()
frequency.reverse()

plt.barh(words, frequency)
plt.savefig("charts/PixarPosBarPlot")
plt.show()

wordcloud = WordCloud(stopwords=stop_words, max_words=30, background_color="white").generate(pos)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("charts/PixarPosWordCloud.png")


neu = " ".join(neu).lower()
neu = re.sub(r'[^a-zA-Z0-9 ]', '', neu)
pixar_tokenized = word_tokenize(neu)

pixar_filtered = []
for w in pixar_tokenized:
    if w not in stop_words:
        pixar_filtered.append(w)

pixar_lem = []
for w in pixar_filtered:
    pixar_lem.append(lem.lemmatize(w))

fdist = FreqDist(pixar_lem).most_common(10)
words = []
frequency = []
for i in fdist:
    words.append(i[0])
    frequency.append(i[1])
words.reverse()
frequency.reverse()

plt.barh(words, frequency)
plt.savefig("charts/PixarNeuBarPlot")
plt.show()

wordcloud = WordCloud(stopwords=stop_words, max_words=30, background_color="white").generate(neu)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("charts/PixarNeuWordCloud.png")

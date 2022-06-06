import re
import pandas as pd
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk import word_tokenize, WordNetLemmatizer, FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

tweets = pd.read_csv("tweets/dreamworks.csv")

dreamworks_content_array = pd.DataFrame(tweets, columns=['content']).to_numpy().flatten()

sid = SentimentIntensityAnalyzer()
lem = WordNetLemmatizer()
stop_words = stopwords.words("english")
stop_words.extend(['dreamworks', 'im'])

neg = []
pos = []
neu = []
for i in dreamworks_content_array:
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
dreamworks_tokenized = word_tokenize(neg)

dreamworks_filtered = []
for w in dreamworks_tokenized:
    if w not in stop_words:
        dreamworks_filtered.append(w)

dreamworks_lem = []
for w in dreamworks_filtered:
    dreamworks_lem.append(lem.lemmatize(w))

fdist = FreqDist(dreamworks_lem).most_common(10)
words = []
frequency = []
for i in fdist:
    words.append(i[0])
    frequency.append(i[1])
words.reverse()
frequency.reverse()

plt.barh(words, frequency)
plt.savefig("charts/DreamworksNegBarPlot")
plt.show()

wordcloud = WordCloud(stopwords=stop_words, max_words=100, background_color="white").generate(neg)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("charts/DreamworksNegWordCloud.png")

pos = " ".join(pos).lower()
pos = re.sub(r'[^a-zA-Z0-9 ]', '', pos)
dreamworks_tokenized = word_tokenize(pos)

dreamworks_filtered = []
for w in dreamworks_tokenized:
    if w not in stop_words:
        dreamworks_filtered.append(w)

dreamworks_lem = []
for w in dreamworks_filtered:
    dreamworks_lem.append(lem.lemmatize(w))

fdist = FreqDist(dreamworks_lem).most_common(10)
words = []
frequency = []
for i in fdist:
    words.append(i[0])
    frequency.append(i[1])
words.reverse()
frequency.reverse()

plt.barh(words, frequency)
plt.savefig("charts/DreamworksPosBarPlot")
plt.show()

wordcloud = WordCloud(stopwords=stop_words, max_words=100, background_color="white").generate(pos)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("charts/DreamworksPosWordCloud.png")


neu = " ".join(neu).lower()
neu = re.sub(r'[^a-zA-Z0-9 ]', '', neu)
dreamworks_tokenized = word_tokenize(neu)

dreamworks_filtered = []
for w in dreamworks_tokenized:
    if w not in stop_words:
        dreamworks_filtered.append(w)

dreamworks_lem = []
for w in dreamworks_filtered:
    dreamworks_lem.append(lem.lemmatize(w))

fdist = FreqDist(dreamworks_lem).most_common(10)
words = []
frequency = []
for i in fdist:
    words.append(i[0])
    frequency.append(i[1])
words.reverse()
frequency.reverse()

plt.barh(words, frequency)
plt.savefig("charts/DreamworksNeuBarPlot")
plt.show()

wordcloud = WordCloud(stopwords=stop_words, max_words=100, background_color="white").generate(neu)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("charts/DreamworksNeuWordCloud.png")

import re
import pandas as pd
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk import word_tokenize, WordNetLemmatizer, FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

tweets = pd.read_csv("tweets/disney.csv")

disney_content_array = pd.DataFrame(tweets, columns=['content']).to_numpy().flatten()

sid = SentimentIntensityAnalyzer()
lem = WordNetLemmatizer()
stop_words = stopwords.words("english")
stop_words.extend(['disney', 'im', 'show', 'people', 'dont', 'didnt', 'movie', 'disneys', 'one', 'lol',
                   'know', 'cant', 'get', 'got', 'make', 'made', 'think', 'going', 'also', 'would',
                   'thing', 'go', 'see', 'say', 'said', 'u', 'amp', 'thats', 'youre', 'theyre', 'us'])

neg = []
pos = []
neu = []
for i in disney_content_array:
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
disney_tokenized = word_tokenize(neg)

disney_filtered = []
for w in disney_tokenized:
    if w not in stop_words:
        disney_filtered.append(w)

disney_lem = []
for w in disney_filtered:
    disney_lem.append(lem.lemmatize(w))

fdist = FreqDist(disney_lem).most_common(10)
words = []
frequency = []
for i in fdist:
    words.append(i[0])
    frequency.append(i[1])
words.reverse()
frequency.reverse()

plt.barh(words, frequency)
plt.savefig("charts/DisneyNegBarPlot")
plt.show()

wordcloud = WordCloud(stopwords=stop_words, max_words=30, background_color="white").generate(neg)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("charts/DisneyNegWordCloud.png")

pos = " ".join(pos).lower()
pos = re.sub(r'[^a-zA-Z0-9 ]', '', pos)
disney_tokenized = word_tokenize(pos)

disney_filtered = []
for w in disney_tokenized:
    if w not in stop_words:
        disney_filtered.append(w)

disney_lem = []
for w in disney_filtered:
    disney_lem.append(lem.lemmatize(w))

fdist = FreqDist(disney_lem).most_common(10)
words = []
frequency = []
for i in fdist:
    words.append(i[0])
    frequency.append(i[1])
words.reverse()
frequency.reverse()

plt.barh(words, frequency)
plt.savefig("charts/DisneyPosBarPlot")
plt.show()

wordcloud = WordCloud(stopwords=stop_words, max_words=30, background_color="white").generate(pos)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("charts/DisneyPosWordCloud.png")


neu = " ".join(neu).lower()
neu = re.sub(r'[^a-zA-Z0-9 ]', '', neu)
disney_tokenized = word_tokenize(neu)

disney_filtered = []
for w in disney_tokenized:
    if w not in stop_words:
        disney_filtered.append(w)

disney_lem = []
for w in disney_filtered:
    disney_lem.append(lem.lemmatize(w))

fdist = FreqDist(disney_lem).most_common(10)
words = []
frequency = []
for i in fdist:
    words.append(i[0])
    frequency.append(i[1])
words.reverse()
frequency.reverse()

plt.barh(words, frequency)
plt.savefig("charts/DisneyNeuBarPlot")
plt.show()

wordcloud = WordCloud(stopwords=stop_words, max_words=30, background_color="white").generate(neu)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("charts/DisneyNeuWordCloud.png")

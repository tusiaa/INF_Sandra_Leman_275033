import re
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud

tweets = []
dreamworks_content_array = []
for j in range(5):
    tweets.append(pd.read_csv(f"tweets/months/dreamworks{j+1}.csv"))
    dreamworks_content_array.append(pd.DataFrame(tweets[j], columns=['content']).to_numpy().flatten())

sid = SentimentIntensityAnalyzer()

for j in range(5):
    neg = []
    pos = []
    neu = []
    for i in dreamworks_content_array[j]:
        if sid.polarity_scores(i)["compound"] > 0:
            pos.append(i)
        elif sid.polarity_scores(i)["compound"] == 0:
            neu.append(i)
        else:
            neg.append(i)

    print(j+1)
    print("neg: ", len(neg))
    print("neu: ", len(neu))
    print("pos: ", len(pos))


stop_words = stopwords.words("english")
stop_words.extend(['dreamworks', 'im', 'show', 'people', 'dont', 'didnt', 'movie', 'disney', 'disneys', 'one', 'lol',
                   'know', 'cant', 'get', 'got', 'make', 'made', 'think', 'going', 'also', 'would', 'film', 'guy', '2',
                   'thing', 'go', 'see', 'say', 'said', 'thats', 'youre', 'theyre', 'us', 'animation', 'movies', 'amp'])

march = " ".join(dreamworks_content_array[2]).lower()
march = re.sub(r'[^a-zA-Z0-9 ]', '', march)

april = " ".join(dreamworks_content_array[3]).lower()
april = re.sub(r'[^a-zA-Z0-9 ]', '', april)

wordcloud = WordCloud(stopwords=stop_words, max_words=30, background_color="white").generate(march)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("charts/DreamworksMarchWordCloud.png")

wordcloud = WordCloud(stopwords=stop_words, max_words=30, background_color="white").generate(april)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("charts/DreamworksAprilWordCloud.png")

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

tweets = []
dreamworks_content_array = []
for j in range(5):
    tweets[j] = pd.read_csv(f"tweets/months/dreamworks{j+1}.csv")
    dreamworks_content_array[j] = pd.DataFrame(tweets[j], columns=['content']).to_numpy().flatten()

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

    print(j)
    print("neg: ", len(neg))
    print("neu: ", len(neu))
    print("pos: ", len(pos))




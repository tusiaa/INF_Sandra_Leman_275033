from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import nltk
# nltk.download('vader_lexicon')

with open('opinion1.txt', encoding="utf8") as f:
    opinion1 = f.read()
with open('opinion2.txt', encoding="utf8") as f:
    opinion2 = f.read()

sid = SentimentIntensityAnalyzer()
print(opinion1)
ss = sid.polarity_scores(opinion1)
for k in sorted(ss):
    print('{0}: {1}, \n'.format(k, ss[k]), end='')

print(opinion2)
ss = sid.polarity_scores(opinion2)
for k in sorted(ss):
    print('{0}: {1}, \n'.format(k, ss[k]), end='')



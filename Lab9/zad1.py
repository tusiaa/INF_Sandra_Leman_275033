from wordcloud import WordCloud
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

with open('article.txt') as f:
    article = f.read().lower()

tokenized_text = word_tokenize(article)

stop_words = stopwords.words("english")
stop_words.extend(['.', ',', '\'\'', '"', '``', '(', ')', '\'s'])

filtered_text = []
for w in tokenized_text:
    if w not in stop_words:
        filtered_text.append(w)

ps = PorterStemmer()
stemmed_text = []
for w in filtered_text:
    stemmed_text.append(ps.stem(w))

lem = WordNetLemmatizer()
lem_text = []
for w in filtered_text:
    lem_text.append(lem.lemmatize(w))

print("\nTokenized:", tokenized_text)
print("Liczba slow:", len(tokenized_text))

fdist = FreqDist(tokenized_text)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nFiltered:", filtered_text)
print("Liczba slow:", len(filtered_text))

fdist = FreqDist(filtered_text)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nStemmed:", stemmed_text)
print("Liczba slow:", len(stemmed_text))

fdist = FreqDist(stemmed_text)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

print("\nLemmatized:", lem_text)
print("Liczba slow:", len(lem_text))

fdist = FreqDist(lem_text)
print("Najczesciej wystepujace", fdist.most_common(10))
print("Liczba slow:", len(fdist))

fdist = FreqDist(lem_text).most_common(10)
words = []
frequency = []
for i in fdist:
    words.append(i[0])
    frequency.append(i[1])
words.reverse()
frequency.reverse()

plt.barh(words, frequency)
plt.savefig("BarPlot1")
plt.show()

wordcloud = WordCloud(stopwords=stop_words).generate(article)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("WordCloud1.png")



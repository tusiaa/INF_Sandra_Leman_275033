import itertools
import snscrape.modules.twitter as sntwitter
import pandas as pd

search = '"#disney lang:en"'

tweets = itertools.islice(sntwitter.TwitterSearchScraper(search).get_items(), 10000)

df = pd.DataFrame(tweets)[['date', 'content']]
df.to_csv("tweets/disney.csv")
print(df)


search = '"#dreamworks lang:en"'

tweets = itertools.islice(sntwitter.TwitterSearchScraper(search).get_items(), 10000)

df = pd.DataFrame(tweets)[['date', 'content']]
df.to_csv("tweets/dreamworks.csv")
print(df)


search = '"#pixar lang:en"'

tweets = itertools.islice(sntwitter.TwitterSearchScraper(search).get_items(), 10000)

df = pd.DataFrame(tweets)[['date', 'content']]
df.to_csv("tweets/pixar.csv")
print(df)



import pandas as pd
import snscrape.modules.twitter as sntwitter
import itertools

search = '"disney"'

tweets = itertools.islice(sntwitter.TwitterSearchScraper(search).get_items(), 100)

df = pd.DataFrame(tweets)[['date', 'content']]
df.to_csv("100disney.csv")
print(df)

search = '"disney near:"Gdańsk" within:10km since:2022-01-01 until:2022-05-30"'

tweets = sntwitter.TwitterSearchScraper(search).get_items()

df = pd.DataFrame(tweets)[['date', 'content']]
df.to_csv("disneyGdańsk.csv")
print(df)




import itertools
import snscrape.modules.twitter as sntwitter
import pandas as pd

for j in range(5):
    df1 = pd.DataFrame()
    for i in range(30):
        search = f'"#disney lang:en since:2022-0{j+1}-{i+1} until:2022-0{j+1}-{i+2}"'
        tweets = itertools.islice(sntwitter.TwitterSearchScraper(search).get_items(), 1000)
        df2 = pd.DataFrame(tweets, columns=['date', 'content'])
        df1 = pd.concat([df1, df2], ignore_index=True)
    print(df1)
    df1.to_csv(f"tweets/months/disney{j+1}.csv")

for j in range(5):
    df1 = pd.DataFrame()
    for i in range(30):
        search = f'"#dreamworks lang:en since:2022-0{j+1}-{i+1} until:2022-0{j+1}-{i+2}"'
        tweets = itertools.islice(sntwitter.TwitterSearchScraper(search).get_items(), 1000)
        df2 = pd.DataFrame(tweets, columns=['date', 'content'])
        df1 = pd.concat([df1, df2], ignore_index=True)
    print(df1)
    df1.to_csv(f"tweets/months/dreamworks{j+1}.csv")

for j in range(5):
    df1 = pd.DataFrame()
    for i in range(30):
        search = f'"#pixar lang:en since:2022-0{j+1}-{i+1} until:2022-0{j+1}-{i+2}"'
        tweets = itertools.islice(sntwitter.TwitterSearchScraper(search).get_items(), 1000)
        df2 = pd.DataFrame(tweets, columns=['date', 'content'])
        df1 = pd.concat([df1, df2], ignore_index=True)
    print(df1)
    df1.to_csv(f"tweets/months/pixar{j+1}.csv")


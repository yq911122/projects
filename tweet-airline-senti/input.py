import pandas as pd

data = pd.read_csv("input/Airline-Sentiment-2-w-AA.csv")
data.columns = [col.replace(":", "_") for col in data.columns]

columns_to_keep = [u'airline_sentiment', u'airline', u'retweet_count', u'text', 'tweet_created']

data = data[columns_to_keep]

data['sentiment'] = data.airline_sentiment.map({'negative':0,'neutral':2,'positive':4})
data['tweet_created'] = pd.to_datetime(data['tweet_created'])
data['hour'] = data.tweet_created.map(lambda x: x.hour)
data['dayofweek'] = data.tweet_created.map(lambda x: x.dayofweek)
data = data.drop(['airline_sentiment','tweet_created'], axis=1)

data.to_csv("input/Tweets.csv", index=False)

data = data.drop('text', axis=1)
data.to_csv("input/Tweets_R.csv", index=False)
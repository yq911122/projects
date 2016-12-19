# %matplotlib inline

import matplotlib.pyplot as plt

import pandas as pd

import importHelper

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

import re

pd.options.mode.chained_assignment = None

lm = importHelper.load('lm')
proc = importHelper.load('textpreprocessor')
cv = importHelper.load('cvMachine')

URL = './input/Tweets.csv'	
def load_data(url=URL):
	return pd.read_csv(url)

def clean_tweet(s):
	'''
	:s : string; a tweet

	:return : list; words that don't contain url, @somebody, and in utf-8 and lower case
	'''
	extra_patterns=['date','time','url','at']

	s = unicode(s, errors='ignore')
	s = re.sub(clean_tweet.pattern_airline, '', s, 1)

	sents = sent_tokenize(s)
	# sents, entities = proc.remove_entities(sents, extra_patterns)

	words = [word_tokenize(s) for s in sents]
	words = [e for sent in words for e in sent]
	# return [e.lower() for e in words]
	return [clean_tweet.stemmer.stem(e.lower()) for e in words]

def plot_stop_words(s):
	freqwords = proc.get_stop_words_and_freq(s,n=100)

	freq = [s[1] for s in freqwords]

	plt.title('frequency of top 100 most frequent words')
	plt.plot(freq)
	plt.xlim([-1,100])
	plt.ylim([0,1.1*max(freq)])
	plt.ylabel('frequency')
	plt.show()

def clean_data(df):
	import itertools
	import numpy as np

	df = df[df['retweet_count'] <= 2]

	clean_tweet.stemmer = PorterStemmer()
	clean_tweet.pattern_airline = re.compile(r'@\w+')

	df.loc[:,'text'] = df.loc[:,'text'].map(clean_tweet)
	# plot_stop_words(df['text'])
	df.loc[:,'text'] = proc.remove_stop_words(df['text'],n=20)
	
	airlines = df['airline'].unique()
	dayofweek = df['dayofweek'].unique()
	dfs = [df[(df['airline'] == a)] for a in airlines]
	# df = df.sample(frac=1/6.0)
	# return df.text, df.sentiment
	# dfs = [df[(df['airline'] == a) & (df['dayofweek'] == b)] for (a, b) in itertools.product(*([airlines, dayofweek]))]
	dfs = [df for df in dfs if len(df) >= 10]
	dfs = [df.reindex(np.random.permutation(df.index)) for df in dfs]
	return [(df.text, df.sentiment) for df in dfs]

# def test(x,y,test):
# 	lm = importHelper.load('lm')
# 	m = lm.lm()
# 	m.fit(x,y)
# 	m.predict(test)
# 	return m

def main():
	dfs = clean_data(load_data())
	models = [lm.lm()]*len(dfs)
	avg_score = [cv.cross_validation(model, X, Y, avg=True, cv=2) for model, (X, Y) in zip(models, dfs)]
	print [m.get_predictive_words() for m in models]
	# print avg_score

if __name__ == '__main__':
	main()

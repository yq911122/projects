import importHelper
import pandas as pd
import numpy as np

import string

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

import re

ev = importHelper.load('everything')
cv = importHelper.load('cvMachine')
proc = importHelper.load('textpreprocessor')

URL = 'traindata.csv'

def load_data(url=URL):
	df = pd.read_csv(url, header=None)
	df.columns = ['score','usrid','timestamp','no_query', 'usr', 'tweet']
	return df[['score','timestamp','tweet']].sample(frac=0.001)

def clean_tweet(s):
	'''
	:s : string; a tweet

	:return : list; words that don't contain url, @somebody, and in utf-8 and lower case
	'''
	extra_patterns=['date','time','url','at']
	# pattern_at = re.compile(r'@\w+')
	# pattern_url = re.compile(r'^https?:\/\/.*[\r\n]*')
	# s = pattern_at.sub('',pattern_url.sub('',s))
	s = unicode(s, errors='ignore')

	sents = sent_tokenize(s)
	sents, entities = proc.remove_entities(sents, extra_patterns)

	words = [word_tokenize(s) for s in sents]
	words = [e for sents in words for e in sents]
	# return [e.lower() for e in words]
	return [e.lower() if e not in entities else e for e in words]

get_weekday = lambda x: x.split()[0]

def clean_data(df):
	df['weekday'] = df['timestamp'].map(get_weekday)
	df['tweet'] = df['tweet'].map(clean_tweet)
	df['tweet'] = proc.remove_stop_words(df['tweet'], 10, nltk=True)
	df = df.drop('timestamp', 1)
	df = df.reindex(np.random.permutation(df.index))
	return df


def lm_proc(df):
	lm = importHelper.load('lm')

	def clean_data_lm(df):
		def get_xy(df):
			return df.tweet, df.score

		df = clean_data(df)
		weekends = ev.Contain(['Fri', 'Sat', 'Sun', 'Mon'])
		dfs = [df[df['weekday'] != weekends], df[df['weekday'] == weekends]]

		return [get_xy(df) for df in dfs]

	def load_lms(params):
		import itertools
		keys = params.keys()
		params_grid = itertools.product(*(params.values()))
		return [lm.lm(**dict(zip(keys,par))) for par in params_grid]

	dfs = clean_data_lm(df)

	params = {'a': [0.05, 0.1, 0.15, 0.2, 0.25],
			'smooth_method': ['jelinek_mercer','dirichlet']
			}

	models_on_dfs = [load_lms(params)]*len(dfs)
	# models = [m.fit(X, Y) for m, (X, Y) in zip(models, dfs)]

	scores = {}
	for models_on_df, (X, Y) in zip(models_on_dfs, dfs):
		for model in models_on_df:
			params = model.get_params()
			# print params
			avg_score = cv.cross_validation(model, X, Y, avg=True)
			if params not in scores:
				scores[params] = [avg_score]
			else: scores[params].append(avg_score)
	return scores

	# if df = df.sample(frac=0.01)
	# scores = {(0.1, 'dirichlet'): [0.95187580853816312, 0.93239901071723019], (0.1, 'jelinek_mercer'): [0.9531694695989652, 0.92720527617477333], (0.15, 'jelinek_mercer'): [0.95135834411384224, 0.92572135201978567], (0.2, 'dirichlet'): [0.9510996119016818, 0.93124484748557301], (0.25, 'jelinek_mercer'): [0.94851228978007762, 0.92151690024732069], (0.05, 'dirichlet'): [0.95291073738680476, 0.93198680956306679], (0.05, 'jelinek_mercer'): [0.95394566623544641, 0.93009068425391594], (0.25, 'dirichlet'): [0.95058214747736103, 0.93107996702390783], (0.2, 'jelinek_mercer'): [0.95058214747736103, 0.9243198680956306], (0.15, 'dirichlet'): [0.95135834411384224, 0.9316570486397362]}

def dis_proc(df):
	def clean_data_dis(df):
		pass
	pass

def test():
	df = load_data(URL)
	# print len(df)
	return dis_proc(df)



def main():
	df = load_data(URL)
	# lm_scores = lm_proc(df)
	dis_scores = dis_proc(df)
	print dis_scores

if __name__ == '__main__':
	main()


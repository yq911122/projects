import pandas as pd
import numpy as np

import importHelper

import loader

lm = importHelper.load('lm')

class lm_cook_processer():
	"""
	process data into right format for classification. stop words are considered. items in each recipe is 'cleaned' by extracting the last word of each item. For example, "soy milk" will be "milk" after processing, which can be chosen whether to applied or not by tiny change. 
	"""
	stop_words = []
	
	def __init__(self, df, test=False):
		self.df = df
		self.test = test

	def get_stop_words(self, s, n=8):
		'''
		(u'salt', 23743), (u'pepper', 23569), (u'oil', 22824), (u'sauce', 12822), (u'onions', 12341), (u'sugar', 12340), (u'cheese', 10837), (u'water', 9572), (u'garlic', 9555)
		'''
		from collections import Counter
		l = []
		s.map(lambda x: l.extend(x))
		return [x[0] for x in Counter(l).most_common(n)]

	def union_list(self):
		self.df.loc[:,'ingredients'] = self.df.loc[:,'ingredients'].map(self._wrap_last_word)
		if not self.test: lm_cook_processer.stop_words = self.get_stop_words(self.df['ingredients'])
		self.df.loc[:,'ingredients'] = self.df.loc[:,'ingredients'].map(lambda x: [e for e in x if e not in lm_cook_processer.stop_words])

	def _wrap_last_word(self,l):
		return [e.split(' ')[-1] for e in l]		

def clean_data(proc):
	proc.union_list()
	return proc.df

def cross_validation(df, x_name, y_name, par, cv=5):
	df = df.reindex(np.random.permutation(df.index))
	df['row'] = range(len(df))
	df.set_index(['row'],inplace=True)
	k = [(len(df)-1)/cv*j for j in range(cv+1)]
	score = [0.0]*cv
	for i in range(cv):		
		train = pd.concat([df.loc[:k[i],:],df.loc[k[i+1]:,:]])
		test = df.loc[k[i]:k[i+1],:]
		model = lm.fit(train,x_name,y_name,par)
		pred = lm.predict(test[x_name],model)
		score[i] = (pred == test[y_name]).sum()/float(len(test))
	return sum(score)/float(cv)

def pred_to_out(pred,index):
	'''
	transform data into the file in the same format of the sample file
	'''
	df = pd.DataFrame(pred)
	df['Id'] = index
	df.set_index(['Id'],inplace=True)
	df.columns = ['cuisine']
	df.to_csv('result.csv',index_label='Id')

def main():

	train = loader.load_data() 
	train = clean_data(lm_cook_processer(train))

	test = loader.load_data(test=True)
	test = clean_data(lm_cook_processer(test,test=True))

	# tuning parameter a for language model
	# a = [0.05,0.1,0.15,0.2,0.25,0.3]
	# for e in a:
	# 	print "a: " + str(e)
	# 	print "score: " + str(cross_validation(train,'ingredients','cuisine', e))
	model = lm(train,'ingredients','cuisine')
	pred = lm.predict(test['ingredients'])
	pred_to_out(pred,test.index)
	

if __name__ == '__main__':
	main()
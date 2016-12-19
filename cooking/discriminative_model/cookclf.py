import pandas as pd
import importHelper

import loader

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

proc = importHelper.load('preprocessor')
# cross_validation = importHelper.load('cvMachine')


class tfidf_cook_processer(proc.preprocessor):
	"""
	process data into right format for classification. tf*idf is applied to transform the data and stop words are considered. 
	items in each recipe is 'cleaned' by extracting the last word of each item. For example, "soy milk" will be "milk" after processing, which can be chosen whether to applied or not by tiny change. 

	the class is inherited by preprocessor in preprocessor.py, see yq911122/module on Github.

	"""

	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.feature_extraction.text import TfidfTransformer
	cv = CountVectorizer()
	tfidf = TfidfTransformer()
	stop_words = []

	def __init__(self, df, test=False, y_name = 'cuisine'):
		super(tfidf_cook_processer, self).__init__(df,test,y_name)
		

	def get_stop_words(self, s, n=8):
		'''
		out: (u'salt', 23743), (u'pepper', 23569), (u'oil', 22824), (u'sauce', 12822), (u'onions', 12341), (u'sugar', 12340), (u'cheese', 10837), (u'water', 9572), (u'garlic', 9555)
		'''
		from collections import Counter
		l = []
		s.map(lambda x: l.extend(x))
		l = [x[0] for x in Counter(l).most_common(n)]
		return l

	def to_tfidf(self):
		l = []
		self.X['ingredients'].map(lambda x: l.extend([x]))
		if not self.test:
			cv_fit_voc = tfidf_cook_processer.cv.fit_transform(l)
			self.X = tfidf_cook_processer.tfidf.fit_transform(cv_fit_voc)
		else:
			self.X = tfidf_cook_processer.tfidf.transform(tfidf_cook_processer.cv.transform(l))

	def numerize_y(self):
		self.Y, ref = self._numerizeSeries(self.Y)
		ref = {v:k for k,v in ref.items()}
		return ref

	def union_list(self, stop_words=True):
		self.X.loc[:,'ingredients'] = self.X.loc[:,'ingredients'].map(self._wrap_last_word)	# comment it if preserve all items rather than select the last word of each item
		if stop_words:
			if not self.test: 
				tfidf_cook_processer.stop_words = self.get_stop_words(self.X['ingredients'])
			self.X.loc[:,'ingredients'] = self.X.loc[:,'ingredients'].map(lambda x: [e for e in x if e not in tfidf_cook_processer.stop_words]).map(' '.join)
		else:
			self.X.loc[:,'ingredients'] = self.X.loc[:,'ingredients'].map(' '.join)


	def _wrap_last_word(self,l):
		return [e.split(' ')[-1] for e in l]

def clean_data(proc,test=False):
	proc.union_list()
	proc.to_tfidf()
	if not test: 
		ref = proc.numerize_y()
		return ref, proc.getProcessedData()
	return proc.getProcessedData()


# def test():
	# '''
	# for debug use
	# '''
# 	train = loader.load_data() 
# 	ref, (train_x, train_y) = clean_data(tfidf_cook_processer(train))

# 	test = loader.load_data(test=True)
# 	test_x = clean_data(tfidf_cook_processer(test,test=True),test=True)
# 	return train_x, train_y, test_x, test, ref

def pred_to_out(pred,ref,test):
	'''
	transform predict values to file in the format of the sample output
	'''
	df = pd.DataFrame(pred)
	df['Id'] = test.index
	df['cuisine'] = df[0].map(ref)
	df.drop([0],axis=1,inplace=True)
	df.set_index(['Id'],inplace=True)
	df.to_csv('result_tuning.csv',index_label='Id')



def main():

	train = loader.load_data() 
	ref, (train_x, train_y) = clean_data(tfidf_cook_processer(train))

	test = loader.load_data(test=True)
	test_x = clean_data(tfidf_cook_processer(test,test=True),test=True)

	# get from cross validation
	gamma = 1	
	C = 3.1622776601683795

	clf = SVC(gamma=gamma, C=C, probability=True)
	# print cross_validation.cvScore(clf, train_x, train_y).mean()

	# random forest
	# clf = RandomForestClassifier(n_estimators=100) #rank 1078

	clf.fit(train_x,train_y)
	pred_to_out(clf.predict(test_x),ref,test)


if __name__ == '__main__':
	main()

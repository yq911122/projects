import pandas as pd

trainUrl = 'train.json'
testUrl = 'test.json'

def load_data(test=False):
	'''
	load data from .json file and transform it into pd.DataFrame.
	'''
	if not test:
		df = pd.read_json(trainUrl)
	else: df = pd.read_json(testUrl)
	df.set_index(['id'],drop=True, inplace=True)
	return df
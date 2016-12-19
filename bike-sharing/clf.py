import pandas as pd
import numpy as np

from sklearn import linear_model
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tsclf import GroupRegressor

from numpy import vectorize

TRAINURL = './input/train_processed.csv'
TESTURL = './input/test_processed.csv'

#['weather','atemp','humidity','windspeed','hour','month','holiday','workingday','dayofweek','season','casual','registered','count']
features = ['weather','atemp','humidity','windspeed','hour','month','holiday','workingday','dayofweek','year']

y_cols = ['casual','registered','count']

cols_train = features + y_cols
cols_test = features

import math
exp = vectorize(math.exp)
log = vectorize(math.log)

# #A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
# def rmsle(y, y_pred):
# 	assert len(y) == len(y_pred)
# 	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
# 	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5


def load_data(url):
	if url == TRAINURL:	return pd.read_csv(url, usecols = cols_train)
	if url == TESTURL: return pd.read_csv(url, usecols = cols_test)

def clean_data(df, istrain=True):
	# df['season'] = df['month'].apply(lambda x: 1*int(1<=x<=3)+2*int(4<=x<=6)+3*int(7<=x<=9)+4*(10<=x<=12))
	# df = df.drop('month',axis=1)
	if istrain:	return df.reindex(np.random.permutation(df.index))
	return df

def log_plus_1_transform(series):
	return (series+1).apply(math.log)


def clean_data_linear_model(df, istrain=True):
	df['casual_peak'] = df['hour'].apply(lambda x: int((8 < x < 20)))
	df['registered_peak'] = df['hour'].apply(lambda x: int(((6 < x < 10) or (16 < x < 20))))
	df['weekend'] = df['dayofweek'].apply(lambda x: int(x == 5 or x == 6))
	df['year'] = df['year'] - df['year'].min()
	# return df
	return df.drop(['hour','dayofweek'], axis=1)

def fit_and_predict(model, train_x, train_y, test_x):
	model.fit(train_x, train_y)
	# print model.score(train_x, train_y)
	# print train_x.columns
	# print model.coef_
	return pd.Series(model.predict(test_x))



def main(): 
	train = clean_data(load_data(TRAINURL))
	test = clean_data(load_data(TESTURL),istrain=False)
	

	# clf = RandomForestRegressor(n_estimators=2000, min_samples_split = 11, random_state = 0)	#0.38038
	# params = {'n_estimators': 150, 'max_depth': 5, 'random_state': 0, 'min_samples_leaf' : 10, 'learning_rate': 0.1, 'subsample': 0.7, 'loss': 'ls'}
	# clf = GradientBoostingRegressor(**params)	#0.37033	

	clf = GroupRegressor(y_transform=log_plus_1_transform, subModel=np.mean)
	# train = clean_data_linear_model(train)
	# test = clean_data_linear_model(test,istrain=False)
	

	# train_x = train[[e for e in train.columns if e not in ['registered_peak'] + y_cols]]
	# test_x = test[[e for e in test.columns if e != 'registered_peak']]
	train_x = train[[e for e in train.columns if e not in y_cols]]
	test_x = test

	print train_x.head()
	print test_x.head()
	clf.setGroupCols(['year','hour','month','dayofweek'])
	pred_casual = fit_and_predict(clf, train_x, train['casual'], test_x)
	# print clf.feature_importances_
	# print pred_casual.head()

	# train_x = train[[e for e in train.columns if e not in ['causal_peak'] + y_cols]]
	# test_x = test[[e for e in test.columns if e != 'causal_peak']]
	# clf.setGroupCols(['year','season','registered_peak','weekend'])
	pred_registered = fit_and_predict(clf, train_x, train['registered'], test_x)
	# print clf.feature_importances_
	# print pred_registered.head()

	# pred to output
	test_raw = pd.read_csv('./input/test.csv',usecols=['datetime'])
	# pred_count = pred_casual + pred_registered	#0.56372!! for np.mean
	pred_count = pred_casual.map(math.exp) + pred_registered.map(math.exp) - 2	#	0.57176 for np.mean
	# pred_count = fit_and_predict(clf, train_x, train['count'], test)	#0.56372 for np.mean
	pred_count = pd.DataFrame(pred_count)
	pred_count.set_index(test_raw['datetime'],inplace=True)
	pred_count.columns = ['count']
	pred_count.to_csv('res.csv')

	# print rmsle(list(train_b['count']),pred_count)

if __name__ == '__main__':
	main()
	
import pandas as pd
import numpy as np
import importHelper

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC

from collections import Counter

import ga
import wrapper

import math

trainURL = '../train.csv'
testURL = '../test.csv'

cv = importHelper.load('cvMachine')
proc = importHelper.load('preprocessor')

def load_data(url):
	df = pd.read_csv(url, index_col=0)
	return df

def clean_data(df, test=False):
	# remove features with almost zero variance plus 'Wilderness_Area4' and 'Soil_Type40' in case of Multicollinearity
	# remove_list = proc.small_var(df,0.002)
	# remove_list.extend(['Wilderness_Area3', 'Soil_Type10','Hillshade_9am','Hillshade_Noon','Hillshade_3pm'])
	remove_list = ['Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type15', 'Soil_Type21', 'Soil_Type25', 'Soil_Type27', 'Soil_Type28', 'Soil_Type34', 'Soil_Type36', 'Wilderness_Area3', 'Soil_Type10','Hillshade_9am','Hillshade_Noon','Hillshade_3pm']
	df = df.drop(labels = remove_list, axis=1)
	# df = df.astype(np.float32)
	if not test: df = proc.shuffle(df)
	return df


def wrapper_method(x, y):
	'''
	apply wrapper model to select features
	
	:x : features in transet
	:y : labels in transet
	
	:return : columns' indexes that should be reserved
	'''
	clf = RandomForestClassifier(n_estimators=20)
	wrp = wrapper.wrapper(x, y, clf, ga.ga)
	best_ind = wrp.select_features()
	# print best_ind
	cols_inx = [i for i in range(len(best_ind)) if best_ind[i] == 1]
	return cols_inx

def select_features(X, Y, method=wrapper_method):
	return method(X,Y)
	

def pred_to_out(pred,id_col):
	'''
	from prediction result to the format same to sample and write it into .csv file

	:pred : prediction result
	:id_col: original id from test dataset 
	'''
	result = pd.DataFrame(pred)
	result['Id'] = id_col
	result.set_index(['Id'],inplace=True)
	result.columns = ['Cover_Type']
	result.to_csv('raw_result2.csv')


def get_xy(df):
	return df[[e for e in df.columns if e != 'Cover_Type']], df['Cover_Type']	

def predict(test,clf, k):
	pred = []
	tests = np.vsplit(test, k)
	i = 0
	for t in tests:
		i += 1
		# print i/float(k)
		pred.extend(clf.predict(t))
	return pred

def sample(df,wt):
	dfs = [df.loc[df[wt['name']] == i,:] for i in wt['value'].keys()]
	dfs_sampled = [df.sample(frac=i) for df, i in zip(dfs,wt['value'].values())]
	return pd.concat(dfs_sampled)

def weight_from_prediction(pred):
	wt = Counter(pred)
	wt = {k:v/float(max(wt.values())) for k,v in wt.items()}
	return wt

def weight_distance(w1, w2):
	diff2 = 0.0
	for k in w1.keys():
		try: diff2 += (w1[k]-w2[k])**2
		except KeyError: continue
	return math.sqrt(diff2)

def main():
	np.set_printoptions(suppress=True)

	train = clean_data(load_data(trainURL))
	test = clean_data(load_data(testURL), test=True)

	# MINISIZE = 10000
	i = 0
	ITER = 20
	k = 23

	train_x, train_y = 	get_xy(train)

	clf = ExtraTreesClassifier(n_estimators=100, max_features='log2',max_depth=40)
	clf.fit(train_x,train_y)
	pred = predict(test,clf,k)
	
	wt = weight_from_prediction(pred)
	prev_wt = wt
	print wt
	wt = {'name':'Cover_Type','value':wt}

	while i < ITER:	#0.77293, 0.80570, 0.80497
	
	# train_x2 = train_x.drop(['Wilderness_Area1','Wilderness_Area2','Wilderness_Area4'], axis=1)
	# train_x3 = train_x.drop(['Hillshade_9am','Hillshade_Noon','Hillshade_3pm'],axis=1)
	# rsrv_inx = select_features(train_x3, train_y)
	# train_x3 = train_x3[rsrv_inx]
	# train_x = train_x[rsrv_inx]
	# test = test[rsrv_inx]

	# wt = {1: 0.36460520608868663, 2: 0.4875992234239568, 3: 0.06153745533655071, 4: 0.004727957426008412, 5: 0.01633873310706147, 6: 0.029890948896064105, 7: 0.03530047572167184}
	
		i += 1
		print i

		train_xi, train_yi = get_xy(sample(train,wt))

		clf.fit(train_xi,train_yi)
		pred = predict(test,clf,k)
		
		wt = weight_from_prediction(pred)
		print weight_distance(prev_wt, wt)
		prev_wt = wt
		print wt
		wt = {'name':'Cover_Type','value':wt}

	# print cv.confusion_matrix_from_cv(clf, train_x, train_y)

	# print wt

	# print cv.confusion_matrix_from_cv(clf, train_x, train_y)

	 # [    0.77375353     0.73424658     0.84722222     0.9771167      0.9596662    0.90098545     0.97332063]]

	# params = {"max_depth":[33,35,37]}

	# clf = SVC(C=2**5.5,gamma=2**-18)

	# print cv.sklearn_cross_validation(clf, train_x, train_y)

	# clf = SVC()
	# params = {"C": [2**5.5], "gamma": [2**i for i in [-17.5,-18,-18.5]]}
	# print cv.param_selector(clf, params, train_x, train_y)

	# clf.fit(train_x, train_y)
	

	pred_to_out(pred,test.index)

	

if __name__ == '__main__':
	main()


#C=4, gamma=0.0009765625
#[ 0.30026455  0.31613757  0.29563492  0.33597884  0.51025132]

#C=16,gamma=3.81469726562e-06
#[ 0.7417328   0.73313492  0.73082011  0.73082011  0.83465608]

#C=32,gamma=3.81469726562e-06
#[ 0.7526455   0.73974868  0.7364418   0.72916667  0.83763228]

#C=32,gamma=3.81469726562e-06
#[ 0.7526455   0.73974868  0.7364418   0.72916667  0.83763228]

#C=2**5.5,gamma=2**-18
#[ 0.75892857  0.7417328   0.73743386  0.73247354  0.84193122]

#max_depth=None
#[ 0.77943122  0.75231481  0.78108466  0.80787037  0.86574074]

#max_depth=40
#[ 0.77843915  0.75727513  0.78174603  0.80820106  0.86474868]

#max_depth=35
#[ 0.77414021  0.75793651  0.77612434  0.80224868  0.86309524]
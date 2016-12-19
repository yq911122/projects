import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import importHelper

from sklearn.ensemble import RandomForestClassifier

cv = importHelper.load('cvMachine')
proc = importHelper.load('preprocessor')

trainUrl = './data/train.csv'
testUrl = './data/test.csv'

nafrac = 0.6
corthr = 0.6

def load_data(url):
	return pd.read_csv(url, index_col=0)

def get_correlated_cols(df, thr):
	cor = np.corrcoef(df.T)
	np.fill_diagonal(cor,0)
	cor_indice = np.where(cor>thr)
	cor_indice = [(i,j) for i, j in zip(cor_indice[0], cor_indice[1])]
	cor_cols = []
	for (i,j) in cor_indice:
		if (j,i) not in cor_cols:
			cor_cols.append((df.columns[i], df.columns[j]))
	return cor_cols
	 

def clean_data(df):
	# fig, ax = plt.subplots()
	# ax.hist(df.count(),20)
	# plt.show()
	def clean_df(df):
		str_cols_sub = [e for e in str_cols if e in df.columns]
		num_cols_sub = [e for e in num_cols if e in df.columns]
		x_cols = [e for e in df.columns if e != 'target']
		df.dropna(thresh=nafrac*len(df.columns), inplace=True)
		df[str_cols_sub] = df[str_cols_sub].fillna(df[str_cols_sub].mode())
		df[num_cols_sub] = df[num_cols_sub].fillna(df[num_cols_sub].mean())
		cor_cols = get_correlated_cols(df[x_cols],corthr)
		rmv_cols = [e[0] for e in cor_cols]
		df = df[[e for e in df.columns if e not in rmv_cols]]

		return df

	type_list = df.dtypes
	str_cols = type_list[type_list == np.object].index.tolist()
	num_cols = type_list[type_list != np.object].index.tolist()
	df[str_cols] = df[str_cols].apply(lambda l: proc.numerize(l,False))
	#train set
	if not clean_data.l:
		x_cols = [e for e in df.columns if e != 'target']

		small_var_cols = proc.small_var(df[x_cols])
		df.drop(small_var_cols, axis=1, inplace=True)
		
		x_cols = [e for e in x_cols if e not in small_var_cols]
		count = df[x_cols].count()
		clean_data.l = count[count>100000].index.tolist()

		df1 = df[[e for e in df.columns if e not in clean_data.l]]
		df1 = clean_df(df1)

		idx2 = [e for e in df.index if e not in df1.index]
		df2 = df[['target']+clean_data.l].loc[idx2,:]

		df2 = clean_df(df2)
		

	return df1, df2
clean_data.l = None

	# df1 = df.dropna(how='any')
	# df2 = df.loc[[e for e in df.index if e not in df1.index],:]
	# return df1, df2

def plot_distributions(df):
	pass


# ['v3', 'v10', 'v12', 'v14', 'v21', 'v22', 'v24', 'v31', 'v34', 'v38', 'v40', 'v47', 'v50', 'v52', 'v56', 'v62', 'v66', 'v71', 'v72', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110', 'v112', 'v114', 'v125', 'v129']



def main():
	train = load_data(trainUrl)

	train1, train2 = clean_data(train)
	print len(train1.columns)-1
	print len(train2.columns)-1
	# train1.to_csv('./data/train1.csv')
	# train2.to_csv('./data/train2.csv')
	x_cols1 = [e for e in train1.columns if e != 'target']
	x_cols2 = [e for e in train2.columns if e != 'target']
	# train1 = train[l]
	# train2 = train[[e for e in train.columns if e not in l]]

	
	# train1, train2 = clean_data(train[x_cols])

	clf = RandomForestClassifier()
	s1 = cv.sklearn_cross_validation(clf, train1[x_cols1], train1['target'])
	s2 = cv.sklearn_cross_validation(clf, train2[x_cols2], train2['target'])
	a1, a2 = len(train1), len(train2)
	a1, a2 = a1/float(a1+a2), a2/float(a1+a2)
	print [a1*x+a2*y for x, y in zip(s1,s2)]



	# 1: simple dropna(how='any') reaches [ 0.68552928  0.66525901  0.68355856  0.67774648  0.68028169] in cv=5, rf(default setting)
	# 2: dropping small vars + split trainset into > 100000 cols and dropna(frac) and fillna with mean or mode reaches [0.70375149918891822, 0.70567578096728578, 0.70365324437051702, 0.70492528699497847, 0.70160101110958561] in cv=5, rf(default setting)
	# 3: based on 2, removing correlated features (cor>0.6) results in [0.70620058280931985, 0.70230829804093831, 0.70194761995307153, 0.70737477037675778, 0.69910777262755464] with only 10 and 13 features, compared with orginial 52 and 24 features, potentailly prevent overfitting and collinearillty.

if __name__ == '__main__':
		main()	
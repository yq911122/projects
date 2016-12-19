# statistical language model. dirichlet and jelinek mercer discount methods are applied

import importHelper
import pandas as pd
import random

everything = importHelper.load('everything')
static_vars = everything.static_vars

def jelinek_mercer(ct, p_ml,p_ref,a=0.1):
	from math import log
	log_p_s = (p_ml*(1-a)+p_ref.loc[p_ml.index]*a).map(log)
	return log_p_s

def dirichlet(ct, p_ml,p_ref,a=0.1):
	from math import log
	d = len(p_ml)
	u = a / (1+a)*d
	log_p_s = ((ct+u*p_ref.loc[ct.index])/(d+u)).map(log)
	return log_p_s

def lm(df, x_name, y_name, a=0.1, smooth_method=jelinek_mercer):
	'''
	df: DataFrame containing features and category. Features are actually a list of words, standing for the document.
	x_name: column name of features
	y_name: column name of the category
	a: discount parameter; should be tuned via cross validation
	smooth_method: method selected to discount the probabilities

	out: language model
	'''
	cats = df[y_name].unique()	
	p_ref = df_to_prob(df[x_name])
	model = pd.DataFrame()
	model['unseen'] = p_ref*a
	for c in cats:
		ct = df_to_ct(df[df[y_name] == c][x_name])
		p_ml = ct_to_prob(ct)
		model[c] = smooth_method(ct, p_ml,p_ref)
		model[c].fillna(model['unseen'],inplace=True)
	model.drop(['unseen'],axis=1,inplace=True)
	return model

def df_to_prob(df):
	'''
	df: [[a],[b,c],...]
	out: pd.DataFrame({a:0.3,b:0.3,...})
	'''
	return ct_to_prob(df_to_ct(df))	

def df_to_ct(df):
	from collections import Counter	
	l = []
	df.map(lambda x: l.extend(x))
	return pd.Series(dict(Counter(l)))

def ct_to_prob(d):
	total_occur = d.sum()
	return d/float(total_occur)


def predict(df, model):
	return df.map(lambda x: predict_item(x,model,len(df)))

@static_vars(counter=0)
def predict_item(l, model, N):
	predict_item.counter += 1
	if predict_item.counter % 20 == 0: print predict_item.counter/float(N)
	in_list = [e for e in l if e in model.index]
	if not in_list: 
		return model.columns[random.randint(0,len(model.columns)-1)]
	s = model.loc[in_list,:].sum(axis=0)
	return s.loc[s==s.max()].index[0]

def main():
	pass

if __name__ == '__main__':
	main()
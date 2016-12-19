import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import os
import warnings

import importHelper

from scipy.stats import itemfreq
from sklearn.ensemble import RandomForestClassifier

cor = importHelper.load("cor")
cv = importHelper.load("cvMachine")

def _histogram(x, ax):
	pars = _histogram.pars
	ax.hist(x, **pars)
_histogram.pars = {}

def plot_df(df,path,f):
	def plot_and_save(l):
		fig, ax = plt.subplots()
		f(l.values, ax)
		plt.title(l.name)
		p = path + l.name + '.png'
		plt.savefig(p)
		plt.close(fig) 

	if not os.path.exists(path):
		os.makedirs(path)
	else:
		warnings.warn("folder already exists, there is risk of overwriting.")

	df.apply(lambda x: plot_and_save(x))

def plot_dfs(dfs, path, f):
	# cols = [set(e.columns) for e in dfs]
	# c1 = cols[0]
	# for e in cols[1:]:
	# 	c1 = c1.intersection(e)
	if not os.path.exists(path):
		os.makedirs(path)
	else:
		warnings.warn("folder already exists, there is risk of overwriting.")

	figarr, axarr = [None]*len(dfs[0].columns), [None]*len(dfs[0].columns)
	i = 0
	for e in dfs[0].columns:
		j = 0
		for df in dfs:
			x = df[e].values
			if axarr[i] is None:
				figarr[i], axarr[i] = plt.subplots(len(dfs), sharex=True)
				f(x, axarr[i][0])
				plt.title(e)
				j += 1
				continue
			f(x, axarr[i][j])
			if j == len(dfs) - 1:
				newpath = path + str(e) + '.png'
				figarr[i].savefig(newpath)
				plt.close(figarr[i])

			j += 1
		i += 1

def cor_boxplot(df):
	cor_df = cor.cor_df(df)
	cor_df = np.reshape(cor_df, cor_df.shape[0]*cor_df.shape[1])
	plt.boxplot(cor_df)
	plt.show()

def discrete_scatter_plot(f1, f2, p):
	fig, ax = plt.subplots()
	items = pd.Series([(v1, v2) for v1, v2 in zip(f1, f2)]).value_counts()
	x = [e[0] for e in items.index.tolist()]
	y = [e[1] for e in items.index.tolist()]
	s = np.sqrt(items.values)
	
	plt.scatter(x, y, s=s, alpha=0.3)
	f1_name, f2_name = str(f1.name), str(f2.name)
	plt.title(f1_name+"-"+f2_name)
	ax.set_xlabel(f1_name)
	ax.set_ylabel(f2_name)
	fig.savefig(p+f1_name+'_'+f2_name+'.png')
	plt.close(fig)

def rm_analysis(df):
	rf = RandomForestClassifier()
	print cv.sklearn_cross_validation(rf, df[['srch_adults_cnt', 'srch_children_cnt']],df['srch_rm_cnt'])
	print cor.cor_df(df[['srch_adults_cnt', 'srch_children_cnt','srch_rm_cnt']])

def main():
	df1 = pd.read_csv('./data/train1_booking.csv', index_col=0)
	df2 = pd.read_csv('./data/train1_unbooking.csv', index_col=0)
	continus_cols2 = [u'orig_destination_distance', u'price_multiplier', u'cnt']
	discrete_cols2 = [u'site_name', u'posa_continent', u'user_location_country',u'user_location_region', u'user_location_city',u'user_id', u'is_mobile', u'is_package', u'channel',u'srch_destination_id', u'srch_destination_type_id', u'hotel_continent', u'hotel_country', u'hotel_market', u'plan_hour', 'people_type', 'trip_type']

	continus_extra, discrete_extra = [u'plan_days', u'travel_days'], [u'travel_month']
	continus_cols1 = continus_cols2 + continus_extra
	discrete_cols1 = discrete_cols2 + discrete_extra

	x_cols1 = continus_cols1 + discrete_cols1
	x_cols2 = continus_cols2 + discrete_cols2
	y_col = [u'hotel_cluster']
	
	# _histogram.pars = {'bins':50, 'alpha':0.5}
	# plot_df(df1[continus_extra+discrete_extra], './plot_booking/', _histogram)
	# plot_dfs([df1[continus_cols2+discrete_cols2+y_col], df2], './plot/', _histogram)

	# cor_df = cor.cor_df(df2)
	# idx = df2.columns.tolist().index(y_col[0])
	# a = pd.Series(cor_df[:,idx], index=df2.columns)
	# print a
	# cor_boxplot(df1)
	# cor_boxplot(df2)

	# cor.plot_cor_cols(df1, 0.1, y_col[0])
	# cor.plot_cor_cols(df2, 0.1, y_col[0])

	# discrete_scatter_plot(df1['travel_days'], df1['plan_days'])
	# discrete_scatter_plot(df1['srch_adults_cnt'], df1['srch_children_cnt'])

	discrete_scatter_plot(df1['hotel_country'], df1['hotel_market'],'./')
	discrete_scatter_plot(df1['hotel_cluster'], df1['hotel_market'],'./')
	discrete_scatter_plot(df1['hotel_country'], df1['hotel_cluster'],'./')
	discrete_scatter_plot(df1['hotel_cluster'], df1['travel_days'],'./')
	discrete_scatter_plot(df1['hotel_cluster'], df1['srch_destination_id'],'./')
	rm_analysis(df1)


if __name__ == '__main__':
	main()
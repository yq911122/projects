import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

import importHelper

lm = importHelper.load('lm')
ev = importHelper.load('everything')

c = 0.2

def mydist(x, y):
	return int(x[0] == y[0]) + c * (x[1] - y[1])**2	

def main():
	df = pd.read_csv('./data/destinations.csv')
	d_cols = [e for e in df.columns if e != 'srch_destination_id']

	ref = pd.read_csv('./data/hotel_destinations1.csv')
	old_dest = df['srch_destination_id'].values.tolist()
	all_dest = ref.index.values.tolist()
	old_dest = [e for e in old_dest if e in all_dest]
	new_dest = [e for e in all_dest if e not in old_dest]

	ref_old = ref.loc[old_dest]
	ref_new = ref.loc[new_dest]

	# print ref_old.shape
	# print ref_new.shape

	ref_old_by_country =  ref_old.groupby(['hotel_country'])
	ref_new_by_country = ref_new.groupby(['hotel_country'])
	x_col = ['srch_destination_type_id','is_package']
	fill = pd.DataFrame(index = ref_new['srch_destination_id'], columns=d_cols)
	fill[d_cols] = 0

	for country, group in ref_new_by_country:
		old_group = ref_old_by_country.get_group(country)
		X, y = old_group[x_col].values, old_group['srch_destination_id'].values
		try:
			knn = KNeighborsClassifier(n_neighbors=4, algorithm='ball_tree',metric='pyfunc', func=mydist)
			knn.fit(X, y)
			_, near_dest = knn.kneighbors(group[x_col].values)
		except ValueError:
			if not isinstance(y, (list, tuple)):
				y = [y]
			near_dest = y
		for dest, neighbors in zip(group['srch_destination_id'].values,near_dest):
			fill.loc[dest] = df[df['srch_destination_id'] == ev.Contain(neighbors)][d_cols].mean()
	fill.reset_index(inplace=True)
	# print fill.head()
	df = pd.concat([df, fill], axis=1)
	df.to_csv('./data/destinations_processed.csv')

if __name__ == '__main__':
	main()
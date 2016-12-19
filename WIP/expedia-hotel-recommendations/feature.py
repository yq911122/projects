import pandas as pd
import numpy as np

import importHelper

dt = importHelper.load("discretizer")
fs = importHelper.load("featureSelector")
entropy = importHelper.load("entropy")
proc = importHelper.load("preprocessor")

def symmetricalUncertainty(x, y):
	return 2*entropy.infoGain(x, y)/(entropy.ent(x)+entropy.ent(y))

def process(df, discrete_cols, continus_cols, y_col):
	x_cols = discrete_cols + continus_cols
	cuts = dt.EntropyDiscretize(df[continus_cols].values, df[y_col].astype(np.int32).values)
	print cuts
	# df[continus_cols] = proc.discretize_df(df, continus_cols, cut)
	# fs = fs.corrSelector(df[x_cols], df[y_col], symmetricalUncertainty, 0.0)
	# print fs.process()


def main():
	df1 = pd.read_csv('./data/train1_booking.csv', index_col=0)
	df2 = pd.read_csv('./data/train1_unbooking.csv', index_col=0)
	continus_cols2 = [u'orig_destination_distance', u'cnt']
	discrete_cols2 = [u'site_name', u'posa_continent', u'user_location_country',u'user_location_region', u'user_location_city',u'user_id', u'is_mobile', u'is_package', u'channel',u'srch_destination_id', u'srch_destination_type_id', u'srch_adults_cnt', u'srch_children_cnt', u'srch_rm_cnt',  u'hotel_continent', u'hotel_country', u'hotel_market', u'plan_hour']

	continus_extra, discrete_extra = [u'plan_days', u'travel_days'], [u'travel_month']
	continus_cols1 = continus_cols2 + continus_extra
	discrete_cols1 = discrete_cols2 + discrete_extra

	x_cols1 = continus_cols1 + discrete_cols1
	x_cols2 = continus_cols2 + discrete_cols2
	y_col = [u'hotel_cluster']

	process(df1, discrete_cols1, continus_cols1, y_col)


if __name__ == '__main__':
	main()

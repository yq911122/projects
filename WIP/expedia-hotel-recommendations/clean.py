import pandas as pd
import numpy as np

import importHelper

ent = importHelper.load("entropy")

def load_data():
	return pd.read_csv('./data/train1.csv', index_col=0, parse_dates=[1, 12, 13])

def people_type(people):
	"""
	:rtype: 1: single adult: child = 0, adult = 1
			2: couples: child = 0, adult = 2
			3: families: child > 0, adult > 0
			4: friends: child = 0, adult > 2
			5: others
	"""
	child, adult = people[0], people[1]
	if child == 0 and adult == 1: return 1
	if child == 0 and adult == 2: return 2
	if child > 0 and adult > 0: return 3
	if child == 0 and adult > 2: return 4
	return 5

def trip_type(days):
	"""
	:rtype: 1: one day immediate trip: plan < 2, travel = 1
			2: multiple days immediate trip: plan < 2, travel > 1
			3: others
	"""
	plan, travel = days[0], days[1]
	if plan < 2 and travel == 1: return 1
	if plan < 2 and travel > 1: return 2
	return 3

def hotel_prob(df, path):
	grouped = df[['srch_destination_type_id','is_package','hotel_country']].groupby(df['srch_destination_id'])
	agg = pd.DataFrame()
	agg['srch_destination_type_id'] = grouped.apply(lambda g: g['srch_destination_type_id'].unique()[0])
	agg['is_package'] = grouped.apply(lambda g: g['is_package'].sum() / float(g.shape[0]))
	agg['hotel_country'] = grouped.apply(lambda g: g['hotel_country'].unique()[0])
	print agg.head()
	agg.to_csv(path+'.csv')


def clean_data(df):
	# df['plan_days'] = (df['srch_ci'] - df['date_time']).dt.days.apply(lambda e: max(e, 0))
	# df['travel_days'] = (df['srch_co'] - df['srch_ci']).dt.days
	# df['travel_month'] = df['srch_ci'].apply(lambda e: e.month)
	# df['plan_hour'] = df['date_time'].apply(lambda e: e.hour)
	# df.drop(['srch_ci','srch_co','date_time'], axis=1, inplace=True)
	# df['orig_destination_distance'].fillna(0, inplace=True)
	# df['price_multiplier'] = df['travel_days'] * df['srch_rm_cnt']

	# df['people_type'] = df[['srch_children_cnt','srch_adults_cnt']].apply(people_type, axis=1)
	# # df.drop(['srch_children_cnt','srch_adults_cnt'], axis=1, inplace=True)

	# df['trip_type'] = df[['plan_days','travel_days']].apply(trip_type, axis=1)
	# # df.drop(['plan_days','travel_days'], axis=1, inplace=True)

	# df['foreign'] = (df['user_location_country'] != df['hotel_country']).apply(int)
	# df['diff_conti'] = (df['posa_continent'] != df['hotel_continent']).apply(int)

	df1, df2 = df[df['is_booking']==1], df[df['is_booking']==0]
	# df1.drop('is_booking', axis=1, inplace=True)
	# df2.drop(['is_booking','plan_days','travel_days','travel_month','trip_type'], axis=1, inplace=True)

	hotel_prob(df1, './data/hotel_destinations1')
	hotel_prob(df2, './data/hotel_destinations2')
	
	return df1, df2

def main():
	train1, train2 = clean_data(load_data())
	# train1.to_csv('./data/train1_booking.csv')
	# train2.to_csv('./data/train1_unbooking.csv')

if __name__ == '__main__':
	main()
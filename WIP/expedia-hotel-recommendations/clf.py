import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import importHelper
cv = importHelper.load("cvMachine")

def main():
	df1 = pd.read_csv('./data/train1_booking.csv', index_col=0)
	# df2 = pd.read_csv('./data/train1_unbooking.csv', index_col=0)

	X1, y1 = df1[[e for e in df1.columns if e != 'hotel_cluster']], df1['hotel_cluster']
	# X2, y2 = df2[[e for e in df2.columns if e != 'hotel_cluster']], df2['hotel_cluster']
	rf = RandomForestClassifier()
	print cv.sklearn_cross_validation(rf, X1, y1)

	# print cv.sklearn_cross_validation(rf, X2, y2)

if __name__ == '__main__':
	main()
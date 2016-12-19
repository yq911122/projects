from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

from pyspark.sql.functions import col, udf, explode

from pyspark.sql.types import *

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

import importHelper

# lm = importHelper.load('lm')
# ev = importHelper.load('everything')

c = 0.2

PATH = '/media/quan/Q/github/expedia-hotel-recommendations/data/'

def load_data(sqlContext,url):
	df = sqlContext.read.load(url,
							format="com.databricks.spark.csv",
							header='true',
							inferSchema='true')
	# print df.dtypes
	return df

def mydist(x, y):
	return int(x[0] == y[0]) + c * (x[1] - y[1])**2	

knn_neighbors = 4
def fill_with_neighbors(df, ref):
	def renames(data, old_columns, new_columns):
		data = reduce(lambda d, idx: d.withColumnRenamed(old_columns[idx], new_columns[idx]), xrange(len(old_columns)), data)
		return data

	def get_neighbors_in_country(attrs):
		attrs = list(attrs)
		attrs[0], attrs[1] = list(attrs[0]), list(attrs[1])
		if not attrs[1]:
			return []
		if not attrs[0]:
			attrs[0] = [[-1, 0, 0]]
		exist_hotels, new_hotels = np.array(attrs[0], ndmin=2), np.array(attrs[1], ndmin=2)

		X, y = exist_hotels[:,1:], exist_hotels[:,0]
		X2, y2 = new_hotels[:,1:], new_hotels[:,0]

		# print "X:"
		# print X.shape

		# print "y:"
		# print y.shape

		# print "X2:"
		# print X2.shape

		# print "y2:"
		# print y2.shape

		if X.shape[0] <= knn_neighbors:
			near_dest = np.array(np.tile(y, (y2.shape[0], 1)), ndmin=2)
			exist_dests = near_dest.reshape((1, near_dest.shape[0]*near_dest.shape[1]))
		else:
			knnParams = {'n_neighbors': knn_neighbors, 
				 'algorithm': 'ball_tree',
				 'metric': 'pyfunc',
				 'func': mydist}
			knn = KNeighborsClassifier(**knnParams)
			knn.fit(X, y)
			near_dest = knn.kneighbors(X2, return_distance=False)
			exist_dests = near_dest.reshape((1, near_dest.shape[0]*near_dest.shape[1]))
			exist_dests[0] = [y[e] for e in exist_dests[0]]

		print "near_dest:"
		print near_dest.shape	
		
		y2_repeated = np.repeat(y2, near_dest.shape[1])

		print "exist_dests:"
		print exist_dests.shape

		print "y2_repeated"
		print y2_repeated.shape

		result = np.vstack([exist_dests, y2_repeated]).T.tolist()

		return result

	d_cols = [e for e in df.columns if e != 'srch_destination_id']
	# x_col = ['srch_destination_type_id','is_package']
	# y_col = 'srch_destination_id'
	# data_col = x_col + [y_col]

	ref_exist = ref.join(df, ref.srch_destination_id == df.srch_destination_id) \
				   .drop(df.srch_destination_id) \
				   .select(ref.columns)
	ref_new = ref.subtract(ref_exist)

	ref_exist = ref_exist.rdd.map(list).map(lambda x: (x[-1], x[:-1]))
	ref_new = ref_new.rdd.map(list).map(lambda x: (x[-1], x[:-1]))
	
	neighbors = ref_exist.cogroup(ref_new) \
						 .mapValues(get_neighbors_in_country) \
						 .values() \
						 .filter(lambda x: len(x) > 0) \
						 .flatMap(lambda x: x) \
						 .toDF(['a', 'b'])

	neighbors = neighbors.select(neighbors.a.cast(IntegerType()).alias('exist_destination_id'), 
								neighbors.b.cast(IntegerType()).alias('new_destination_id'))					 

	dests = neighbors.join(df, neighbors.exist_destination_id == df.srch_destination_id) \
					 .groupby('new_destination_id') \
					 .mean(*d_cols)
	new = renames(dests, dests.schema.names, ['srch_destination_id']+d_cols)
	return new

def main():
	conf = SparkConf().setAppName("dm").setMaster("local")
	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)

	df = load_data(sqlContext,PATH+'destinations.csv')
	ref = load_data(sqlContext,PATH+'hotel_destinations_all1.csv')
	
	new  = fill_with_neighbors(df, ref)
	if not new.rdd.isEmpty():
		df = df.unionAll(new)
	else:
		print "no new hotels"
	df.write.format('com.databricks.spark.csv') \
				.mode('overwrite') \
				.options(header="true") \
				.save(PATH+'destinations_processed_all.csv')
	

if __name__ == '__main__':
	main()
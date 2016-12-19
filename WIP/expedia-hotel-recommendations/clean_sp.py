from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf, datediff, month, hour, first
# from pyspark.sql.DataFrame import fillna
from pyspark.sql.types import IntegerType, StructType

from pyspark.sql.functions import col
import pyspark.sql.functions as func

import pandas as pd
import numpy as np
import importHelper


ent = importHelper.load("entropy")

PATH = '/media/quan/Q/github/expedia-hotel-recommendations/data/'

def load_data(sqlContext,url):
	df = sqlContext.read.load(url,
							format="com.databricks.spark.csv",
							header='true',
							inferSchema='true')

	df = df.withColumnRenamed('', 'id')
	return df

# def convertColumn(df, name, new_type):
# 	df1 = df.withColumnRenamed(name, "swap")
# 	return df1.withColumn(name, df_1.col("swap").cast(new_type)).drop("swap")

def people_type(child, adult):
	"""
	:rtype: 1: single adult: child = 0, adult = 1
			2: couples: child = 0, adult = 2
			3: families: child > 0, adult > 0
			4: friends: child = 0, adult > 2
			5: others
	"""
	if child == 0 and adult == 1: return 1
	if child == 0 and adult == 2: return 2
	if child > 0 and adult > 0: return 3
	if child == 0 and adult > 2: return 4
	return 5

def trip_type(plan, travel):
	"""
	:rtype: 1: one day immediate trip: plan < 2, travel = 1
			2: multiple days immediate trip: plan < 2, travel > 1
			3: others
	"""
	if plan < 2 and travel == 1: return 1
	if plan < 2 and travel > 1: return 2
	return 3

def hotel_prob(df, path):
	exprs = [func.first('srch_destination_type_id'), func.sum('is_package'), func.count('is_package'), func.first('hotel_country')]
	grouped = df.select(['srch_destination_type_id','is_package','hotel_country', 'srch_destination_id']).groupBy(df.srch_destination_id).agg(*exprs)

	grouped = grouped.withColumn('is_package', grouped['sum(is_package)']/grouped['count(is_package)'])
	grouped = grouped.select(col('srch_destination_id'), col('is_package'), col('first(srch_destination_type_id)()').alias('srch_destination_type_id'), col('first(hotel_country)()').alias('hotel_country'))

	grouped.write.format('com.databricks.spark.csv') \
				 .mode('overwrite') \
				 .options(header="true") \
				 .save(path)


def clean_data(df):
	get_people_type = udf(people_type, IntegerType())
	get_trip_type = udf(trip_type, IntegerType())
	compare = udf(lambda x, y: int(x != y), IntegerType())
	greater_than_zero =  udf(lambda x: max(x, 0))

	df = df.na.fill({'orig_destination_distance':0})

	df = df.withColumn('people_type', get_people_type(df.srch_children_cnt, df.srch_adults_cnt))
	# df.drop(['srch_children_cnt','srch_adults_cnt'], axis=1, inplace=True)

	# df.drop(['plan_days','travel_days'], axis=1, inplace=True)
	df = df.withColumn('foreign', compare(df.user_location_country, df.hotel_country))
	df = df.withColumn('diff_conti', compare(df.posa_continent, df.hotel_continent))
	df = df.withColumn('plan_hour', hour(df.date_time))


	df1 = df.filter(df.is_booking == 1)

	df1 = df1.na.drop()

	df1 = df1.withColumn('plan_days', greater_than_zero(datediff(df1.srch_ci, df1.date_time)))
	df1 = df1.withColumn('travel_days', greater_than_zero(datediff(df1.srch_co, df1.srch_ci)))
	df1 = df1.withColumn('travel_month', month(df1.srch_ci))
	df1 = df1.withColumn('price_multiplier', df1.travel_days*df1.srch_rm_cnt)
	df1 = df1.withColumn('trip_type', get_trip_type(df1.plan_days, df1.travel_days))

	df1 = df1.select([c for c in df1.columns if c not in ['is_booking', 'srch_ci','srch_co','date_time']])

	df2 = df.filter(df.is_booking == 0)
	df2 = df2.select([c for c in df2.columns if c not in ['is_booking','srch_ci','srch_co','date_time']])
	df2 = df2.na.drop()

	hotel_prob(df1, PATH+'hotel_destinations_all1.csv')
	hotel_prob(df2, PATH+'hotel_destinations_all2.csv')

	return df1, df2


def main():
	conf = SparkConf().setAppName("clean").setMaster("local")
	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)
	train = load_data(sqlContext,PATH+'train.csv')
	train1, train2 = clean_data(train)

	train1.write.format('com.databricks.spark.csv') \
				.mode('overwrite') \
				.options(header="true") \
				.save(PATH+'train_sp_booking.csv')
	train2.write.format('com.databricks.spark.csv') \
				.mode('overwrite') \
				.options(header="true") \
				.save(PATH+'train_sp_unbooking.csv')

if __name__ == '__main__':
	main()
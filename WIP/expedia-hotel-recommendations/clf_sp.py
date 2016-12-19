from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

from pyspark.ml.feature import StringIndexer

from pyspark.sql.functions import udf

import pandas as pd

import importHelper
cv = importHelper.load("cvMachine")

PATH = '/media/quan/Q/github/expedia-hotel-recommendations/data/'

def load_data(sqlContext,url):
	df = sqlContext.read.load(url,
							format="com.databricks.spark.csv",
							header='true',
							inferSchema='true')
	print df.count()
	print df.dtypes

	return df

def test():
	conf = SparkConf().setAppName("clean").setMaster("local")
	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)

	df = load_data(PATH+'/data/train1_booking.csv')
	assembler = VectorAssembler(inputCols=[x for x in df.columns if x != 'hotel_cluster'], outputCol='features')
	assembler.transform(df)

	df = df.withColumnRenamed('hotel_cluster', 'label')
	df = df.withColumn('features', vectorize(df[features]))
		   .select(['features', 'label'])

	stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
	si_model = stringIndexer.fit(df)
	td = si_model.transform(df)

	rf = RandomForestClassifier()
	pipeline = Pipeline(stages=[assembler, rf])

	model  = pipeline.fit(td)


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
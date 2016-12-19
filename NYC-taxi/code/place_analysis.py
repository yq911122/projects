import csv
from pyspark import SparkContext
import numpy

def isEvening(s):
# s type: '2013-01-01 00:19:00'
	t = int(s[11:13])
	if 17 <= t < 24:
		return True
	else:
		return False


sc = SparkContext(appName="PolyCt")

input_file_fare = sc.textFile("hdfs:///user/yuanq/trip_fare_1.csv")
header = input_file_fare.take(1)[0]
fare_data = input_file_fare.filter(lambda line: line != header)\
					.map(lambda line: line.split(','))

input_file_trip = sc.textFile("hdfs:///user/yuanq/trip_data_1.csv")
header = input_file_trip.take(1)[0]
trip_data = input_file_trip.filter(lambda line: line != header)\
					.map(lambda line: line.split(','))


fare = fare_data.map(lambda x: [x[1],[x[4],x[8]]])
trip = trip_data.filter(lambda x: isEvening(x[6])) \
				.map(lambda x: [x[1],x[13] + ',' + x[12]])

# fare.saveAsTextFile('hdfs:///user/yuanq/fare_place')
# trip.saveAsTextFile('hdfs:///user/yuanq/trip_place')

fractions = {u'CRD':0.001,u'NOC':0.001,u'CSH':0.001,u'DIS':0.001,u'UNK':0.001}

data = fare.join(trip).map(lambda x: [x[1][0][0],[float(x[1][0][1]),x[1][1]]]).sampleByKey(False, fractions, 2)

data.saveAsTextFile('hdfs:///user/yuanq/final_place')


input_file = sc.textFile("hdfs:///user/yuanq/final_place/part-000[00-21]*")

fractions = {'CSH':0.001,"CRD":0.001}
data = input_file.map(lambda line: line[2:-3].split(',')) \
				.map(lambda x: [x[0][1:-1], [float(x[1][2:]),','.join([x[2][3:],x[3]])]])\
				.filter(lambda x: x[0] == u'CSH' or x[0] == u'CRD') \
				.sampleByKey(False,fractions,2).collect()
# data.saveAsTextFile('hdfs:///user/yuanq/test')

with open('places_samples.csv', 'wb') as output:
	writer = csv.writer(output, delimiter = ',')
	writer.writerows(data)
	output.close()
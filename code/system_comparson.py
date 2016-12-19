import csv, itertools
from pyspark import SparkContext

sc = SparkContext(appName="PolyCt")

trip_data = sc.textFile("hdfs:///user/yuanq/pick_dis2/part-000[00-13]*")
trip_data = trip_data.map(lambda l: l[2:-1].split(','))

fare_data = sc.textFile("hdfs:///user/yuanq/trip_fare_1.csv")
header = fare_data.take(1)[0]
fare_data = fare_data.filter(lambda line: line != header).map(lambda line: line.split(','))


#different systems market share
sys = trip_data.map(lambda x: [x[1][2:-1],1]).countByKey()
print sys

sys_type = fare_data.map(lambda x:[(x[2],x[4]),1]) \
					.countByKey()
print sys_type

# tip amount vs. payment type
tip_type_total = fare_data.map(lambda x: [x[4], x[8]]) \
					.reduceByKey(lambda a,b: float(a) + float(b)).collect()
tip_tpye_ct = fare_data.map(lambda x:[x[4],x[8]]) \
						.countByKey()
print tip_type_total
print tip_tpye_ct

# tip amount vs. total fares
tip_total = fare_data.map(lambda x: [float(x[8]), float(x[10])]) \
						.filter(lambda x: x[0] < 100.0)

tip_total_tp = tip_total.reduceByKey(lambda a,b: a+b).collect()
tip_total_ct = tip_total.countByKey()

tip_total_tp = dict(tip_total_tp)

tip_total = {}
for k in tip_total_tp:
	tip_total[k] = tip_total_tp[k] / float(tip_total_ct[k])

print tip_total

with open('tip_total.csv','wb') as output:
	csv_writer = csv.writer(output,delimiter=',')
	csv_writer.writerow(tip_total.keys())
	csv_writer.writerow(tip_total.values())
	output.close()
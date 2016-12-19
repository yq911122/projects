import csv
from pyspark import SparkContext
# import os
# import tempfile
# os.environ['MPLCONFIGDIR'] = '/home2/yuanq'
# from matplotlib.path import Path


def cleanloc(s):
	l = s[1:-2].split(',')

	return (float(l[0]), float(l[1]))

def point_in_poly(pt,poly):

	n = len(poly)
	inside = False
	x = pt[0]
	y = pt[1]

	p1x,p1y = poly[0]
	for i in range(n+1):
		p2x,p2y = poly[i % n]
		if y > min(p1y,p2y):
			if y <= max(p1y,p2y):
				if x <= max(p1x,p2x):
					if p1y != p2y:
						xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
					if p1x == p2x or x <= xints:
						inside = not inside
		p1x,p1y = p2x,p2y
	return int(inside)

def tofloat(x):
	if x != '' or x != ' ':
		return float(x)
	else:
		return 0.0

def totime(x):
	return x[11:13]

with open('polypath_dis.csv', 'rU') as polyinput:
	raw_data = csv.reader(polyinput, delimiter = ',')
	polypaths = list(raw_data)
	i = 0
	j = 0
	for i in range(len(polypaths)):
		for j in range(len(polypaths[i])):
			polypaths[i][j] = cleanloc(polypaths[i][j])
	polyinput.close()


sc = SparkContext(appName="PolyCt")

input_file = sc.textFile("trip_data_1.csv")
header = input_file.take(1)[0]
polycounts = input_file.filter(lambda line: line != header) \
						.map(lambda line: line.split(',')) \
						.map(lambda x: [x[1],x[2],totime(x[5]), tofloat(x[11]),tofloat(x[10])]) \
						.map(lambda x: [x[0],x[1],x[2]] + [point_in_poly([x[3],x[4]],polypaths[i]) for i in range(len(polypaths))])
# polycounts.saveAsTextFile('hdfs:///user/yuanq/pick_dis2')
polycounts = polycounts.reduce(lambda a, b: [x + y for x, y in zip(a, b)])
print polycounts



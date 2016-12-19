import csv
import matplotlib.pyplot as plt

with open('tip_total.csv','rU') as inputfile:
	csv_reader = csv.reader(inputfile,delimiter=',')
	tip = csv_reader.next()
	fare = csv_reader.next()
	inputfile.close()

tip = [float(x) for x in tip]
fare = [float(x) for x in fare]

tip_fare = dict(zip(tip,fare))
tip_fare_final = {}
for k in tip_fare:
	if k <= 100:
		tip_fare_final[k] = tip_fare[k]
# print tip_fare_final

print tip_fare_final.keys()
print tip_fare_final.values()
# plt.scatter(tip,fare)
# plt.show()
# print tip
# print fare
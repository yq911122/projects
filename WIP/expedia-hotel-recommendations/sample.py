import numpy as np
import pandas as pd


trainUrl = './data/train1.csv'

def main():
	i, j = 0, 0
	trains = []
	chunksize = 10 ** 5
	for chunk in pd.read_csv(trainUrl, chunksize=chunksize, index_col=0):
		i += 1
		print i
		trains.append(chunk.sample(frac=0.05))
		if i == 100:
			train = pd.concat(trains)
			i = 0
			trains = []
			if j == 0:
				train.to_csv('./data/train1_sp.csv')
				j += 1
			else:
				train.to_csv('./data/train1_sp.csv', mode='a', header=False)
				
	train = pd.concat(trains)
	print train.head()
	if i < 100:
		train.to_csv('./data/train1_sp.csv', header=True)
	else:
		train.to_csv('./data/train1_sp.csv', mode='a', header=False)

	
if __name__ == '__main__':
	main()
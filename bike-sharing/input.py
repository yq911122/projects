import pandas as pd

TRAINURL = './input/train.csv'
TESTURL = './input/test.csv'

def load_data(url):
	return pd.read_csv(url, parse_dates=[0])

def clean_data(df):
	df.loc[:,'hour'] = df['datetime'].map(lambda x: x.hour)
	df.loc[:,'dayofweek'] = df['datetime'].map(lambda x: x.dayofweek)
	df.loc[:,'month'] = df['datetime'].map(lambda x: x.month)
	df.loc[:,'year'] = df['datetime'].map(lambda x: x.year)
	df = df.drop(['datetime'], axis=1)
	return df

def main():
	train = clean_data(load_data(TRAINURL))
	test = clean_data(load_data(TESTURL))

	train.to_csv('./input/train_processed.csv', index=False)
	test.to_csv('./input/test_processed.csv', index=False)

if __name__ == '__main__':
	main()



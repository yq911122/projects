import pandas as pd

from sklearn.utils import shuffle

df = pd.read_csv('train.csv', header=0, index_col=0)
df['target'] = df['target'].map(lambda x: int(x[-1])) - 1
df = shuffle(df)
df.to_csv('train_prep.csv', index_label='id')
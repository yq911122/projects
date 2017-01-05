import theano
import theano.tensor as T
import numpy as np
import pandas as pd

from sklearn.preprocessing import scale

from theano_dl.dbn import DBN

files = ['loglinear1.stack.csv', 
        'loglinear2.stack.csv', 
        'xgb1.stack.csv',
        'xgb2.stack.csv',
        'extratrees.stack.csv',
        'svc.stack.csv',
        'knn1.stack.csv',
        'knn2.stack.csv',
        'knn3.stack.csv',
        'knn4.stack.csv']

def load_X(paths):
    dfs = []
    for p in paths:
        df = pd.read_csv(p, index_col=0)
        dfs.append(df)
    X = pd.concat(dfs, axis=1).values
    return scale(X, axis=0)

def load_y(path):
    y = pd.read_csv(url, header=0, index_col=0, usecols=['target'])
    return y.values.astype(np.int32)

def save_prediction(path, proba):
    columns = ['Class_'+str(i) for i in xrange(1, 10)]
    df = pd.DataFrame(proba, columns=columns)
    df.index.rename('id')
    df.index += 1
    df.to_csv(path, index_label='id')

train_paths = ['./train/'+file for file in files]
test_paths = ['./test/'+file for file in files]
y_path = 'train_prep.csv'

train_X = load_X(train_paths)
train_y = load_y(y_path)
test_X = load_X(test_paths)

train_X = theano.shared(
            np.asarray(train_X, dtype=theano.config.floatX),
            borrow=True)

test_X = theano.shared(
            np.asarray(test_X, dtype=theano.config.floatX),
            borrow=True)

train_y = theano.shared(
            np.asarray(train_y, dtype=theano.config.floatX),
            borrow=True)

train_y = T.cast(train_y, 'int32')

n_in = 93 * len(files)
n_hiddens = [1000, 1000, 1000]
n_out = 9

model = DBN(
    n_in=n_in, 
    n_hiddens=n_hiddens, 
    n_out=n_out, 
    k=2, 
    batch_size=500, 
    learning_rate=0.01, 
    n_epoch=20, 
    n_epoch_prefit=5, 
    criterion=0.05, 
    penalty='l1', 
    alpha=0)

model.prefit(train_X)
model.fit(train_X, train_y)
proba = model.predict_proba(test_X)
save_prediction('result.csv', proba)
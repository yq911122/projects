import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, scale
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb

url1 = 'train_prep.csv'
url2 = 'test.csv'

def load_data(url, train=True):
    df = pd.read_csv(url, header=0, index_col=0)
    # df = df.sample(frac=0.01)
    if train:
        return df[[col for col in df.columns if col != 'target']].values, df['target'].values.astype(np.int32)
    return df.values

def save_prediction(path, proba):
    columns = ['Class_'+str(i) for i in xrange(1, 10)]
    df = pd.DataFrame(proba, columns=columns)
    df.index.rename('id')
    df.index += 1
    df.to_csv(path, index_label='id')

def train_xgb1(train_X, train_y, test_X):
    model = xgb.XGBClassifier(
        max_depth=15,
        learning_rate=0.1,
        objective="binary:logistic",
        subsample=0.9,
        colsample_bytree=0.7
        )

    model.fit(train_X, train_y, eval_metric="logloss")
    pred_y = model.predict_proba(test_X)
    save_prediction("./test/xgb1.stack.csv", pred_y)

    pred_y = model.predict_proba(train_X)
    save_prediction("./train/xgb1.stack.csv", pred_y)

def train_xgb2(train_X, train_y, test_X):

    def extend_dataset(X):
        X = X.astype(np.float32)
        
        log2FloorX = np.floor(np.log2(X + 1))
        X_feats = np.append(X, log2FloorX, axis = 1)

        logFloor = [3,4,5,6,7,8,9,12,13]
        for v in logFloor:
            val = np.floor(np.divide(np.log(X + 1),np.log(v)))
            X_feats = np.append(X_feats, val, axis = 1)
        
        logExpFloorX = np.floor(np.log(X+1))
        X_feats = np.append(X_feats, logExpFloorX, axis = 1)
        
        sqrtFloorX = np.floor(np.sqrt(X+1))
        X_feats = np.append(X_feats, sqrtFloorX, axis = 1)
        
        powX = np.power(X+1,2)
        X_feats = np.append(X_feats, powX, axis = 1)

        return X_feats

    train_set_X = train_X.copy()
    test_set_X = test_X.copy()

    train_set_X = extend_dataset(train_set_X)
    test_set_X = extend_dataset(test_set_X)

    train_x = xgb.DMatrix( train_set_X, missing = -999.0)
    train = xgb.DMatrix( train_set_X, label=train_y, missing = -999.0)
    test = xgb.DMatrix( test_set_X, missing = -999.0 )

    param = {}
    param['objective'] = 'multi:softprob'
    param['eval_metric'] = 'mlogloss'
    param['eta'] = 0.05
    param['silent'] = 1
    param['num_class'] = 9
    param['nthread'] = 6
    param['max_depth'] = 50
    param['min_child_weight'] = 5
    param['colsample_bylevel'] = 0.012        
    param['colsample_bytree'] = 1.0

    plst = list(param.items())

    bst = xgb.train( plst, train, 500 )

    pred_y = bst.predict_proba(test)
    save_prediction("./test/xgb2.stack.csv", pred_y)

    pred_y = model.predict_proba(train_x)
    save_prediction("./train/xgb2.stack.csv", pred_y)

def train_linear1(train_X, train_y, test_X):
    def prep(X):
        means = np.array([[1.0] if np.mean(row)==0.0 else [np.mean(row)] for row in X])
        X_feats = np.divide(X, means)
        X_feats = np.append(X, X_feats, axis = 1)
        return scale(X_feats, axis=0)

    train_set_X = prep(train_X)
    test_set_X = prep(test_X)

    model = LogisticRegression(penalty='l1', C=0.01)
    model.fit(train_set_X, train_y)

    pred_y = model.predict_proba(test_set_X)
    save_prediction("./test/loglinear1.stack.csv", pred_y)

    pred_y = model.predict_proba(train_set_X)
    save_prediction("./train/loglinear1.stack.csv", pred_y)

def train_linear2(train_X, train_y, test_X):

    def prep(X):
        means = np.array([[1.0] if np.mean(row)==0.0 else [np.mean(row)] for row in X])
        X_feats = np.divide(X, means)
        return scale(X_feats, axis=0)

    train_set_X = prep(train_X)
    test_set_X = prep(test_X)

    model = LogisticRegression(penalty='l2', C=0.01)
    model.fit(train_set_X, train_y)

    pred_y = model.predict_proba(test_set_X)
    save_prediction("./test/loglinear2.stack.csv", pred_y)

    pred_y = model.predict_proba(train_set_X)
    save_prediction("./train/loglinear2.stack.csv", pred_y)

def train_ensemble(train_X, train_y, test_X):

    def to_tfidf(X):
        X = X.astype(np.float32)
        tfidf = TfidfTransformer()
        X = tfidf.fit_transform(X).toarray()

        return X

    train_set_X = train_X.copy()
    test_set_X = test_X.copy()

    train_set_X = to_tfidf(train_set_X)
    test_set_X = to_tfidf(test_X)

    model = ExtraTreesClassifier(
        n_estimators=300,
        criterion="entropy",
        max_features=30,
        max_depth=25)

    model.fit(train_set_X, train_y)

    pred_y = model.predict_proba(test_set_X)
    save_prediction("./test/extratrees.stack.csv", pred_y)

    pred_y = model.predict_proba(train_set_X)
    save_prediction("./train/extratrees.stack.csv", pred_y)

def train_svm(train_X, train_y, test_X):

    model = SVC(
        C=10,
        gamma=0.01,
        probability=True)

    model.fit(train_X, train_y)

    pred_y = model.predict_proba(test_X)
    save_prediction("./test/svc.stack.csv", pred_y)

    pred_y = model.predict_proba(train_X)
    save_prediction("./train/svc.stack.csv", pred_y)

def train_knn1(train_X, train_y, test_X):
    
    def prep(X):
        X = X.astype(np.float32)
        X[X > 10] = 10.
        return scale(X, axis=0)

    train_set_X = train_X.copy()
    test_set_X = test_X.copy()

    train_set_X = prep(train_set_X)
    test_set_X = prep(test_set_X)

    model = KNeighborsClassifier(
        n_neighbors=20,
        weights="distance",
        p=2)
    model.fit(train_set_X, train_y)

    pred_y = model.predict_proba(test_set_X)
    save_prediction("./test/knn1.stack.csv", pred_y)

    pred_y = model.predict_proba(train_set_X)
    save_prediction("./train/knn1.stack.csv", pred_y)

def train_knn2(train_X, train_y, test_X):
    
    def prep(X):
        X = X.astype(np.float32)
        tfidf = TfidfTransformer()
        X = tfidf.fit_transform(X).toarray()
        return scale(X, axis=0)

    train_set_X = train_X.copy()
    test_set_X = test_X.copy()

    train_set_X = prep(train_set_X)
    test_set_X = prep(test_set_X)

    model = KNeighborsClassifier(
        n_neighbors=20,
        weights="distance",
        p=2)
    model.fit(train_set_X, train_y)

    pred_y = model.predict_proba(test_set_X)
    save_prediction("./test/knn2.stack.csv", pred_y)

    pred_y = model.predict_proba(train_set_X)
    save_prediction("./train/knn2.stack.csv", pred_y)

def train_knn3(train_X, train_y, test_X):
    
    def prep(X):
        X = X.astype(np.float32)
        means = np.array([[1.0] if np.mean(row)==0.0 else [np.mean(row)] for row in X])
        X_feats = np.divide(X, means)
        X_feats2 = np.floor(np.log10(X + 1.))
        X = np.append(X_feats, X_feats2, axis = 1)
        return scale(X, axis=0)

    train_set_X = train_X.copy()
    test_set_X = test_X.copy()

    train_set_X = prep(train_set_X)
    test_set_X = prep(test_set_X)

    model = KNeighborsClassifier(
        n_neighbors=20,
        weights="distance",
        p=2)
    model.fit(train_set_X, train_y)

    pred_y = model.predict_proba(test_set_X)
    save_prediction("./test/knn3.stack.csv", pred_y)

    pred_y = model.predict_proba(train_set_X)
    save_prediction("./train/knn3.stack.csv", pred_y)

def train_knn4(train_X, train_y, test_X):
    
    def prep(X):
        X = X.astype(np.float32)
        X = 2.*np.sqrt(X + (3./8.))
        return X

    train_set_X = train_X.copy()
    test_set_X = test_X.copy()

    train_set_X = prep(train_set_X)
    test_set_X = prep(test_set_X)

    model = KNeighborsClassifier(
        n_neighbors=20,
        weights="distance",
        p=2)
    model.fit(train_set_X, train_y)

    pred_y = model.predict_proba(test_set_X)
    save_prediction("./test/knn4.stack.csv", pred_y)

    pred_y = model.predict_proba(train_set_X)
    save_prediction("./train/knn4.stack.csv", pred_y)

train_X, train_y = load_data(url1)
test_X = load_data(url2, False)

# train_linear1(train_X, train_y, test_X)
# train_linear2(train_X, train_y, test_X)
# train_xgb1(train_X, train_y, test_X)
# train_ensemble(train_X, train_y, test_X)
# train_svm(train_X, train_y, test_X)
train_knn1(train_X, train_y, test_X)
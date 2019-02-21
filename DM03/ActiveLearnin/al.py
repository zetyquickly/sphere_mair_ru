import argparse
import commands
import numpy as np

from math import sqrt
from datetime import datetime
from active import BoostingActive 
import subprocess as sp
import multiprocessing as mp
import pickle

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import os

__author__ = 'g.kasparyants'
PREFIT = 100
PROBA = 0.1
POOL = 100

def predict_submission(model, pathToData):
    X_test = np.genfromtxt(pathToData, usecols=(1,2,3,4,5,6,7,8,9,10), skip_header=True, dtype=np.float32, delimiter=',')
    y_pred = model.predict(X_test)
    ids    = np.arange(1, y_pred.shape[0] + 1, dtype=np.int64)
    np.savetxt('submission-' + str(datetime.now()) + '.csv', np.stack([ids, y_pred], axis=-1), fmt='%d,%f', header='Id,Target', comments='')

def make_test(N):
    X = 7 * np.random.rand(N, 10) + .01
    X_new, y_new = batch_oracle(X)
    if os.path.exists('test.pkl'):
        with open('test.pkl', 'r') as f:
            X, y = pickle.load(f)
        X = np.concatenate([X, X_new], axis=0)
        y = np.concatenate([y, y_new], axis=0)
    else:
        X = X_new
        y = y_new
    with open('test.pkl', 'w') as f:
        pickle.dump((X, y), f)

def make_train(N):
    X = 7 * np.random.rand(N, 10) + .01
    return batch_oracle(X)


def main(pathToData):
    loss = lambda y_true, y_pred : sqrt(mean_squared_error(y_true, y_pred))
    model = ActiveTree()

    X_unlabeled = np.genfromtxt(pathToData, usecols=(1,2,3,4,5,6,7,8,9,10), skip_header=True, dtype=np.float32, delimiter=',')
    mask = (np.random.rand(X_unlabeled.shape[0]) > 0.99999)

    X_train, y_train = batch_oracle(X_unlabeled[mask])
    X_unlabeled = X_unlabeled[~mask, :]

    X_test, y_test = batch_oracle(X_unlabeled[np.random.rand(X_unlabeled.shape[0]) > 0.99999])

    model.fit(X_train, y_train, X_unlabeled)

    while(True):
        try:
            X_sample = model.wantedToKnow()
            print("wanted to know", X_sample.shape[0])

            X_train, y_train = batch_oracle(X_sample)
            model.partial_fit(X_train, y_train)

            y_pred = model.predict(X_test)
            print("loss at %d is %f" % (t, loss(y_test, y_pred)))
        except KeyboardInterrupt:
            break

    model.partial_fit(X_test, y_test)
    model.save()
    predict_submission(model, pathToData)


def batch_oracle(X):
    if not hasattr(batch_oracle, 'pool'):
        batch_oracle.pool = mp.Pool(POOL)
    data = list(X)
    data = batch_oracle.pool.map(oracle, data)
    data = filter(lambda x:x is not None, data)
    if len(data) == 0 or np.random.rand() > PROBA:
        data = 7 * np.random.rand(24, 10) + .01
        return batch_oracle(data)
    data = np.stack(data)
    return data[:,:-1], data[:,-1]


def oracle(x):
    x = list(x)
    query = "./Oracle.static " + ' '.join(map(lambda y:str(y), x))
    if query.strip() == 'Function is undefined at this point!' or query.strip() == 'inf':
        return None
    return np.array(x + [float(commands.getoutput(query))])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    main(args.path)

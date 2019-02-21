from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
from collections import namedtuple, Counter
from itertools import groupby
from sklearn.externals import joblib
from collections import Counter
from functools import reduce
from scipy.optimize import fmin
import sklearn
import sklearn.cluster

def preprocess(X):
    return np.concatenate([
        X,
        np.exp(X),
        np.log(X),
        np.sin(X),
        np.sqrt(X),
        np.square(X),
        np.cbrt(X)
    ], axis=1)


class ActiveTree(BaseEstimator, RegressorMixin):
    def __init__(self, lr=1e-1, k=1000, max_depth=10, percentile=0.75, sample_num_near=10, sample_num_random=10, cluster_num=10, sample_range=1.0):
        self.lr = lr
        self.k = k
        self.max_depth = max_depth
        self.percentile = percentile
        self.sample_num_near = sample_num_near
        self.sample_num_random = sample_num_random
        self.cluster_num = cluster_num
        self.sample_range = sample_range

    def fit(self, X, y, U):
        y = np.log(np.clip(y, 1e-10, None))
        self.model = DecisionTreeRegressor(max_depth=self.max_depth)
        self.B = RandomForestRegressor(n_estimators=1000)
        self.X, self.y = X, y

        self.model.fit(self.X, self.y)
        self.B.fit(preprocess(self.X), self.y)
        self.U = U
        return self

    def partial_fit(self, X, y):
        y = np.log(np.clip(y, 1e-10, None))
        self.X = np.concatenate([self.X, X], axis=0)
        self.y = np.concatenate([self.y, y], axis=0)

        self.max_depth += 10
        self.model = DecisionTreeRegressor(max_depth=self.max_depth)
        
        self.model.fit(self.X, self.y)

    def wantedToKnow(self):
        # max values
        inds = self.model.apply(self.U)
        values = self.model.tree_.value[inds, 0, 0]
        th = np.percentile(values, int(100 * self.percentile))
        top = np.where(values > th)[0]
        i = top[np.random.randint(0, top.shape[0], size=self.sample_num_near)]
        X_sample = self.sampleNear(self.U[i])
        np.delete(self.U, i, axis=0)

        # random values
        X_sample = np.concatenate([X_sample, 10 * np.random.rand(self.sample_num_random, 10)], axis=0)

        # maximum bootstrap loss values
        y_pred = self.predict(self.U)
        y_distr = self.B.predict(preprocess(self.U))
        w = (y_pred - y_distr)**2
        ind = np.argsort(w)[-min(w.shape[0], self.k):]
        X_sample = np.concatenate([X_sample, self.U[ind]], axis=0)
        np.delete(self.U, ind, axis=0)

        # scipy optimizers
        def tmp(x):
            X = self.sampleNear([x])
            return np.mean(self.predict(X))
        
        x0 = fmin(lambda x : -tmp(x), self.U[np.random.randint(0, self.U.shape[0])])
        X_sample = np.concatenate([X_sample, self.sampleNear([x0])], axis=0)
        
        # clusters
        cluster = sklearn.cluster.KMeans(n_clusters=self.cluster_num)
        centers = cluster.fit(self.U).cluster_centers_ 
        X_sample = np.concatenate([X_sample, self.sampleNear(centers)], axis=0)

        return X_sample

    def sampleNear(self, X):
        return np.concatenate([np.random.multivariate_normal(x, np.diag([self.sample_range] * 10), self.sample_num_near) for x in X], axis=0)   
    
    def predict(self, X):
        return np.exp(self.model.predict(X))

    def save(self):
        joblib.dump(self.model, 'model.pkl')
        
    def restore(self):
        self.model = joblib.load('model.pkl')


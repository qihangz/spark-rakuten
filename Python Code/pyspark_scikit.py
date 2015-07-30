from pyspark import SparkConf, SparkContext
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB

import pandas as pd
import numpy as np

import numpy as np
X = np.random.randint(2, size=(6, 100))
Y = np.array([1, 2, 3, 4, 4, 5])

clf.fit(X, Y)
print(clf.predict(X[2]))

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=12000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)

# or read from a file (for instance)
#df = pd.read_csv('data.csv', sep=' ', header=None)
#X = df[[1,2,3,4,5,6,7,8,9,10]].as_matrix()
#y = df[[0]][0].tolist()

# Partition data
def dataPart(X, y, start, stop): return dict(X=X[start:stop, :], y=y[start:stop])

def train(data):
    X = data[0]
    y = data[1]
    return BernoulliNB().fit(X,y)

# Merge 2 Models
from sklearn.base import copy
def merge(left,right):
    new = copy.deepcopy(left)
    new.estimators_ += right.estimators_
    new.n_estimators = len(new.estimators_)  
    return new

data = dict(X=X, y=y)

forest = sc.parallelize(data).map(train).reduce(merge)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

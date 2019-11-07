import pdb
import argparse
from enum import Enum

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description='decide on baseline model')
parser.add_argument('--model', default='SVM')
parser.add_argument('--try_hyperparams', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
CLF_TYPE = Enum('CLF', 'SVM K_SVM Logistic')
try:
    clf_type = CLF_TYPE[args.model]
except:
    print('invalid model parameter -- must be "SVM" "K_SVM" or "Logistic"')
    return

PATH = ''
X = np.load(PATH + 'X.npy', allow_pickle=True)
y = np.load(PATH + 'y.npy', allow_pickle=True)
X, y = shuffle(X, y)
X_train = X[:1200,:]
y_train = y[:1200]
#y_train[y_train==0] = -1
X_val = X[1200:1500,:]
y_val = y[1200:1500]
#y_val[y_val==0] = -1

hyperparams = [0.1, 1, 10, 100] if args.try_hyperparams else [1]
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

for C in hyperparams:
    if clf_type == CLF_TYPE.K_SVM:
        clf = SVC(C = C).fit(X_train, y_train)
    elif clf_type == CLF_TYPE.Logistic:
        clf = LogisticRegression(penalty='l1', C = C).fit(X_train, y_train)
    else:
        print('clf type not implemented')
        return
    accuracy = clf.score(X_val, y_val)
    print('Predictor Accuracy on validation set:', accuracy)
    print('----')

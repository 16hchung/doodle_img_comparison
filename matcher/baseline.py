import numpy as np
import pdb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

#PATH = '/Users/alexnam/cs221/doodle_img_comparison/221Project/'
PATH = ''
X = np.load(PATH + 'X.npy', allow_pickle=True)
#pdb.set_trace()
y = np.load(PATH + 'y.npy', allow_pickle=True)
X, y = shuffle(X, y)
import pdb;pdb.set_trace()
X_train = X[:1200,:]
y_train = y[:1200]
#y_train[y_train==0] = -1
X_val = X[1200:1500,:]
y_val = y[1200:1500]
#y_val[y_val==0] = -1

hyperparams = [0.1, 1, 10, 100]
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

for C in hyperparams:
    #clf = SVC(C = C).fit(X_train, y_train)
  #  y_pred = clf.predict(X_val)
    clf = LogisticRegression(penalty='l1', C = C).fit(X_train, y_train)
    accuracy = clf.score(X_val, y_val)
#  accuracy = svm.score(X_val, y_val)
#accuracy = sum(if y_pred[i] != y_val[i] for i in range(len(y_val))) / len(y_val)
    print('Predictor Accuracy on validation set:', accuracy)
    print('----')

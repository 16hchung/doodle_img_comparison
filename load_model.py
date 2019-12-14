from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pickle
import numpy as np
import sklearn

X = np.load('Downloads/vgg_moredata_X_test.npy', allow_pickle=True)
print("Finished loading X testing")
y = np.load('Downloads/vgg_moredata_y_test.npy', allow_pickle=True)
print("Finished loading y testing")

def enforce_svm_labels(y):
    y[y==0]=-1
    return y

def load_model():
    with open('Downloads/K_SVM_baseline.pkl', 'rb') as f:
        model = pickle.load(f)
        return model

def load_transformer():
    with open('Downloads/moredata_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        return scaler


#m = load_model()
#t = load_transformer()
#X = t.transform(X)
#print("Finished loading model")
#y_pred = m.predict(X)
y_pred = np.load('y_pred.npy', allow_pickle=True)
print("Finished predictions")
import pdb;pdb.set_trace()
y = enforce_svm_labels(y)
print('===Confusion Matrix===')
print(confusion_matrix(y, y_pred, labels=[-1,1]))

print('===F1 score===')
print(f1_score(y, y_pred, average='binary'))

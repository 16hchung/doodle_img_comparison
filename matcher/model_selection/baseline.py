import pdb
import argparse
from enum import Enum

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import pickle

from util.helper_functions import enforce_svm_labels, enforce_logistic_labels
from util.constants import *

def main():
    parser = argparse.ArgumentParser(description='decide on baseline model')
    parser.add_argument('--model', default='K_SVM')
    parser.add_argument('--try_hyperparams', default=True)
    parser.add_argument('--x_fname_prefix', default='X_')

    args = parser.parse_args()
    CLF_TYPE = Enum('CLF', 'SVM K_SVM Logistic')
    try:
        clf_type = CLF_TYPE[args.model]
    except:
        print('invalid model parameter -- must be "SVM" "K_SVM" or "Logistic"')
        return

    train_fname_suffix = 'train.npy'
    val_fname_suffix = 'val.npy'
    X_train = np.load(BITMAP_FEATURES + args.x_fname_prefix + train_fname_suffix, allow_pickle=True)
    y_train = np.load(BITMAP_FEATURES + 'y_' + train_fname_suffix, allow_pickle=True)
    X_train,y_train = shuffle(X_train,y_train)
    X_val = np.load(BITMAP_FEATURES + args.x_fname_prefix + val_fname_suffix, allow_pickle=True)
    y_val = np.load(BITMAP_FEATURES + 'y_' + val_fname_suffix, allow_pickle=True)
    X_val,y_val = shuffle(X_val,y_val)
    if clf_type == CLF_TYPE.K_SVM or clf_type == CLF_TYPE.SVM:
        y_train = enforce_svm_labels(y_train)
        y_val = enforce_svm_labels(y_val)
    else:
        y_train = enforce_logistic_labels(y_train)
        y_val = enforce_logistic_labels(y_val)


    hyperparams = [0.1, 1, 10, 100] if args.try_hyperparams else [1]
    # hyperparams = [1000,10000] if args.try_hyperparams else [1]
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    for C in hyperparams:
        if clf_type == CLF_TYPE.K_SVM:
            clf = SVC(C = C).fit(X_train, y_train)
        elif clf_type == CLF_TYPE.Logistic:
            clf = LogisticRegression(penalty='l1', C = C).fit(X_train, y_train)
        elif clf_type == CLF_TYPE.SVM:
            clf = LinearSVC(C=C).fit(X_train, y_train)
        else:
            print('clf type not implemented')
            return
        accuracy = clf.score(X_val, y_val)
        model_fname = MATCHING_OUTPUT_PATH + '{0:}_baseline_Cis{1:.2f}_accuris{2:.4f}.pkl'.format(clf_type.name, C, accuracy)
        with open(model_fname, 'wb') as fid:
            pickle.dump(clf, fid)
        print('Predictor Accuracy on validation set:', accuracy)
        print('----')

if __name__=='__main__':
    main()
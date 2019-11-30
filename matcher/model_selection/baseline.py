import pdb
import argparse
from enum import Enum
from tqdm import tqdm

import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import validation_curve
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import pickle

from util.helper_functions import enforce_svm_labels, enforce_logistic_labels
from util.constants import *

CLF_TYPE = Enum('CLF', 'SVM K_SVM Logistic')

def draw_validation_curve(X, y, clf_type):
    param_range = np.logspace(0, 2, 5)
    if str(clf_type) == str(CLF_TYPE.K_SVM):
        train_scores, test_scores = validation_curve(SVC(), X, y, param_name="C", verbose=2, param_range=param_range,cv=3, scoring="accuracy", n_jobs=2)
    elif str(clf_type) == str(CLF_TYPE.Logistic):
        train_scores, test_scores = validation_curve(LogisticRegression(), X, y, param_name="C", verbose=2,param_range=param_range,cv=5, scoring="accuracy", n_jobs=1)
    elif str(clf_type) == str(CLF_TYPE.SVM):
        train_scores, test_scores = validation_curve(LinearSVC(), X, y, param_name="C", verbose=2,param_range=param_range,cv=5, scoring="accuracy", n_jobs=1)
    else:
        print('clf type not implemented')
        return
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with " + str(clf_type))
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.2,color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2,color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

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
    # adhoc = '/Users/alexnam/doodle_img_comparison/'
    adhoc = ''
    X = np.load(adhoc + BITMAP_FEATURES + 'vgg_moredata_X_' + train_fname_suffix, allow_pickle=True)
    y = np.load(adhoc + BITMAP_FEATURES + 'vgg_moredata_y_' + train_fname_suffix, allow_pickle=True)
    print('finished loading data')
    X,y = shuffle(X,y)
    #X_val = np.load(BITMAP_FEATURES + args.x_fname_prefix + val_fname_suffix, allow_pickle=True)
    #y_val = np.load(BITMAP_FEATURES + 'y_' + val_fname_suffix, allow_pickle=True)
    #X_val,y_val = shuffle(X_val,y_val)
    print('shuffled')
    if clf_type == CLF_TYPE.K_SVM or clf_type == CLF_TYPE.SVM:
        y = enforce_svm_labels(y)
    #    y_val = enforce_svm_labels(y_val)
    else:
        y = enforce_logistic_labels(y)
    #    y_val = enforce_logistic_labels(y_val)
    print('changed labels')
    hyperparams = [0.1, 1, 10, 100] if args.try_hyperparams else [1]
    # hyperparams = [1000,10000] if args.try_hyperparams else [1]
    split = int(.2*len(X))
    X_train = X[split:][:]
    y_train = y[split:][:]
    X_val = X[:split][:]
    y_val = y[:split][:]
    print('data was split')
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    with open(MATCHING_OUTPUT_PATH + 'moredata_scaler.pkl', 'wb') as fid:
        pickle.dump(scaler, fid)
    X_val = scaler.transform(X_val)
    print('data was scaled')

    # draw_validation_curve(X_train, y_train, clf_type)
    # return
    for C in tqdm(hyperparams):
        if clf_type == CLF_TYPE.K_SVM:
            clf = SVC(C = C, verbose=2).fit(X_train, y_train)
        elif clf_type == CLF_TYPE.Logistic:
            clf = LogisticRegression(penalty='l1', C = C).fit(X_train, y_train)
        elif clf_type == CLF_TYPE.SVM:
            clf = LinearSVC(C=C).fit(X_train, y_train)
        else:
            print('clf type not implemented')
            return
        accuracy = clf.score(X_val, y_val)
        model_fname = MATCHING_OUTPUT_PATH + 'moredata_{0:}_baseline_Cis{1:.2f}_accuris{2:.4f}.pkl'.format(clf_type.name, C, accuracy)
        with open(model_fname, 'wb') as fid:
            pickle.dump(clf, fid)
        print('Predictor Accuracy on validation set:', accuracy)
        print('----')

if __name__=='__main__':
    main()

import numpy as np
from pprint import pprint
import random
import argparse

from util.constants import *

# since X is not an adjacency matrix, X does not need to be n*n, could be n*m where n = # of doodles and m = # of images
def generate_data(n, multi = 1):
    X = np.zeros((n,n), dtype=int)
    for i in range(len(X)): # t represents the number of diff. labels
        for j in range(len(X[0])):
            l = np.random.randint(multi + 1, size = 1)[0]
            X[i][j] = l
    return X

def pivot_bi_cluster(num_doodle, num_img, X):
    C = []
    doodle_IDs = list(np.linspace(0, num_doodle - 1, num = num_doodle, dtype = int))
    image_IDs = list(np.linspace(0, num_img - 1, num = num_img, dtype = int))
    while True:
        pivot = random.choice(doodle_IDs)
        doodle_IDs.remove(pivot)
        cluster = []
        cluster.append(str(pivot)+'D')
        # fill the cluster with every image that is linked with the pivot
        R_main = X[pivot]
        # now go through every remaining l2 and check whether R1,2 or R2 --> then decide on the action for every l2
        for d in doodle_IDs:
            R_curr = X[d]
            R12 = set(R_main).intersection(set(R_curr))
            R2 = set(R_curr).difference(set(R_main))
            R1 = set(R_main).difference(set(R_curr))
            p = 1.0
            if len(R2) > 0: 
                p = float(len(R12)) / len(R2)
            if p > 1: 
                p = 1
            r = np.random.rand(1)
            if r < p:
                if len(R12) >= len(R1):
                    cluster.append(str(d) + 'D')
                else:
                    C.append([str(d)+'D'])
                doodle_IDs.remove(d)
        for im in R_main:
            cluster.append(str(im)+'Im')
            image_IDs.remove(im)
        C.append(cluster)
        if len(image_IDs) == 0 or len(doodle_IDs) == 0:
            break
    
    print(C)
    print('---remaining doodles---')
    print(doodle_IDs)
    print('---remaining images---')
    print(image_IDs)
    return C

def generate_data(n, multi = 1):
    X = np.zeros((n,n), dtype=int)
    for i in range(len(X)): # t represents the number of diff. labels
        for j in range(len(X[0]) - i):
           if i != j: # no self edge
               l = np.random.randint(multi + 1, size = 1)[0]
               X[i][j] = l
               X[j][i] = l
    return X

# col = image, row = doodle (not exactly an adjacency matrix in a traditional sense because the matrix is asymmetric and X_ij represents [doodle i, image j] != X_ji representing [doodle j, image i]
# X = generate_data(20)

parser = argparse.ArgumentParser()
parser.add_argument('--features_fname', default='svm_2d_2i_5cat_clustering.py.npy')
args = parser.parse_args()

infname = GRAPH_FEATURES + args.features_fname
edges_info = np.load(infname, allow_pickle=True).item()
X = edges_info['X']

print('EDGES')
pprint(X)
print('============================\n\n')

print('DOODLE ROWS')
pprint(edges_info['doodle_rows'])
print('============================\n\n')

print('IMG COLS')
pprint(edges_info['image_cols'])
print('============================\n\n')

print('----')
C = pivot_bi_cluster(X)

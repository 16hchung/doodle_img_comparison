
def enforce_svm_labels(y):
    y[y==0] = -1
    return y

def enforce_logistic_labels(y):
    y[y==-1] = 0
    return y
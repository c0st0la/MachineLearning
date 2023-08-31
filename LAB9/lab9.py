import numpy

from functions import *

if __name__=="__main__":
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    DTRnew = numpy.vstack((DTR, numpy.ones((1,DTR.shape[1]))))
    DTEnew = numpy.vstack((DTE, numpy.ones((1,DTE.shape[1]))))
    C = 10.0
    K = 10.0
    c=1
    d=2
    gamma = 1.0
    svm = support_vector_machine(DTR, LTR, DTE, LTE, K, C)
    svm2 = support_vector_machine_kernel(DTR, LTR, DTE, LTE, K, C, 'r', c=None, d=None, gamma=gamma)



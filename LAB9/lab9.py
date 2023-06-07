import numpy

from functions import *

if __name__=="__main__":
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    DTRnew = numpy.vstack((DTR, numpy.ones((1,DTR.shape[1]))))
    DTEnew = numpy.vstack((DTE, numpy.ones((1,DTE.shape[1]))))
    C = 10
    K = 10
    svm = support_vector_machine(DTR, LTR, DTE, LTE, K, C)




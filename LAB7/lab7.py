from functions import *

if __name__ == "__main__":
    # the scipy.fmin_l_bfgs_b returns three values
    # x is the estimated position of the minimum
    # f is the objective value at the minimum
    # d contains additional information (check the documentation)
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    l = 10 ** -6
    logistic_regression_binary(DTR, LTR, DTE, LTE, l)
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    logistic_regression_multiclass(DTR, LTR, DTE, LTE, feature_space_dimension=4, num_classes=3, l=l)

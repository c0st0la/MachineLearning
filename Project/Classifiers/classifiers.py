import numpy
from Project import functions


def compute_MVG_KFold_DCF(D, L, numFold, applicationWorkingPoint, costs, labels):
    num_samples = int(D.shape[1] / numFold)
    totDCF = 0
    perm = numpy.random.permutation(D.shape[1])
    D = D[:, perm]
    L = L[perm]
    thresholds = [i for i in numpy.arange(-30, 30, 0.1)]
    for i in range(numFold):
        DCFsNormalized1 = []
        (DTR, LTR), (DTE, LTE) = functions.K_fold_generate_Training_and_Testing_samples(D, L, i, numFold, num_samples)
        llr_MVG = functions.compute_MVG_llrs(DTR, LTR, DTE, labels)
        for threshold in thresholds:
            optimalBayesDecisionPredictions = functions.compute_optimal_bayes_decision_given_threshold(llr_MVG,
                                                                                                       threshold)
            confusionMatrix = functions.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
            DCFsNormalized1.append(
                functions.compute_normalized_detection_cost_function(confusionMatrix, applicationWorkingPoint, costs))
        totDCF += min(DCFsNormalized1)
    return totDCF / numFold

def compute_NB_KFold_DCF(D, L, numFold, applicationWorkingPoint, costs, labels):
    num_samples = int(D.shape[1] / numFold)
    totDCF = 0

    perm = numpy.random.permutation(D.shape[1])
    D = D[:, perm]
    L = L[perm]
    thresholds = [i for i in numpy.arange(-30, 30, 0.1)]
    for i in range(numFold):
        DCFsNormalized1 = []
        (DTR, LTR), (DTE, LTE) = functions.K_fold_generate_Training_and_Testing_samples(D, L, i, numFold, num_samples)
        llr_NB = functions.compute_NB_llrs(DTR, LTR, DTE, labels)
        for threshold in thresholds:
            optimalBayesDecisionPredictions = functions.compute_optimal_bayes_decision_given_threshold(llr_NB,
                                                                                                       threshold)
            confusionMatrix = functions.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
            DCFsNormalized1.append(
                functions.compute_normalized_detection_cost_function(confusionMatrix, applicationWorkingPoint, costs))
        totDCF += min(DCFsNormalized1)
    return totDCF / numFold

def compute_TC_KFold_DCF(D, L, numFold, applicationWorkingPoint, costs, labels):
    num_samples = int(D.shape[1] / numFold)
    totDCF = 0

    perm = numpy.random.permutation(D.shape[1])
    D = D[:, perm]
    L = L[perm]
    thresholds = [i for i in numpy.arange(-30, 30, 0.1)]
    for i in range(numFold):
        DCFsNormalized1 = []
        (DTR, LTR), (DTE, LTE) = functions.K_fold_generate_Training_and_Testing_samples(D, L, i, numFold, num_samples)
        llr_TC = functions.compute_TC_llrs(DTR, LTR, DTE, labels)
        for threshold in thresholds:
            optimalBayesDecisionPredictions = functions.compute_optimal_bayes_decision_given_threshold(llr_TC,
                                                                                                       threshold)
            confusionMatrix = functions.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
            DCFsNormalized1.append(
                functions.compute_normalized_detection_cost_function(confusionMatrix, applicationWorkingPoint, costs))
        totDCF += min(DCFsNormalized1)
    return totDCF / numFold

def compute_TNB_KFold_DCF(D, L, numFold, applicationWorkingPoint, costs, labels):
    num_samples = int(D.shape[1] / numFold)
    totDCF = 0

    perm = numpy.random.permutation(D.shape[1])
    D = D[:, perm]
    L = L[perm]
    thresholds = [i for i in numpy.arange(-30, 30, 0.1)]
    for i in range(numFold):
        DCFsNormalized1 = []
        (DTR, LTR), (DTE, LTE) = functions.K_fold_generate_Training_and_Testing_samples(D, L, i, numFold, num_samples)
        llr_TNB = functions.compute_TNB_llrs(DTR, LTR, DTE, labels)
        for threshold in thresholds:
            optimalBayesDecisionPredictions = functions.compute_optimal_bayes_decision_given_threshold(llr_TNB,
                                                                                                       threshold)
            confusionMatrix = functions.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
            DCFsNormalized1.append(
                functions.compute_normalized_detection_cost_function(confusionMatrix, applicationWorkingPoint, costs))
        totDCF += min(DCFsNormalized1)
    return totDCF / numFold

def compute_LR_KFold_DCF(D, L, numFold, classPriorProbabilities, applicationWorkingPoint, costs, lambdaValues):
    num_samples = int(D.shape[1] / numFold)

    x = dict()
    perm = numpy.random.permutation(D.shape[1])
    D = D[:, perm]
    L = L[perm]
    thresholds = [i for i in numpy.arange(-30, 30, 0.1)]
    for i in range(numFold):
        DCFsNormalized1 = []
        (DTR, LTR), (DTE, LTE) = functions.K_fold_generate_Training_and_Testing_samples(D, L, i, numFold, num_samples)
        for lambd in lambdaValues:
            llr_LR = functions.compute_logistic_regression_binary_llr(DTR, LTR, DTE, lambd, classPriorProbabilities)
            for threshold in thresholds:
                optimalBayesDecisionPredictions = functions.compute_optimal_bayes_decision_given_threshold(llr_LR,
                                                                                                           threshold)
                confusionMatrix = functions.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
                DCFsNormalized1.append(
                    functions.compute_normalized_detection_cost_function(confusionMatrix, applicationWorkingPoint,
                                                                         costs))
            if lambd not in list(x.keys()):
                x[lambd] = min(DCFsNormalized1)
            else:
                x[lambd] = x[lambd] + min(DCFsNormalized1)
    for key in list(x.keys()):
        x[key] = x[key] / numFold
    return x


def compute_QLR_KFold_DCF(D, L, numFold, classPriorProbabilities, applicationWorkingPoint, costs, lambdaValues):
    num_samples = int(D.shape[1] / numFold)
    totDCF = 0

    x = dict()
    perm = numpy.random.permutation(D.shape[1])
    D = D[:, perm]
    L = L[perm]
    thresholds = [i for i in numpy.arange(-30, 30, 0.1)]
    for i in range(numFold):
        DCFsNormalized1 = []
        (DTR, LTR), (DTE, LTE) = functions.K_fold_generate_Training_and_Testing_samples(D, L, i, numFold, num_samples)
        DTR = functions.quadratic_expansion(DTR)
        DTE = functions.quadratic_expansion(DTE)
        for lambd in lambdaValues:
            llr_QLR = functions.compute_logistic_regression_binary_quadratic_llr(DTR, LTR, DTE, lambd,
                                                                                 classPriorProbabilities)
            for threshold in thresholds:
                optimalBayesDecisionPredictions = functions.compute_optimal_bayes_decision_given_threshold(llr_QLR,
                                                                                                           threshold)
                confusionMatrix = functions.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
                DCFsNormalized1.append(
                    functions.compute_normalized_detection_cost_function(confusionMatrix, applicationWorkingPoint,
                                                                         costs))
            if lambd not in list(x.keys()):
                x[lambd] = min(DCFsNormalized1)
            else:
                x[lambd] = x[lambd] + min(DCFsNormalized1)
    for key in list(x.keys()):
        x[key] = x[key] / numFold
    return x


def compute_SVM_KFold_DCF(D, L, numFold, classPriorProbabilities, applicationWorkingPoint, costs, CList, K):
    num_samples = int(D.shape[1] / numFold)

    x = dict()
    perm = numpy.random.permutation(D.shape[1])
    D = D[:, perm]
    L = L[perm]
    thresholds = [i for i in numpy.arange(-30, 30, 0.1)]
    for i in range(numFold):
        DCFsNormalized1 = []
        (DTR, LTR), (DTE, LTE) = functions.K_fold_generate_Training_and_Testing_samples(D, L, i, numFold, num_samples)
        for C in CList:
            llr_SVM = functions.compute_support_vector_machine_llr(DTR, LTR, DTE, LTE, K, C, classPriorProbabilities)
            for threshold in thresholds:
                optimalBayesDecisionPredictions = functions.compute_optimal_bayes_decision_given_threshold(llr_SVM,
                                                                                                           threshold)
                confusionMatrix = functions.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
                DCFsNormalized1.append(
                    functions.compute_normalized_detection_cost_function(confusionMatrix, applicationWorkingPoint,
                                                                         costs))
            if C not in list(x.keys()):
                x[C] = min(DCFsNormalized1)
            else:
                x[C] = x[C] + min(DCFsNormalized1)
    for key in list(x.keys()):
        x[key] = x[key] / numFold
    return x


def compute_PolySVM_KFold_DCF(D, L, numFold, classPriorProbabilities, applicationWorkingPoint, costs, CList, K, d, c):
    # CList = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5]
    num_samples = int(D.shape[1] / numFold)

    x = dict()
    perm = numpy.random.permutation(D.shape[1])
    D = D[:, perm]
    L = L[perm]
    thresholds = [i for i in numpy.arange(-30, 30, 0.1)]
    for i in range(numFold):
        DCFsNormalized1 = []
        (DTR, LTR), (DTE, LTE) = functions.K_fold_generate_Training_and_Testing_samples(D, L, i, numFold, num_samples)
        for C in CList:
            llr_PolySVM = functions.compute_support_vector_machine_kernel_llr(DTR, LTR, DTE, LTE, K, C,
                                                                              'p', classPriorProbabilities, c=c,
                                                                              d=d)
            for threshold in thresholds:
                optimalBayesDecisionPredictions = functions.compute_optimal_bayes_decision_given_threshold(
                    llr_PolySVM,
                    threshold)
                confusionMatrix = functions.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
                DCFsNormalized1.append(functions.compute_normalized_detection_cost_function(confusionMatrix,
                                                                                            applicationWorkingPoint,
                                                                                            costs))
            if C not in list(x.keys()):
                x[C] = min(DCFsNormalized1)
            else:
                x[C] = x[C] + min(DCFsNormalized1)
    for key in list(x.keys()):
        x[key] = x[key] / numFold
    return x

def compute_RadialBasisSVM_KFold_DCF(D, L, numFold, classPriorProbabilities, costs, CList, K, gammaValues, d, c):
    # CList = [10 ** -5, 10 ** -3, 10 ** -1, 10, 10 ** 3, 10 ** 5]
    # gammaValues = [10 ** -3, 10 ** -2, 10 ** -1]
    num_samples = int(D.shape[1] / numFold)

    x = dict()
    perm = numpy.random.permutation(D.shape[1])
    D = D[:, perm]
    L = L[perm]
    thresholds = [i for i in numpy.arange(-30, 30, 0.1)]
    for i in range(numFold):
        DCFsNormalized1 = []
        (DTR, LTR), (DTE, LTE) = functions.K_fold_generate_Training_and_Testing_samples(D, L, i, numFold, num_samples)
        for gamma in gammaValues:
            for C in CList:
                llr_PolySVM = functions.compute_support_vector_machine_kernel_llr(DTR, LTR, DTE, LTE, K, C,
                                                                                  'r', classPriorProbabilities, c=c,
                                                                                  d=d, gamma=gamma)
                for threshold in thresholds:
                    optimalBayesDecisionPredictions = functions.compute_optimal_bayes_decision_given_threshold(
                        llr_PolySVM,
                        threshold)
                    confusionMatrix = functions.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
                    DCFsNormalized1.append(functions.compute_normalized_detection_cost_function(confusionMatrix,
                                                                                                applicationWorkingPoint,
                                                                                                costs))
                if (C, gamma) not in list(x.keys()):
                    x[(C, gamma)] = min(DCFsNormalized1)
                else:
                    x[(C, gamma)] = x[(C, gamma)] + min(DCFsNormalized1)
    for key in list(x.keys()):
        x[key] = x[key] / numFold
    return x

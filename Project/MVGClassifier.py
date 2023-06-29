import matplotlib.pyplot as plt
from functions2 import *
import seaborn

if __name__ == "__main__":


    DTRSplitted, LTR, DTROriginal=read_file("Train.txt")
    DTESplitted, LTE, DTEOriginal=read_file("Test.txt")
    classPriorProbabilities1 = numpy.array([9/10, 1/10], dtype=float)
    classPriorProbabilities2 = numpy.array([5/10, 5/10], dtype=float)
    classPriorProbabilities3 = numpy.array([1/10, 9/10], dtype=float)
    costs = numpy.array([1.0, 1.0], dtype=float)
    labels = [i for i in range(0, numpy.amax(LTR) + 1)]
    numFold = 4

    DTROriginalNormalized = (DTROriginal - compute_mean(DTROriginal))/ to_column(DTROriginal.std(axis=1))
    DTEOriginalNormalized = (DTEOriginal - compute_mean(DTEOriginal)) / to_column(DTEOriginal.std(axis=1))


    # NOW I WILL COMPUTE DCF GIVEN A SET OF TRHESHOLDS


    thresholds = [i for i in numpy.arange(-30,30, 0.1)]
    DCFsNormalized1 = []
    DCFsNormalized2 = []
    DCFsNormalized3 = []

    llr_MVG = compute_MVG_llrs(DTROriginalNormalized, LTR, DTEOriginalNormalized, labels)
    for threshold in thresholds:
        optimalBayesDecisionPredictions = compute_optimal_bayes_decision_given_threshold(llr_MVG, threshold)
        confusionMatrix = compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
        DCFsNormalized1.append(
            compute_normalized_detection_cost_function(confusionMatrix, classPriorProbabilities1,costs))
        DCFsNormalized2.append(
            compute_normalized_detection_cost_function(confusionMatrix, classPriorProbabilities2, costs))
        DCFsNormalized3.append(
            compute_normalized_detection_cost_function(confusionMatrix, classPriorProbabilities3, costs))
    DFCmin1 = min(DCFsNormalized1)
    DFCmin2 = min(DCFsNormalized2)
    DFCmin3 = min(DCFsNormalized3)


    kFoldDCFmin1 = K_fold_cross_validation_DCF(DTROriginalNormalized, LTR, "MVG",
                                               numFold, classPriorProbabilities1, costs, labels)
    kFoldDCFmin2 = K_fold_cross_validation_DCF(DTROriginalNormalized, LTR, "MVG",
                                               numFold, classPriorProbabilities2, costs, labels)
    kFoldDCFmin3 = K_fold_cross_validation_DCF(DTROriginalNormalized, LTR, "MVG",
                                               numFold, classPriorProbabilities3, costs, labels)

    print("Prior probability for class 0 is : ", classPriorProbabilities1[0])
    print("Prior probability for class 1 is : ", classPriorProbabilities1[1])
    print("The min of the normalized DFC for MVG classifier: %.3f" % DFCmin1)
    print("The min of the normalized DFC for MVG classifier with K-fold algo (%d fold): %.3f\n" % (numFold, kFoldDCFmin1))

    print("Prior probability for class 0 is : ", classPriorProbabilities2[0])
    print("Prior probability for class 1 is : ", classPriorProbabilities2[1])
    print("The min of the normalized DFC for MVG classifier : %.3f" % DFCmin2)
    print("The min of the normalized DFC for MVG classifier with K-fold algo (%d fold): %.3f\n" % (numFold, kFoldDCFmin2))


    print("Prior probability for class 0 is : ", classPriorProbabilities3[0])
    print("Prior probability for class 1 is : ", classPriorProbabilities3[1])
    print("The min of the normalized DFC for MVG classifier : %.3f" % DFCmin3)
    print("The min of the normalized DFC for MVG classifier with K-fold algo (%d fold): %.3f\n" % (numFold, kFoldDCFmin3))


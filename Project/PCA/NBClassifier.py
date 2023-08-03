import numpy
from Project import functions

if __name__ == "__main__":


    DTRSplitted, LTR, DTROriginal=functions.read_file("../Train.txt")
    DTESplitted, LTE, DTEOriginal=functions.read_file("../Test.txt")
    classPriorProbabilities1 = numpy.array([9 / 10, 1 / 10], dtype=float)
    classPriorProbabilities2 = numpy.array([5 / 10, 5 / 10], dtype=float)
    classPriorProbabilities3 = numpy.array([1 / 10, 9 / 10], dtype=float)
    costs = numpy.array([1.0, 1.0], dtype=float)
    labels = [i for i in range(0, numpy.amax(LTR) + 1)]
    numFold = 5
    thresholds = [i for i in numpy.arange(-30, 30, 0.1)]

    DTROriginalNormalized = (DTROriginal - functions.compute_mean(DTROriginal))/ functions.to_column(DTROriginal.std(axis=1))
    DTEOriginalNormalized = (DTEOriginal - functions.compute_mean(DTEOriginal)) / functions.to_column(DTEOriginal.std(axis=1))

    DOriginalNormalized = numpy.concatenate((DTROriginalNormalized, DTEOriginalNormalized), axis=1)
    # DOriginal = numpy.concatenate((DTROriginal, DTEOriginal), axis=1)
    L = numpy.concatenate((LTR, LTE), axis=0)

    DTRNormalizedOriginalFilteredTrue, DTRNormalizedOriginalFilteredFalse = functions.filter_dataset_by_labels(DTROriginalNormalized, LTR)

    toPrint= ""
    for subDimensionPCA in range(7, DTROriginal.shape[0]):
        toPrint += "PCA with %d dimension" % subDimensionPCA + "\n"
        DTRNormalizedPCAOriginal, P = functions.compute_PCA(DTROriginalNormalized, subDimensionPCA)
        DTENormalizedPCAOriginal = numpy.dot(P.T, DTEOriginalNormalized)
        DOriginalPCANormalized = numpy.concatenate((DTRNormalizedPCAOriginal, DTENormalizedPCAOriginal), axis=1)
        DCFsNormalized1 = []
        DCFsNormalized2 = []
        DCFsNormalized3 = []

        llr_NB = functions.compute_NB_llrs(DTRNormalizedPCAOriginal, LTR, DTENormalizedPCAOriginal, labels)
        for threshold in thresholds:
            optimalBayesDecisionPredictions = functions.compute_optimal_bayes_decision_given_threshold(llr_NB, threshold)
            confusionMatrix = functions.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
            DCFsNormalized1.append(
                functions.compute_normalized_detection_cost_function(confusionMatrix, classPriorProbabilities1, costs))
            DCFsNormalized2.append(
                functions.compute_normalized_detection_cost_function(confusionMatrix, classPriorProbabilities2, costs))
            DCFsNormalized3.append(
                functions.compute_normalized_detection_cost_function(confusionMatrix, classPriorProbabilities3, costs))
        DFCmin1 = min(DCFsNormalized1)
        DFCmin2 = min(DCFsNormalized2)
        DFCmin3 = min(DCFsNormalized3)

        kFoldDCFmin1 = functions.K_fold_cross_validation_DCF(DOriginalPCANormalized, L, "NB",
                                                   numFold, classPriorProbabilities1, costs, labels)
        kFoldDCFmin2 = functions.K_fold_cross_validation_DCF(DOriginalPCANormalized, L, "NB",
                                                   numFold, classPriorProbabilities2, costs, labels)
        kFoldDCFmin3 = functions.K_fold_cross_validation_DCF(DOriginalPCANormalized, L, "NB",
                                                   numFold, classPriorProbabilities3, costs, labels)


        toPrint += "Prior probability for class 0 is : " + str(classPriorProbabilities1[0]) + "\n"
        toPrint += "Prior probability for class 1 is : " + str(classPriorProbabilities1[1]) + "\n"
        toPrint += "The min of the normalized DCF for NB classifier: %.3f\n" % DFCmin1
        toPrint +=  "The min of the normalized DCF for NB classifier with K-fold algo (%d fold): %.3f\n" % (
        numFold, kFoldDCFmin1) + "\n"

        toPrint += "Prior probability for class 0 is : " + str(classPriorProbabilities2[0]) + "\n"
        toPrint += "Prior probability for class 1 is : " + str(classPriorProbabilities2[1]) + "\n"
        toPrint += "The min of the normalized DCF for NB classifier : %.3f\n" % DFCmin2
        toPrint += "The min of the normalized DCF for NB classifier with K-fold algo (%d fold): %.3f\n" % (
        numFold, kFoldDCFmin2) + "\n"

        toPrint += "Prior probability for class 0 is : " + str(classPriorProbabilities3[0]) + "\n"
        toPrint += "Prior probability for class 1 is : " + str(classPriorProbabilities3[1]) + "\n"
        toPrint += "The min of the normalized DCF for NB classifier : %.3f\n" % DFCmin3
        toPrint += "The min of the normalized DCF for NB classifier with K-fold algo (%d fold): %.3f\n" % (
        numFold, kFoldDCFmin3) + "\n"

        with open("datiPCA/datiZScorePCANB.txt", "w") as fp:
            fp.write(toPrint)

        ##

        toPrint = ""
        for subDimensionPCA in range(7, DTROriginal.shape[0]):
            toPrint += "PCA with %d dimension" % subDimensionPCA + "\n"
            DTRPCAOriginal, P = functions.compute_PCA(DTROriginal, subDimensionPCA)
            DTEPCAOriginal = numpy.dot(P.T, DTEOriginal)
            DPCAriginal = numpy.concatenate((DTRPCAOriginal, DTEPCAOriginal), axis=1)
            DCFsNormalized1 = []
            DCFsNormalized2 = []
            DCFsNormalized3 = []

            llr_NB = functions.compute_NB_llrs(DTRPCAOriginal, LTR, DTEPCAOriginal, labels)
            for threshold in thresholds:
                optimalBayesDecisionPredictions = functions.compute_optimal_bayes_decision_given_threshold(llr_NB,
                                                                                                           threshold)
                confusionMatrix = functions.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
                DCFsNormalized1.append(
                    functions.compute_normalized_detection_cost_function(confusionMatrix, classPriorProbabilities1,
                                                                         costs))
                DCFsNormalized2.append(
                    functions.compute_normalized_detection_cost_function(confusionMatrix, classPriorProbabilities2,
                                                                         costs))
                DCFsNormalized3.append(
                    functions.compute_normalized_detection_cost_function(confusionMatrix, classPriorProbabilities3,
                                                                         costs))
            DFCmin1 = min(DCFsNormalized1)
            DFCmin2 = min(DCFsNormalized2)
            DFCmin3 = min(DCFsNormalized3)

            kFoldDCFmin1 = functions.K_fold_cross_validation_DCF(DPCAriginal, L, "NB",
                                                                 numFold, classPriorProbabilities1, costs, labels)
            kFoldDCFmin2 = functions.K_fold_cross_validation_DCF(DPCAriginal, L, "NB",
                                                                 numFold, classPriorProbabilities2, costs, labels)
            kFoldDCFmin3 = functions.K_fold_cross_validation_DCF(DPCAriginal, L, "NB",
                                                                 numFold, classPriorProbabilities3, costs, labels)

            toPrint += "Prior probability for class 0 is : " + str(classPriorProbabilities1[0]) + "\n"
            toPrint += "Prior probability for class 1 is : " + str(classPriorProbabilities1[1]) + "\n"
            toPrint += "The min of the normalized DCF for NB classifier: %.3f\n" % DFCmin1
            toPrint += "The min of the normalized DCF for NB classifier with K-fold algo (%d fold): %.3f\n" % (
                numFold, kFoldDCFmin1) + "\n"

            toPrint += "Prior probability for class 0 is : " + str(classPriorProbabilities2[0]) + "\n"
            toPrint += "Prior probability for class 1 is : " + str(classPriorProbabilities2[1]) + "\n"
            toPrint += "The min of the normalized DCF for NB classifier : %.3f\n" % DFCmin2
            toPrint += "The min of the normalized DCF for NB classifier with K-fold algo (%d fold): %.3f\n" % (
                numFold, kFoldDCFmin2) + "\n"

            toPrint += "Prior probability for class 0 is : " + str(classPriorProbabilities3[0]) + "\n"
            toPrint += "Prior probability for class 1 is : " + str(classPriorProbabilities3[1]) + "\n"
            toPrint += "The min of the normalized DCF for NB classifier : %.3f\n" % DFCmin3
            toPrint += "The min of the normalized DCF for NB classifier with K-fold algo (%d fold): %.3f\n" % (
                numFold, kFoldDCFmin3) + "\n"

            with open("datiPCA/datiRawPCANB.txt", "w") as fp:
                fp.write(toPrint)

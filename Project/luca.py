import numpy
from functions import *
from functions2 import *

if __name__ == "__main__":


    DTRSplitted, LTR, DTROriginal=read_file("Train.txt")
    DTESplitted, LTE, DTEOriginal=read_file("Test.txt")
    classPriorProbabilities = numpy.array([9/10, 1/10], dtype=float)
    costs = numpy.array([1.0, 1.0], dtype=float)
    labels = [i for i in range(0, numpy.amax(LTR) + 1)]

    DTROriginalNormalized = (DTROriginal - compute_mean(DTROriginal))/ to_column(DTROriginal.std(axis=1))
    DTEOriginalNormalized = (DTEOriginal - compute_mean(DTEOriginal)) / to_column(DTEOriginal.std(axis=1))

    DTRNormalizedOriginalFilteredTrue, DTRNormalizedOriginalFilteredFalse = filter_dataset_by_labels(
        DTROriginalNormalized, LTR)

    #plot_scatter_attributes_X_label_True_False(DTRNormalizedOriginalFilteredTrue, DTRNormalizedOriginalFilteredFalse,
    # filepath="./FeatureCorrelation/", title="DTROriginalNormalized")


    print(f"No Pre-Processing")

    accuracyMVG = compute_MVG_accuracy(DTROriginalNormalized, LTR, DTEOriginalNormalized, LTE, labels,
                                       classPriorProbabilities)
    print("The MVG accuracy is %.3f" % accuracyMVG)

    accuracyNB = compute_NB_accuracy(DTROriginalNormalized, LTR, DTEOriginalNormalized, LTE, labels,
                                     classPriorProbabilities)
    print("The NB accuracy is %.3f" % accuracyNB)

    accuracyTC = compute_TC_accuracy(DTROriginalNormalized, LTR, DTEOriginalNormalized, LTE, labels,
                                     classPriorProbabilities)
    print("The TC accuracy is %.3f" % accuracyTC)

    accuracyTNB = compute_TNB_accuracy(DTROriginalNormalized, LTR, DTEOriginalNormalized, LTE, labels,
                                       classPriorProbabilities)
    print("The TNB accuracy is %.3f" % accuracyTNB)

    MVGlogLikelihoodRatio = compute_MVG_llr(DTROriginalNormalized, LTR, DTEOriginalNormalized, labels)

    optimalBayesDecisionPredictions = compute_optimal_bayes_decision(MVGlogLikelihoodRatio, classPriorProbabilities,
                                                                     costs)
    confusionMatrix = compute_confusion_matrix(optimalBayesDecisionPredictions, LTE)
    DCF = compute_detection_cost_function(confusionMatrix, classPriorProbabilities, costs)
    DCFNormalized = compute_normalized_detection_cost_function(confusionMatrix, classPriorProbabilities, costs)
    print("The priors probabilities are : ", classPriorProbabilities, "\n")
    print("The costs are :")
    print("Costs of false negative (label a class to 0 when the real is 1) : ", classPriorProbabilities[0], "\n")
    print("Costs of false positive (label a class to 1 when the real is 0) : ", classPriorProbabilities[1], "\n")
    print("Confusion Matrix : \n", confusionMatrix, "\n")
    print("DCF : %.3f\n" % DCF)
    print("Normalized DCF : %.3f\n" % DCFNormalized)

    # for lambd in [10**-8, 10**-5, 10**-4, 10**-1, 1, 10]:
    #     threshold=0
    #
    #     accuracyLR = compute_LR_accuracy(DTROriginalNormalized, LTR, DTEOriginalNormalized, LTE, classPriorProbabilities,
    #                                     lambd, threshold)
    #     print("The LR accuracy {lambda=%f, threshold=%.5f} is %.3f" % (lambd, threshold, accuracyLR))
    #
    #     accuracyQLR = compute_QLR_accuracy(DTROriginalNormalized, LTR, DTEOriginalNormalized, LTE,
    #                                      classPriorProbabilities,
    #                                      lambd, threshold)
    #     print("The QLR accuracy {lambda=%f, threshold=%.5f} is %.3f" % (lambd, threshold, accuracyQLR))
    #
    #     print("Whitening pre processing applied on LR/QLR")
    #
    #     accuracyLR = compute_LR_accuracy(withening_pre_processing(DTROriginalNormalized), LTR, DTEOriginalNormalized, LTE,
    #                                      classPriorProbabilities,
    #                                      lambd, threshold)
    #     print("The LR accuracy {lambda=%f, threshold=%.5f} is %.3f" % (lambd, threshold, accuracyLR))
    #
    #     accuracyQLR = compute_QLR_accuracy(withening_pre_processing(DTROriginalNormalized), LTR, DTEOriginalNormalized, LTE,
    #                                         classPriorProbabilities,
    #                                         lambd, threshold)
    #     print("The QLR accuracy {lambda=%f, threshold=%.5f} is %.3f" % (lambd, threshold, accuracyQLR))
    #
    #     print("Whitening and length pre processing applied on LR/QLR")
    #
    #     accuracyLR = compute_LR_accuracy(length_normalization(withening_pre_processing(DTROriginalNormalized)), LTR, DTEOriginalNormalized,
    #                                      LTE,
    #                                      classPriorProbabilities,
    #                                      lambd, threshold)
    #     print("The LR accuracy {lambda=%f, threshold=%.5f} is %.3f" % (lambd, threshold, accuracyLR))
    #
    #     accuracyQLR = compute_QLR_accuracy(length_normalization(withening_pre_processing(DTROriginalNormalized)), LTR, DTEOriginalNormalized,
    #                                        LTE,
    #                                        classPriorProbabilities,
    #                                        lambd, threshold)
    #     print("The QLR accuracy {lambda=%f, threshold=%.5f} is %.3f" % (lambd, threshold, accuracyQLR))
    #
    #
    # print("\n")


    # WITH PCA I CAN TRY TO REDUCE THE DIMENSION OF THE FEATURE SPACE
    # ACTUALLY OUR FEATURE SPACE IS 10. SO PCA CAN TRY TO CREATE A SUBSPACE WHOSE DIMENSION
    # IS IN THIS RANGE (1, 9)rns the varian
    for subDimensionPCA in range(1, DTROriginal.shape[0]):
        DTRNormalizedPCAOriginal, P = compute_PCA(DTROriginalNormalized, subDimensionPCA)
        # Come dovrei implementare la proiezione di DTE?
        DTENormalizedPCAOriginal = numpy.dot(P.T, DTEOriginalNormalized)
        # DTENormalizedPCAOriginal, P = compute_PCA(DTEOriginalNormalized, subDimensionPCA)
        DTRNormalizedPCAOriginalFilteredTrue, DTRNormalizedPCAOriginalFilteredFalse = filter_dataset_by_labels(DTRNormalizedPCAOriginal, LTR)
        #plot_scatter_attributes_X_label(DTRNormalizedPCAOriginalFilteredTrue, filepath="./FeaturesCorrelationPCA/", title="DTrue PCA")
        pairs.clear()
        #plot_scatter_attributes_X_label(DTRNormalizedPCAOriginalFilteredFalse, filepath="./FeaturesCorrelationPCA/", title="DFalse PCA")
        pairs.clear()

        ### HERE I COMPUTE LDA ###
        DTRNormalizedPCALDAOriginal, W = compute_LDA_generalized_eigenvalue(DTRNormalizedPCAOriginal, LTR, 1, labels)
        DTENormalizedPCALDAOriginal = numpy.dot(W.T, DTENormalizedPCAOriginal)
        DTRNormalizedPCALDAOriginalFilteredTrue, DTRNormalizedPCALDAOriginalFilteredFalse = filter_dataset_by_labels(
        DTRNormalizedPCALDAOriginal, LTR)
        #plot_scatter_attributes_X_label(DTRNormalizedPCALDAOriginalFilteredTrue, filepath="./FeaturesCorrelationPCA/",title="DTrue LDA")
        pairs.clear()
        #plot_scatter_attributes_X_label(DTRNormalizedPCALDAOriginalFilteredFalse, filepath="./FeaturesCorrelationPCA/", title="DFalse LDA")

        print(f"PCA with {subDimensionPCA} dimension")

        accuracyMVG = compute_MVG_accuracy(DTRNormalizedPCAOriginal, LTR, DTENormalizedPCAOriginal, LTE, labels, classPriorProbabilities)
        print("The MVG accuracy is %.3f" % accuracyMVG)

        accuracyNB = compute_NB_accuracy(DTRNormalizedPCAOriginal, LTR, DTENormalizedPCAOriginal, LTE, labels,
                                           classPriorProbabilities)
        print("The NB accuracy is %.3f" % accuracyNB)

        accuracyTC = compute_TC_accuracy(DTRNormalizedPCAOriginal, LTR, DTENormalizedPCAOriginal, LTE, labels,
                                         classPriorProbabilities)
        print("The TC accuracy is %.3f" % accuracyTC)

        accuracyTNB = compute_TNB_accuracy(DTRNormalizedPCAOriginal, LTR, DTENormalizedPCAOriginal, LTE, labels,
                                         classPriorProbabilities)
        print("The TNB accuracy is %.3f" % accuracyTNB)

        accuracyLDA =compute_binary_LDA_accuracy(DTENormalizedPCALDAOriginal, LTE, threshold=0.5)
        print("The LDA accuracy is %.3f" % accuracyLDA)

        # for lambd in [10**-6, 10**-5, 10**-4, 10**-1, 1, 10]:
        #     threshold = 0
        #
        #     accuracyLR = compute_LR_accuracy(DTRNormalizedPCAOriginal, LTR, DTENormalizedPCAOriginal, LTE,
        #                                      classPriorProbabilities,
        #                                      lambd, threshold)
        #     print("The LR accuracy (lambda=%f) is %.3f" % (lambd, accuracyLR))
        #
        #     accuracyQLR = compute_QLR_accuracy(DTRNormalizedPCAOriginal, LTR, DTENormalizedPCAOriginal, LTE,
        #                                        classPriorProbabilities,
        #                                        lambd, threshold)
        #     print("The QLR accuracy (lambda=%f) is %.3f" % (lambd, accuracyQLR))
        # print("\n")
        1

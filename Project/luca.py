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
    print("The TNB accuracy is %.3f\n" % accuracyTNB)

    # WITH PCA I CAN TRY TO REDUCE THE DIMENSION OF THE FEATURE SPACE
    # ACTUALLY OUR FEATURE SPACE IS 10. SO PCA CAN TRY TO CREATE A SUBSPACE WHOSE DIMENSION
    # IS IN THIS RANGE (1, 9)rns the varian
    for subDimensionPCA in range(1, DTROriginal.shape[0]):
        DTRNormalizedPCAOriginal, P = compute_PCA(DTROriginalNormalized, subDimensionPCA)
        # Come dovrei implementare la proiezione di DTE?
        DTENormalizedPCAOriginal = numpy.dot(P.T, DTEOriginalNormalized)
        # DTENormalizedPCAOriginal, P = compute_PCA(DTEOriginalNormalized, subDimensionPCA)
        DTRNormalizedPCAOriginalFilteredTrue, DTRNormalizedPCAOriginalFilteredFalse = filter_dataset_by_labels(DTRNormalizedPCAOriginal, LTR)
        #plot_scatter_attributes_X_label(DTRNormalizedPCAOriginalFilteredTrue, title="DTrue PCA")
        pairs.clear()
        #plot_scatter_attributes_X_label(DTRNormalizedPCAOriginalFilteredFalse, title="DFalse PCA")
        pairs.clear()

        ### HERE I COMPUTE LDA ###
        DTRNormalizedPCALDAOriginal, W = compute_LDA_generalized_eigenvalue(DTRNormalizedPCAOriginal, LTR, 1, labels)
        DTENormalizedPCALDAOriginal = numpy.dot(W.T, DTENormalizedPCAOriginal)
        DTRNormalizedPCALDAOriginalFilteredTrue, DTRNormalizedPCALDAOriginalFilteredFalse = filter_dataset_by_labels(
        DTRNormalizedPCALDAOriginal, LTR)
        #plot_scatter_attributes_X_label(DTRNormalizedPCALDAOriginalFilteredTrue, title="DTrue LDA")
        pairs.clear()
        #plot_scatter_attributes_X_label(DTRNormalizedPCALDAOriginalFilteredFalse, title="DFalse LDA")

        print(f"PCA with {subDimensionPCA} dimension + LDA")

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
        print("The LDA accuracy is %.3f\n" % accuracyLDA)

        1

import numpy
from functions import *
from functions2 import *
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
if __name__ == "__main__":


    DTRSplitted, LTR, DTROriginal=read_file("Train.txt")
    DTESplitted, LTE, DTEOriginal=read_file("Test.txt")
    classPriorProbabilities = numpy.array([9/10, 1/10], dtype=float)
    costs = numpy.array([1.0, 1.0], dtype=float)
    labels = [i for i in range(0, numpy.amax(LTR) + 1)]

    DTROriginalNormalized = (DTROriginal - compute_mean(DTROriginal))/ to_column(DTROriginal.std(axis=1))
    DTEOriginalNormalized = (DTEOriginal - compute_mean(DTEOriginal)) / to_column(DTEOriginal.std(axis=1))

    DTROriginalNormalizedGender = numpy.zeros((2, DTROriginalNormalized.shape[1]))
    DTROriginalNormalizedGender[0, :] = to_row(DTROriginalNormalized[1, :])
    DTROriginalNormalizedGender[1, :] = to_row(DTROriginalNormalized[2, :])
    gmm = GaussianMixture(max_iter=5000,n_components=2, random_state=0)
    gmm.fit(DTROriginalNormalizedGender.T)
    LTRGender = gmm.predict(DTROriginalNormalizedGender.T)
    indexLTRGenderTrue, indexLTRGenderFalse = compute_binary_index_array(LTRGender)

    DTRNormalizedOriginalFilteredGenderTrue, DTRNormalizedOriginalFilteredGenderFalse = filter_dataset_by_labels(
                                                                                DTROriginalNormalized, LTRGender)
    #plot_scatter_attributes_X_label_True_False(DTRNormalizedOriginalFilteredTrue, DTRNormalizedOriginalFilteredFalse,
                                               #filepath="./Gender/", title="DTROriginalNormalizedGender")

    DTEOriginalNormalizedGender = numpy.zeros((2, DTEOriginalNormalized.shape[1]))
    DTEOriginalNormalizedGender[0, :] = to_row(DTEOriginalNormalized[1, :])
    DTEOriginalNormalizedGender[1, :] = to_row(DTEOriginalNormalized[2, :])

    gmm.fit(DTEOriginalNormalizedGender.T)

    LTEGender = gmm.predict(DTEOriginalNormalizedGender.T)
    indexLTEGenderTrue, indexLTEGenderFalse = compute_binary_index_array(LTEGender)

    DTENormalizedOriginalFilteredGenderTrue, DTENormalizedOriginalFilteredGenderFalse = filter_dataset_by_labels(
        DTEOriginalNormalized, LTEGender)
    # plot_scatter_attributes_X_label_True_False(DTENormalizedOriginalFilteredGenderTrue,
    #                                            DTENormalizedOriginalFilteredGenderFalse,
    #                                            filepath="./GenderTest/",
    #                                            title="DTEOriginalNormalizedGender")
    print(f"No Pre-Processing")

    accuracyMVG = compute_MVG_accuracy(DTRNormalizedOriginalFilteredGenderTrue, LTR[indexLTRGenderTrue],
                                       DTENormalizedOriginalFilteredGenderTrue, LTE[indexLTEGenderTrue], labels,
                                       classPriorProbabilities)
    print("The MVG accuracy for label True is %.3f" % accuracyMVG)

    accuracyNB = compute_NB_accuracy(DTRNormalizedOriginalFilteredGenderTrue, LTR[indexLTRGenderTrue],
                                       DTENormalizedOriginalFilteredGenderTrue, LTE[indexLTEGenderTrue], labels,
                                       classPriorProbabilities)
    print("The NB accuracy for label True is %.3f" % accuracyNB)

    accuracyTC = compute_TC_accuracy(DTRNormalizedOriginalFilteredGenderTrue, LTR[indexLTRGenderTrue],
                                       DTENormalizedOriginalFilteredGenderTrue, LTE[indexLTEGenderTrue], labels,
                                       classPriorProbabilities)
    print("The TC accuracy for label True is %.3f" % accuracyTC)

    accuracyTNB = compute_TNB_accuracy(DTRNormalizedOriginalFilteredGenderTrue, LTR[indexLTRGenderTrue],
                                       DTENormalizedOriginalFilteredGenderTrue, LTE[indexLTEGenderTrue], labels,
                                       classPriorProbabilities)
    print("The TNB accuracy for label True is %.3f\n" % accuracyTNB)

    accuracyMVG = compute_MVG_accuracy(DTRNormalizedOriginalFilteredGenderFalse, LTR[indexLTRGenderFalse],
                                       DTENormalizedOriginalFilteredGenderFalse, LTE[indexLTEGenderFalse], labels,
                                       classPriorProbabilities)
    print("The MVG accuracy for label False is %.3f" % accuracyMVG)

    accuracyNB = compute_NB_accuracy(DTRNormalizedOriginalFilteredGenderFalse, LTR[indexLTRGenderFalse],
                                       DTENormalizedOriginalFilteredGenderFalse, LTE[indexLTEGenderFalse], labels,
                                       classPriorProbabilities)
    print("The NB accuracy for label False is %.3f" % accuracyNB)

    accuracyTC = compute_TC_accuracy(DTRNormalizedOriginalFilteredGenderFalse, LTR[indexLTRGenderFalse],
                                       DTENormalizedOriginalFilteredGenderFalse, LTE[indexLTEGenderFalse], labels,
                                       classPriorProbabilities)
    print("The TC accuracy for label False is %.3f" % accuracyTC)

    accuracyTNB = compute_TNB_accuracy(DTRNormalizedOriginalFilteredGenderFalse, LTR[indexLTRGenderFalse],
                                       DTENormalizedOriginalFilteredGenderFalse, LTE[indexLTEGenderFalse], labels,
                                       classPriorProbabilities)
    print("The TNB accuracy for label False is %.3f\n" % accuracyTNB)


    for subDimensionPCA in range(1, DTROriginal.shape[0]):
        DTRNormalizedPCAOriginal, P = compute_PCA(DTRNormalizedOriginalFilteredGenderTrue, subDimensionPCA)
        # Come dovrei implementare la proiezione di DTE?
        DTENormalizedPCAOriginal = numpy.dot(P.T, DTENormalizedOriginalFilteredGenderTrue)

        DTRNormalizedPCALDAOriginal, W = compute_LDA_generalized_eigenvalue(DTRNormalizedPCAOriginal, LTR[indexLTRGenderTrue], 1, labels)
        DTENormalizedPCALDAOriginal = numpy.dot(W.T, DTENormalizedPCAOriginal)
        DTRNormalizedPCALDAOriginalFilteredTrue, DTRNormalizedPCALDAOriginalFilteredFalse = filter_dataset_by_labels(
            DTRNormalizedPCALDAOriginal, LTR[indexLTRGenderTrue])

        print(f"PCA with {subDimensionPCA} dimension")

        accuracyMVG = compute_MVG_accuracy(DTRNormalizedPCAOriginal, LTR[indexLTRGenderTrue], DTENormalizedPCAOriginal, LTE[indexLTEGenderTrue], labels,
                                           classPriorProbabilities)
        print("The MVG accuracy is %.3f" % accuracyMVG)

        accuracyNB = compute_NB_accuracy(DTRNormalizedPCAOriginal, LTR[indexLTRGenderTrue], DTENormalizedPCAOriginal, LTE[indexLTEGenderTrue], labels,
                                         classPriorProbabilities)
        print("The NB accuracy is %.3f" % accuracyNB)

        accuracyTC = compute_TC_accuracy(DTRNormalizedPCAOriginal, LTR[indexLTRGenderTrue], DTENormalizedPCAOriginal, LTE[indexLTEGenderTrue], labels,
                                         classPriorProbabilities)
        print("The TC accuracy is %.3f" % accuracyTC)

        accuracyTNB = compute_TNB_accuracy(DTRNormalizedPCAOriginal, LTR[indexLTRGenderTrue], DTENormalizedPCAOriginal, LTE[indexLTEGenderTrue], labels,
                                           classPriorProbabilities)
        print("The TNB accuracy is %.3f" % accuracyTNB)

        accuracyLDA = compute_binary_LDA_accuracy(DTENormalizedPCALDAOriginal, LTE[indexLTEGenderTrue], threshold=0.5)
        print("The LDA accuracy is %.3f\n" % accuracyLDA)

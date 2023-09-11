import matplotlib.pyplot as plt
from functions2 import *
from functions import *
import seaborn

if __name__ == "__main__":

    DTRSplitted, LTR, DTROriginal = read_file("Train.txt")
    DTESplitted, LTE, DTEOriginal = read_file("Test.txt")
    classPriorProbabilities = numpy.array([9 / 10, 1 / 10], dtype=float)
    costs = numpy.array([1.0, 1.0], dtype=float)
    labels = [i for i in range(0, numpy.amax(LTR) + 1)]

    DTROriginalNormalized = (DTROriginal - compute_mean(DTROriginal)) / to_column(DTROriginal.std(axis=1))
    DTEOriginalNormalized = (DTEOriginal - compute_mean(DTEOriginal)) / to_column(DTEOriginal.std(axis=1))

    DTRNormalizedOriginalFilteredTrue, DTRNormalizedOriginalFilteredFalse = filter_dataset_by_labels(
        DTROriginalNormalized, LTR)
    DTENormalizedOriginalFilteredTrue, DTENormalizedOriginalFilteredFalse = filter_dataset_by_labels(
        DTEOriginalNormalized, LTE)
    DTROriginalFilteredTrue, DTROriginalFilteredFalse = filter_dataset_by_labels(DTROriginal, LTR)

    # print("In the Training Dataset there are :")
    # print("\t-%d samples of class 0" % DTRNormalizedOriginalFilteredFalse.shape[1])
    # print("\t-%d samples of class 1" % DTRNormalizedOriginalFilteredTrue.shape[1])
    #
    # print("In the Testing Dataset there are :")
    # print("\t-%d samples of class 0" % DTENormalizedOriginalFilteredFalse.shape[1])
    # print("\t-%d samples of class 1" % DTENormalizedOriginalFilteredTrue.shape[1])
    #
    # plot_scatter_attributes_X_label_True_False(DTRNormalizedOriginalFilteredTrue, DTRNormalizedOriginalFilteredFalse,
    #                                            filepath="./FeatureCorrelation/", title="Scatter")
    #
    # plot_hist_attributes_X_label_binary(DTRNormalizedOriginalFilteredTrue, DTRNormalizedOriginalFilteredFalse,
    #                                     filepath="./NormalizedFeaturesValues/", title="Hist")
    #
    #
    pearsonCorrCoeff = numpy.corrcoef(DTROriginalNormalized)
    plt.figure()
    seaborn.heatmap(pearsonCorrCoeff, cmap="Greys", annot=False)
    plt.xlabel("PCC")
    plt.savefig("./CorrelationCoefficient/PearsonCorrelationCoefficient")
    plt.clf()
    plt.close()

    pearsonCorrCoeff = numpy.corrcoef(DTRNormalizedOriginalFilteredTrue)
    plt.figure()
    seaborn.heatmap(pearsonCorrCoeff, cmap="Blues", annot=False)
    plt.xlabel("PCC True Samples")
    plt.savefig("./CorrelationCoefficient/PearsonCorrelationCoefficientTrue")
    plt.clf()
    plt.close()

    pearsonCorrCoeff = numpy.corrcoef(DTRNormalizedOriginalFilteredFalse)
    plt.figure()
    seaborn.heatmap(pearsonCorrCoeff, cmap="Oranges", annot=False)
    plt.xlabel("PCC False Samples")
    plt.savefig("./CorrelationCoefficient/PearsonCorrelationCoefficientFalse")
    plt.clf()
    plt.close()
    #
    #
    # ### PCA
    #
    # for subDimensionPCA in range(6, DTROriginal.shape[0]):
    #     DTRNormalizedPCAOriginal, P = compute_PCA(DTROriginalNormalized, subDimensionPCA)
    #     DTENormalizedPCAOriginal = numpy.dot(P.T, DTEOriginalNormalized)
    #     DTRNormalizedPCAOriginalFilteredTrue, DTRNormalizedPCAOriginalFilteredFalse = filter_dataset_by_labels(
    #         DTRNormalizedPCAOriginal, LTR)
    #     plot_scatter_attributes_X_label(DTRNormalizedPCAOriginalFilteredTrue,
    #                                     filepath="./FeaturesCorrelationPCA/", title="DTrue PCA")
    #
    #     plot_scatter_attributes_X_label(DTRNormalizedPCAOriginalFilteredFalse,
    #                                     filepath="./FeaturesCorrelationPCA/", title="DFalse PCA")
    #
    # plot_fraction_explained_variance_pca(DTROriginalNormalized)
    #
    #
    #
    # ## LDA

    # DTRNormalizedLDAOriginal, W = compute_LDA_generalized_eigenvalue(DTROriginalNormalized, LTR, len(labels) - 1,
    #                                                                  labels)
    # DTRNormalizedLDAOriginalTrue, DTRNormalizedLDAOriginalFalse = filter_dataset_by_labels(
    #     DTRNormalizedLDAOriginal, LTR)
    # for i in range(DTRNormalizedLDAOriginalTrue.shape[0]):
    #     plt.figure()
    #     plt.hist(DTRNormalizedLDAOriginalTrue[i, :], bins=10, ec='black', density=True, alpha=0.3, label='DTrue')
    #     plt.hist(DTRNormalizedLDAOriginalFalse[i, :], bins=10, ec='black', density=True, alpha=0.3, label='DFalse')
    #     plt.legend()
    #     plt.ylim([0, 0.5])
    #     plt.xlabel("LDA")
    #     plt.savefig("./LDA/LDAOverview")
    #     plt.clf()

    ## PCA
    # for subDimensionPCA in range(2, 3):
    #     DTRNormalizedPCAOriginal, P = compute_PCA(DTROriginalNormalized, subDimensionPCA)
    #     DTENormalizedPCAOriginal = numpy.dot(P.T, DTEOriginalNormalized)
    #     DTRNormalizedPCAOriginalFilteredTrue, DTRNormalizedPCAOriginalFilteredFalse = filter_dataset_by_labels(
    #         DTRNormalizedPCAOriginal, LTR)
    #     plt.figure()
    #     plt.scatter(DTRNormalizedPCAOriginalFilteredTrue[0, :], DTRNormalizedPCAOriginalFilteredTrue[1, :], label='DTrue')
    #     plt.scatter(DTRNormalizedPCAOriginalFilteredFalse[0, :], DTRNormalizedPCAOriginalFilteredFalse[1, :],
    #                 label='DFalse')
    #     plt.legend()
    #     plt.xlabel("PCA")
    #     plt.savefig("./PCA/PCAOverview")
    #     plt.clf()

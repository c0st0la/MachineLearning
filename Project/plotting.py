import matplotlib.pyplot as plt
from functions2 import *
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


    print("In the Training Dataset there are :")
    print("\t-%d samples of class 0" % DTRNormalizedOriginalFilteredFalse.shape[1])
    print("\t-%d samples of class 1" % DTRNormalizedOriginalFilteredTrue.shape[1])

    print("In the Testing Dataset there are :")
    print("\t-%d samples of class 0" % DTENormalizedOriginalFilteredFalse.shape[1])
    print("\t-%d samples of class 1" % DTENormalizedOriginalFilteredTrue.shape[1])

    plot_scatter_attributes_X_label_True_False(DTRNormalizedOriginalFilteredTrue, DTRNormalizedOriginalFilteredFalse,
                                               filepath="./FeatureCorrelation/", title="Scatter")

    plot_hist_attributes_X_label_binary(DTRNormalizedOriginalFilteredTrue, DTRNormalizedOriginalFilteredFalse,
                                        filepath="./NormalizedFeaturesValues/", title="Hist")


    pearsonCorrCoeff = numpy.corrcoef(DTROriginalNormalized)
    plt.figure()
    seaborn.heatmap(pearsonCorrCoeff, cmap="Blues", annot=False)
    plt.savefig("Pearson Correlation Coefficient")
    plt.clf()

    pearsonCorrCoeff = numpy.corrcoef(DTRNormalizedOriginalFilteredTrue)
    plt.figure()
    seaborn.heatmap(pearsonCorrCoeff, cmap="Blues", annot=False)
    plt.savefig("Pearson Correlation Coefficient True")
    plt.clf()

    pearsonCorrCoeff = numpy.corrcoef(DTRNormalizedOriginalFilteredFalse)
    plt.figure()
    seaborn.heatmap(pearsonCorrCoeff, cmap="Blues", annot=False)
    plt.savefig("Pearson Correlation Coefficient False")
    plt.clf()

    for subDimensionPCA in range(6, DTROriginal.shape[0]):
        DTRNormalizedPCAOriginal, P = compute_PCA(DTROriginalNormalized, subDimensionPCA)
        DTENormalizedPCAOriginal = numpy.dot(P.T, DTEOriginalNormalized)
        DTRNormalizedPCAOriginalFilteredTrue, DTRNormalizedPCAOriginalFilteredFalse = filter_dataset_by_labels(
            DTRNormalizedPCAOriginal, LTR)
        plot_scatter_attributes_X_label(DTRNormalizedPCAOriginalFilteredTrue,
                                        filepath="./FeaturesCorrelationPCA/", title="DTrue PCA")

        plot_scatter_attributes_X_label(DTRNormalizedPCAOriginalFilteredFalse,
                                        filepath="./FeaturesCorrelationPCA/", title="DFalse PCA")


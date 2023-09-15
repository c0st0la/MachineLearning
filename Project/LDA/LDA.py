from Project import functions
import numpy

if __name__=="__main__":
    DTRSplitted, LTR, DTROriginal = functions.read_file("../Train.txt")
    DTESplitted, LTE, DTEOriginal = functions.read_file("../Test.txt")
    classPriorProbabilities1 = numpy.array([9 / 10, 1 / 10], dtype=float)
    classPriorProbabilities2 = numpy.array([5 / 10, 5 / 10], dtype=float)
    classPriorProbabilities3 = numpy.array([1 / 10, 9 / 10], dtype=float)
    costs = numpy.array([1.0, 1.0], dtype=float)
    labels = [i for i in range(0, numpy.amax(LTR) + 1)]
    numFold = 5
    thresholds = [i for i in numpy.arange(-30, 30, 0.1)]

    DTROriginalNormalized = (DTROriginal - functions.compute_mean(DTROriginal)) / functions.to_column(
        DTROriginal.std(axis=1))
    DTEOriginalNormalized = (DTEOriginal - functions.compute_mean(DTEOriginal)) / functions.to_column(
        DTEOriginal.std(axis=1))

    DOriginalNormalized = numpy.concatenate((DTROriginalNormalized, DTEOriginalNormalized), axis=1)
    DOriginal = numpy.concatenate((DTROriginal, DTEOriginal), axis=1)
    L = numpy.concatenate((LTR, LTE), axis=0)

    DTRNormalizedLDAOriginal, W = functions.compute_LDA_generalized_eigenvalue(DTROriginalNormalized, LTR, len(labels) - 1,
                                                                            labels)
    DTRNormalizedLDAOriginalTrue, DTRNormalizedLDAOriginalFalse = functions.filter_dataset_by_labels(
        DTRNormalizedLDAOriginal, LTR)
    functions.plot_hist_attributes_X_label_binary(DTRNormalizedLDAOriginalTrue, DTRNormalizedLDAOriginalFalse,
                                        filepath="./FeatureCorrelationLDA", title="HistLDA")
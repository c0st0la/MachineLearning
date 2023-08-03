import numpy
from functions import *
from functions2 import *

if __name__ == "__main__":
    DTRSplitted, LTR, DTROriginal = read_file("Train.txt")
    DTESplitted, LTE, DTEOriginal = read_file("Test.txt")
    classPriorProbabilities1 = numpy.array([9 / 10, 1 / 10], dtype=float)
    classPriorProbabilities2 = numpy.array([5 / 10, 5 / 10], dtype=float)
    classPriorProbabilities3 = numpy.array([1 / 10, 9 / 10], dtype=float)
    costs = numpy.array([1.0, 1.0], dtype=float)
    labels = [i for i in range(0, numpy.amax(LTR) + 1)]
    numFold = 5
    thresholds = [i for i in numpy.arange(-30, 30, 0.1)]

    DTROriginalNormalized = (DTROriginal - compute_mean(DTROriginal)) / to_column(DTROriginal.std(axis=1))
    DTEOriginalNormalized = (DTEOriginal - compute_mean(DTEOriginal)) / to_column(DTEOriginal.std(axis=1))

    DTRNormalizedOriginalFilteredTrue, DTRNormalizedOriginalFilteredFalse = filter_dataset_by_labels(DTROriginalNormalized, LTR)
    DTROriginalNormalizedModified = numpy.delete(DTROriginalNormalized, [1, 3, 6, 8], axis=0)
    DTEOriginalNormalizedModified = numpy.delete(DTEOriginalNormalized, [1, 3, 6, 8], axis=0)

    dict1 = {1e-06: 0.9902173913043478, 1e-05: 0.9902173913043478, 0.0001: 0.988768115942029, 0.001: 0.988768115942029, 0.01: 0.988768115942029, 0.1: 0.988768115942029, 1: 0.988768115942029, 10: 0.988768115942029, 100: 0.988768115942029, 1000: 0.988768115942029, 10000: 0.988768115942029, 100000: 0.988768115942029}
    dict2 = {1e-06: 0.8696090160741805, 1e-05: 0.8696090160741805, 0.0001: 0.8689575502761351, 0.001: 0.8686724303848784, 0.01: 0.8686724303848784, 0.1: 0.8686724303848784, 1: 0.8686724303848784, 10: 0.8686724303848784, 100: 0.8686724303848784, 1000: 0.8686724303848784, 10000: 0.8686724303848784, 100000: 0.8686724303848784}
    dict3 = {1e-06: 0.9902173913043478, 1e-05: 0.9902173913043478, 0.0001: 0.988768115942029, 0.001: 0.988768115942029, 0.01: 0.988768115942029, 0.1: 0.988768115942029, 1: 0.988768115942029, 10: 0.988768115942029, 100: 0.988768115942029, 1000: 0.988768115942029, 10000: 0.988768115942029, 100000: 0.988768115942029}

    min = 100000
    lambd = 0
    print("dict1")
    for key in list(dict1.keys()):
        if dict1[key]< min:
            min = dict1[key]
            lambd = key
    print("lambda", lambd)
    print("minDCF", min)

    min = 100000
    lambd = 0
    print("dict2")
    for key in list(dict2.keys()):
        if dict2[key] < min:
            min = dict2[key]
            lambd = key
    print("lambda", lambd)
    print("minDCF", min)

    min = 100000
    lambd = 0
    print("dict3")
    for key in list(dict3.keys()):
        if dict3[key] < min:
            min = dict3[key]
            lambd = key
    print("lambda", lambd)
    print("minDCF", min)
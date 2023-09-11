import matplotlib.pyplot as plt
from Project import functions
import numpy

if __name__ == "__main__":

    DTRSplitted, LTR, DTROriginal = functions.read_file("../Train.txt")
    DTESplitted, LTE, DTEOriginal = functions.read_file("../Test.txt")
    classPriorProbabilities1 = numpy.array([9 / 10, 1 / 10], dtype=float)
    classPriorProbabilities2 = numpy.array([5 / 10, 5 / 10], dtype=float)
    classPriorProbabilities3 = numpy.array([1 / 10, 9 / 10], dtype=float)
    costs = numpy.array([1.0, 1.0], dtype=float)
    labels = [i for i in range(0, numpy.amax(LTR) + 1)]
    numFold = 5
    thresholds = [i for i in numpy.arange(-30, 30, 0.1)]
    lambdValues = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10, 10 ** 2, 10 ** 3, 10 ** 4,
                   10 ** 5]
    keys = ['10^-5', '10^-4', '10^-3', '10^-2', '10^-1', '1', '10^1', '10^2', '10^3', '10^4', '10^5']

    DTROriginalNormalized = (DTROriginal - functions.compute_mean(DTROriginal)) / functions.to_column(
        DTROriginal.std(axis=1))
    DTEOriginalNormalized = (DTEOriginal - functions.compute_mean(DTEOriginal)) / functions.to_column(
        DTEOriginal.std(axis=1))

    DOriginalNormalized = numpy.concatenate((DTROriginalNormalized, DTEOriginalNormalized), axis=1)
    DOriginal = numpy.concatenate((DTROriginal, DTEOriginal), axis=1)
    L = numpy.concatenate((LTR, LTE), axis=0)

    dict1 = functions.K_fold_cross_validation_DCF(DTROriginalNormalized, L, "QLR",
                                                  numFold, classPriorProbabilities1, costs, labels, lambdValues)

    dict2 = functions.K_fold_cross_validation_DCF(DTROriginalNormalized, L, "QLR",
                                                  numFold, classPriorProbabilities2, costs, labels, lambdValues)

    dict3 = functions.K_fold_cross_validation_DCF(DTROriginalNormalized, L, "QLR",
                                                  numFold, classPriorProbabilities3, costs, labels, lambdValues)

    dict1 = {keys[i]: list(dict1.values())[i] for i in range(len(list(dict1.keys())))}
    dict2 = {keys[i]: list(dict2.values())[i] for i in range(len(list(dict2.keys())))}
    dict3 = {keys[i]: list(dict3.values())[i] for i in range(len(list(dict3.keys())))}

    plt.figure()
    plt.plot([var for var in list(dict1.keys())], [var for var in list(dict1.values())],
             label=f"minDCF(effPrior={classPriorProbabilities1[1]})")
    plt.plot([var for var in list(dict2.keys())], [var for var in list(dict2.values())],
             label=f"minDCF(effPrior={classPriorProbabilities2[1]})")
    plt.plot([var for var in list(dict3.keys())], [var for var in list(dict3.values())],
             label=f"minDCF(effPrior={classPriorProbabilities3[1]})")
    plt.tick_params(axis='x', labelsize=6)
    plt.legend()
    plt.savefig("./figures/QLR_Zscore_DCFxLambda")
    plt.clf()

    print(dict1)
    print(dict2)
    print(dict3)

    print("dict1")
    print("minDCF: ", dict1[min(dict1, key=dict1.get)])

    print("dict2")
    print("minDCF: ", dict2[min(dict2, key=dict2.get)])

    print("dict3")
    print("minDCF: ", dict3[min(dict3, key=dict3.get)])

    with open("./dati/datiQLR_Zscore.txt", "w") as fp:
        fp.write("dict1 :\n")
        fp.write(str(dict1))
        fp.write('\n')
        fp.write("minDCF: " + str(dict1[min(dict1, key=dict3.get)]))
        fp.write('\n')
        fp.write("dict2 :")
        fp.write(str(dict2))
        fp.write('\n')
        fp.write("minDCF: " + str(dict2[min(dict2, key=dict2.get)]))
        fp.write('\n')
        fp.write("dict3 :")
        fp.write(str(dict3))
        fp.write('\n')
        fp.write("minDCF: " + str(dict3[min(dict3, key=dict3.get)]))

    ## RAW FEATURES

    dict1 = functions.K_fold_cross_validation_DCF(DTROriginal, L, "QLR",
                                                  numFold, classPriorProbabilities1, costs, labels, lambdValues)

    dict2 = functions.K_fold_cross_validation_DCF(DTROriginal, L, "QLR",
                                                  numFold, classPriorProbabilities2, costs, labels, lambdValues)

    dict3 = functions.K_fold_cross_validation_DCF(DTROriginal, L, "QLR",
                                                  numFold, classPriorProbabilities3, costs, labels, lambdValues)

    dict1 = {keys[i]: list(dict1.values())[i] for i in range(len(list(dict1.keys())))}
    dict2 = {keys[i]: list(dict2.values())[i] for i in range(len(list(dict2.keys())))}
    dict3 = {keys[i]: list(dict3.values())[i] for i in range(len(list(dict3.keys())))}

    plt.figure()
    plt.plot([var for var in list(dict1.keys())], [var for var in list(dict1.values())],
             label=f"minDCF(effPrior={classPriorProbabilities1[1]})")
    plt.plot([var for var in list(dict2.keys())], [var for var in list(dict2.values())],
             label=f"minDCF(effPrior={classPriorProbabilities2[1]})")
    plt.plot([var for var in list(dict3.keys())], [var for var in list(dict3.values())],
             label=f"minDCF(effPrior={classPriorProbabilities3[1]})")
    plt.tick_params(axis='x', labelsize=6)
    plt.legend()
    plt.savefig("./figures/QLR_Raw_DCFxLambda")
    plt.clf()

    print(dict1)
    print(dict2)
    print(dict3)

    print("dict1")
    print("minDCF: ", dict1[min(dict1, key=dict1.get)])

    print("dict2")
    print("minDCF: ", dict2[min(dict2, key=dict2.get)])

    print("dict3")
    print("minDCF: ", dict3[min(dict3, key=dict3.get)])

    with open("./dati/datiQLR_Raw.txt", "w") as fp:
        fp.write("dict1 :\n")
        fp.write(str(dict1))
        fp.write('\n')
        fp.write("minDCF: " + str(dict1[min(dict1, key=dict3.get)]))
        fp.write('\n')
        fp.write("dict2 :")
        fp.write(str(dict2))
        fp.write('\n')
        fp.write("minDCF: " + str(dict2[min(dict2, key=dict2.get)]))
        fp.write('\n')
        fp.write("dict3 :")
        fp.write(str(dict3))
        fp.write('\n')
        fp.write("minDCF: " + str(dict3[min(dict3, key=dict3.get)]))

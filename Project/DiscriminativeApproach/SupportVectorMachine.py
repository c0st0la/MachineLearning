import matplotlib.pyplot as plt
from Project import functions
import seaborn
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

    DTROriginalNormalized = (DTROriginal - functions.compute_mean(DTROriginal)) / functions.to_column(
        DTROriginal.std(axis=1))
    DTEOriginalNormalized = (DTEOriginal - functions.compute_mean(DTEOriginal)) / functions.to_column(
        DTEOriginal.std(axis=1))
    K = 1.0

    dict1 = functions.K_fold_cross_validation_DCF(DTROriginalNormalized, LTR, "SVM",
                                                  numFold, classPriorProbabilities1, costs, labels, lambdaValues=[], K=K)

    dict2 = functions.K_fold_cross_validation_DCF(DTROriginalNormalized, LTR, "SVM",
                                                  numFold, classPriorProbabilities2, costs, labels, lambdaValues=[], K=K)

    dict3 = functions.K_fold_cross_validation_DCF(DTROriginalNormalized, LTR, "SVM",
                                                  numFold, classPriorProbabilities3, costs, labels, lambdaValues=[], K=K)

    plt.figure()
    plt.plot([var for var in list(dict1.keys())], [var for var in list(dict1.values())],
             label=f"minDCF(effPrior={classPriorProbabilities1[1]})")
    plt.plot([var for var in list(dict2.keys())], [var for var in list(dict2.values())],
             label=f"minDCF(effPrior={classPriorProbabilities2[1]})")
    plt.plot([var for var in list(dict3.keys())], [var for var in list(dict3.values())],
             label=f"minDCF(effPrior={classPriorProbabilities3[1]})")
    plt.tick_params(axis='x', labelsize=6)
    plt.legend()
    plt.savefig("SVM DCF")
    plt.clf()

    with open("SVMdati.txt", "w") as fp:
        fp.write("dict1 :")
        fp.write(str(dict1))
        fp.write('\n')
        key = min(dict1, key=dict1.get)
        fp.write("minDCF: " + str(dict1[key]))
        fp.write('\n')
        fp.write("dict2 :")
        fp.write(str(dict2))
        fp.write('\n')
        key = min(dict1, key=dict2.get)
        fp.write("minDCF: " + str(dict2[key]))
        fp.write('\n')
        fp.write("dict3 :")
        fp.write(str(dict3))
        fp.write('\n')
        key = min(dict1, key=dict2.get)
        fp.write("minDCF: " + str(dict2[key]))
        fp.close()


import matplotlib.pyplot as plt
import numpy
from Project import functions2
from Project.Classifiers import classifiers


if __name__ == "__main__":

    DTRSplitted, LTR, DTROriginal = functions2.read_file("../Train.txt")
    DTESplitted, LTE, DTEOriginal = functions2.read_file("../Test.txt")
    applicationWorkingPoint1 = numpy.array([9 / 10, 1 / 10], dtype=float)
    applicationWorkingPoint2 = numpy.array([5 / 10, 5 / 10], dtype=float)
    applicationWorkingPoint3 = numpy.array([1 / 10, 9 / 10], dtype=float)
    costs = numpy.array([1.0, 1.0], dtype=float)
    labels = [i for i in range(0, numpy.amax(LTR) + 1)]
    numFold = 5
    thresholds = [i for i in numpy.arange(-30, 30, 0.1)]

    DTROriginalNormalized = (DTROriginal - functions2.compute_mean(DTROriginal)) / functions2.to_column(DTROriginal.std(axis=1))
    DTEOriginalNormalized = (DTEOriginal - functions2.compute_mean(DTEOriginal)) / functions2.to_column(DTEOriginal.std(axis=1))

    DOriginalNormalized = numpy.concatenate((DTROriginalNormalized, DTEOriginalNormalized), axis=1)
    DOriginal = numpy.concatenate((DTROriginal, DTEOriginal), axis=1)
    L = numpy.concatenate((LTR, LTE), axis=0)

    # NOW I WILL COMPUTE DCF OF MVG GIVEN A SET OF THRESHOLDS ON THE NORMALIZED DATASET

    DCFsNormalized1 = []
    DCFsNormalized2 = []
    DCFsNormalized3 = []


    kFoldDCFmin1 = classifiers.compute_MVG_KFold_DCF2(DTROriginalNormalized, LTR, numFold, applicationWorkingPoint1, costs, labels)

    kFoldDCFmin2 = classifiers.compute_MVG_KFold_DCF2(DTROriginalNormalized, LTR, numFold, applicationWorkingPoint2, costs, labels)

    kFoldDCFmin3 = classifiers.compute_MVG_KFold_DCF2(DTROriginalNormalized, LTR, numFold, applicationWorkingPoint3, costs, labels)

    toPrint = ""
    toPrint += "Prior probability for class 0 is : " + str(applicationWorkingPoint1[0]) + "\n"
    toPrint += "Prior probability for class 1 is : " + str(applicationWorkingPoint1[1]) + "\n"
    toPrint += "The min of the normalized DCF for MVG classifier with K-fold algo (%d fold): %.3f\n" % (
        numFold, kFoldDCFmin1) + "\n"

    toPrint += "Prior probability for class 0 is : " + str(applicationWorkingPoint2[0]) + "\n"
    toPrint += "Prior probability for class 1 is : " + str(applicationWorkingPoint2[1]) + "\n"

    toPrint += "The min of the normalized DCF for MVG classifier with K-fold algo (%d fold): %.3f\n" % (
        numFold, kFoldDCFmin2) + "\n"

    toPrint += "Prior probability for class 0 is : " + str(applicationWorkingPoint3[0]) + "\n"
    toPrint += "Prior probability for class 1 is : " + str(applicationWorkingPoint3[1]) + "\n"

    toPrint += "The min of the normalized DCF for MVG classifier with K-fold algo (%d fold): %.3f\n" % (
        numFold, kFoldDCFmin3) + "\n"

    with open("datiMVG/datiZscoreMVG.txt", "w") as fp:
        fp.write(toPrint)


    #

    DCFsNormalized1 = []
    DCFsNormalized2 = []
    DCFsNormalized3 = []



    kFoldDCFmin1 = classifiers.compute_MVG_KFold_DCF2(DTROriginal, LTR, numFold, applicationWorkingPoint1,
                                                     costs, labels)

    kFoldDCFmin2 = classifiers.compute_MVG_KFold_DCF2(DTROriginal, LTR, numFold, applicationWorkingPoint2,
                                                     costs, labels)

    kFoldDCFmin3 = classifiers.compute_MVG_KFold_DCF2(DTROriginal, LTR, numFold, applicationWorkingPoint3,
                                                     costs, labels)

    toPrint = ""
    toPrint += "Prior probability for class 0 is : " + str(applicationWorkingPoint1[0]) + "\n"
    toPrint += "Prior probability for class 1 is : " + str(applicationWorkingPoint1[1]) + "\n"

    toPrint += "The min of the normalized DCF for MVG classifier with K-fold algo (%d fold): %.3f\n" % (
        numFold, kFoldDCFmin1) + "\n"

    toPrint += "Prior probability for class 0 is : " + str(applicationWorkingPoint2[0]) + "\n"
    toPrint += "Prior probability for class 1 is : " + str(applicationWorkingPoint2[1]) + "\n"

    toPrint += "The min of the normalized DCF for MVG classifier with K-fold algo (%d fold): %.3f\n" % (
        numFold, kFoldDCFmin2) + "\n"

    toPrint += "Prior probability for class 0 is : " + str(applicationWorkingPoint3[0]) + "\n"
    toPrint += "Prior probability for class 1 is : " + str(applicationWorkingPoint3[1]) + "\n"

    toPrint += "The min of the normalized DCF for MVG classifier with K-fold algo (%d fold): %.3f\n" % (
        numFold, kFoldDCFmin3) + "\n"

    with open("datiMVG/datiRawMVG.txt", "w") as fp:
        fp.write(toPrint)
import numpy
from Project import functions
from Project.Classifiers import classifiers

if __name__ == "__main__":

    DTRSplitted, LTR, DTROriginal = functions.read_file("../Train.txt")
    DTESplitted, LTE, DTEOriginal = functions.read_file("../Test.txt")
    applicationWorkingPoint1 = numpy.array([9 / 10, 1 / 10], dtype=float)
    applicationWorkingPoint2 = numpy.array([5 / 10, 5 / 10], dtype=float)
    applicationWorkingPoint3 = numpy.array([1 / 10, 9 / 10], dtype=float)
    costs = numpy.array([1.0, 1.0], dtype=float)
    labels = [i for i in range(0, numpy.amax(LTR) + 1)]
    numFold = 5
    thresholds = [i for i in numpy.arange(-30, 30, 0.1)]

    DTROriginalNormalized = (DTROriginal - functions.compute_mean(DTROriginal)) / functions.to_column(
        DTROriginal.std(axis=1))
    DTEOriginalNormalized = (DTEOriginal - functions.compute_mean(DTEOriginal)) / functions.to_column(
        DTEOriginal.std(axis=1))

    DOriginalNormalized = numpy.concatenate((DTROriginalNormalized, DTEOriginalNormalized), axis=1)


    DTRNormalizedOriginalFilteredTrue, DTRNormalizedOriginalFilteredFalse = functions.filter_dataset_by_labels(
        DTROriginalNormalized, LTR)

    with open("datiPCA/datiZScorePCAMVG.txt", "w") as fp:
        fp.write("")

    with open("datiPCA/datiRawPCAMVG.txt", "w") as fp:
        fp.write("")

    for subDimensionPCA in range(8, DTROriginal.shape[0]+1):
        toPrint = ""
        toPrint += "PCA with %d dimension" % subDimensionPCA + "\n"
        DTRNormalizedPCAOriginal, P = functions.compute_PCA(DTROriginalNormalized, subDimensionPCA)
        DTENormalizedPCAOriginal = numpy.dot(P.T, DTEOriginalNormalized)


        kFoldDCFmin1 = classifiers.compute_MVG_KFold_DCF2(DTRNormalizedPCAOriginal, LTR, numFold,
                                                         applicationWorkingPoint1,
                                                         costs, labels)

        kFoldDCFmin2 = classifiers.compute_MVG_KFold_DCF2(DTRNormalizedPCAOriginal, LTR, numFold,
                                                         applicationWorkingPoint2,
                                                         costs, labels)

        kFoldDCFmin3 = classifiers.compute_MVG_KFold_DCF2(DTRNormalizedPCAOriginal, LTR, numFold,
                                                         applicationWorkingPoint3,
                                                         costs, labels)

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

        with open("datiPCA/datiZScorePCAMVG.txt", "a") as fp:
            fp.write(toPrint)

        ##

    for subDimensionPCA in range(8, DTROriginal.shape[0]+1):
        toPrint = ""
        toPrint += "PCA with %d dimension" % subDimensionPCA + "\n"
        DTRPCAOriginal, P = functions.compute_PCA(DTROriginal, subDimensionPCA)
        DTEPCAOriginal = numpy.dot(P.T, DTEOriginal)

        kFoldDCFmin1 = classifiers.compute_MVG_KFold_DCF2(DTRPCAOriginal, LTR, numFold,
                                                         applicationWorkingPoint1, costs, labels)

        kFoldDCFmin2 = classifiers.compute_MVG_KFold_DCF2(DTRPCAOriginal, LTR, numFold,
                                                         applicationWorkingPoint2, costs, labels)

        kFoldDCFmin3 = classifiers.compute_MVG_KFold_DCF2(DTRPCAOriginal, LTR, numFold,
                                                         applicationWorkingPoint3, costs, labels)

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

        with open("datiPCA/datiRawPCAMVG.txt", "a") as fp:
            fp.write(toPrint)

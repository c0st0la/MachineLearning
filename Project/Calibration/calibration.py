import matplotlib.pyplot as plt
import numpy
from Project import functions2, functions
from Project.Classifiers import classifiers
from utils import *

def save_bayes_error_plot(scores, L, costs, path):
    x = []
    DCFsNormalized = []
    DCFsNormalized2 = []
    DCFsNormalizedMin = []
    effPriorLogOdds = numpy.linspace(-4, 4, 50)
    for effPriorLogOdd in effPriorLogOdds:
        print(str(effPriorLogOdd))
        x.append(effPriorLogOdd)
        effPrior = 1 / (1 + (numpy.exp(-effPriorLogOdd)))
        classPriorProbability = numpy.array([1 - effPrior, effPrior], dtype=float)
        print(str(classPriorProbability))
        optimalBayesDecisionPredictions = functions2.compute_optimal_bayes_decision(scores,
                                                                                    classPriorProbability, costs)

        confusionMatrix = functions2.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, L)
        DCFsNormalized.append(
            functions2.compute_normalized_detection_cost_function(confusionMatrix, classPriorProbability, costs))
        thresholds = [i for i in numpy.arange(-30, 30, 0.15)]
        for threshold in thresholds:
            optimalBayesDecisionPredictions = functions2.compute_optimal_bayes_decision_given_threshold(scores,
                                                                                                        threshold)
            confusionMatrix = functions2.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, L)
            DCFsNormalized2.append(
                functions2.compute_normalized_detection_cost_function(confusionMatrix, classPriorProbability, costs))
        DCFsNormalizedMin.append(min(DCFsNormalized2))
        DCFsNormalized2 = []
    mass = max(max(DCFsNormalizedMin), max(DCFsNormalized))
    plt.figure()
    plt.plot(effPriorLogOdds, DCFsNormalized, label='actDCF', color='b')
    plt.plot(effPriorLogOdds, DCFsNormalizedMin,':', label='minDCF', color='r')
    plt.xlabel("$log \\frac{ \\tilde{\pi}}{1-\\tilde{\pi}}$")

    plt.legend()
    plt.ylim([0, mass])
    plt.xlim([-4.1, 4.1])
    plt.savefig(("./figures/"+path))
    plt.clf()
    plt.close()



def save_ApplicationWorkingPoint_DCFs(scores, L, costs, path, applicationWorkingPoints):
    DCFsNormalized = []
    DCFsNormalized2 = []
    DCFsNormalizedMin = []
    applicationWorkingPointsPrint = ['0_1', '0_5', '0_9']
    for i, applicationWorkingPoint in enumerate(applicationWorkingPoints):
        optimalBayesDecisionPredictions = functions2.compute_optimal_bayes_decision(scores,
                                                                                    applicationWorkingPoint, costs)

        confusionMatrix = functions2.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, L)
        DCFsNormalized.append(
            functions2.compute_normalized_detection_cost_function(confusionMatrix, applicationWorkingPoint, costs))
        thresholds = [i for i in numpy.arange(-30, 30, 0.15)]
        for threshold in thresholds:
            optimalBayesDecisionPredictions = functions2.compute_optimal_bayes_decision_given_threshold(scores,
                                                                                                        threshold)
            confusionMatrix = functions2.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, L)
            DCFsNormalized2.append(
                functions2.compute_normalized_detection_cost_function(confusionMatrix, applicationWorkingPoint, costs))
        DCFsNormalizedMin.append(min(DCFsNormalized2))
        DCFsNormalized2 = []
        with open("./DCFs/minDCF_" + path + applicationWorkingPointsPrint[i], "w") as fp:
            fp.write(str(min(DCFsNormalizedMin)))
        with open("./DCFs/actDCF_" + path + applicationWorkingPointsPrint[i], "w") as fp:
            fp.write(str(min(DCFsNormalized)))



if __name__ == "__main__":
    DTRSplitted, LTR, DTROriginal = functions2.read_file("../Train.txt")
    DTESplitted, LTE, DTEOriginal = functions2.read_file("../Test.txt")
    applicationWorkingPoint1 = numpy.array([9 / 10, 1 / 10], dtype=float)
    applicationWorkingPoint2 = numpy.array([5 / 10, 5 / 10], dtype=float)
    applicationWorkingPoint3 = numpy.array([1 / 10, 9 / 10], dtype=float)
    applicationWorkingPoints = [[9 / 10, 1 / 10], [5 / 10, 5 / 10], [1 / 10, 9 / 10]]
    applicationWorkingPointsPrint = ['0_1','0_5','0_9']
    costs = numpy.array([1.0, 1.0], dtype=float)
    labels = [i for i in range(0, numpy.amax(LTR) + 1)]
    numFold = 5
    thresholds = [i for i in numpy.arange(-30, 30, 0.1)]

    DTROriginalNormalized = (DTROriginal - functions2.compute_mean(DTROriginal)) / functions2.to_column(
        DTROriginal.std(axis=1))
    DTEOriginalNormalized = (DTEOriginal - functions2.compute_mean(DTEOriginal)) / functions2.to_column(
        DTEOriginal.std(axis=1))

    with open("./Scores/MVGZscoreUncalibrated_pi_0_5", "r") as fp:
        scores_mvg = fp.readline()
        ltr = fp.readline()
    ltr = numpy.array([int(label) for label in ltr.rstrip().split(" ")])
    scores_mvg = numpy.array([float(score) for score in scores_mvg.rstrip().split(" ")])
    with open("./Scores/QLRZscoreUncalibrated_pi_0_5", "r") as fp:
        scores_qlr = fp.readline()
    scores_qlr = numpy.array([float(score) for score in scores_qlr.rstrip().split(" ")])
    with open("./Scores/QSVMZscoreUncalibrated_pi_0_5", "r") as fp:
        scores_qsvm = fp.readline()
    scores_qsvm = numpy.array([float(score) for score in scores_qsvm.rstrip().split(" ")])

    ## MVG zSCORE
    save_bayes_error_plot(scores_mvg, ltr, costs, "MVGZscoreUncalibrated_pi_0_5_")
    save_ApplicationWorkingPoint_DCFs(scores_mvg, ltr, costs, "MVGZscoreUncalibrated_pi_0_5_", applicationWorkingPoints)

    # scores_mvg, ltr = classifiers.compute_MVG_KFold_score(DTROriginalNormalized, LTR, numFold, labels)
    # save_bayes_error_plot(scores_mvg, ltr, costs, "MVGZscoreUncalibrated_pi_"+str(applicationWorkingPointsPrint[1]))
    # with open("./Scores/" + "MVGZscoreUncalibrated_pi_" + str(applicationWorkingPointsPrint[1]), "w") as fp:
    #     fp.write("")
    # with open("./Scores/" + "MVGZscoreUncalibrated_pi_" + str(applicationWorkingPointsPrint[1]), "a") as fp:
    #     for score in scores_mvg.tolist():
    #         fp.write(str(score) + " ")
    #     fp.write("\n")
    #     for label in list(ltr):
    #         fp.write(str(label) + " ")
    print("sto calibrando...")
    calibratedScore = compute_calibration(scores_mvg, ltr, applicationWorkingPoints[0])
    # with open("./Scores/" + "MVGZscoreCalibrated_pi_" + str(applicationWorkingPointsPrint[1]), "w") as fp:
    #     fp.write("")
    # with open("./Scores/" + "MVGZscoreCalibrated_pi_" + str(applicationWorkingPointsPrint[1]), "a") as fp:
    #     for score in list(calibratedScore):
    #         fp.write(str(score) + " ")
    #     fp.write("\n")
    #     for label in list(ltr):
    #         fp.write(str(label) + " ")
    #
    # save_bayes_error_plot(calibratedScore, ltr, costs, "MVGZscoreCalibrated_pi_"+str(applicationWorkingPointsPrint[1]))
    save_bayes_error_plot(calibratedScore, ltr, costs, "MVGZscoreCalibrated_pi_0_5_")
    save_ApplicationWorkingPoint_DCFs(calibratedScore, ltr, costs, "MVGZscoreCalibrated_pi_0_5_", applicationWorkingPoints)





    ## QLR zscore 0_5
    save_bayes_error_plot(scores_qlr, ltr, costs, "QLRZscoreUncalibrated_pi_0_5_")
    save_ApplicationWorkingPoint_DCFs(scores_qlr, ltr, costs, "QLRZscoreUncalibrated_pi_0_5_", applicationWorkingPoints)
    calibratedScore = compute_calibration(scores_qlr, ltr, applicationWorkingPoints[1])
    save_bayes_error_plot(calibratedScore, ltr, costs, "QLRZscoreCalibrated_pi_0_5_")
    save_ApplicationWorkingPoint_DCFs(calibratedScore, ltr, costs, "QLRZscoreCalibrated_pi_0_5_", applicationWorkingPoints)
    # scores_qlr, ltr = classifiers.compute_QLR_KFold_scores(DTROriginalNormalized, LTR, numFold, 10**-5,
    #                                                        applicationWorkingPoints[1])
    # with open("./Scores/" + "QLRZscoreUncalibrated_pi_" + str(applicationWorkingPointsPrint[1]), "w") as fp:
    #     fp.write("")
    # with open("./Scores/" + "QLRZscoreUncalibrated_pi_" + str(applicationWorkingPointsPrint[1]), "a") as fp:
    #     for score in scores_qlr.tolist():
    #         fp.write(str(score) + " ")
    #     fp.write("\n")
    #     for label in list(ltr):
    #         fp.write(str(label) + " ")
    # save_bayes_error_plot(scores_qlr, ltr, costs, "QLRZscoreUncalibrated_pi_" + str(applicationWorkingPointsPrint[1]))
    #
    #
    # print("sto calibrando...")
    # calibratedScore = compute_calibration(scores_qlr, ltr, applicationWorkingPoints[1])
    # with open("./Scores/" + "QLRZscoreCalibrated_pi_" + str(applicationWorkingPointsPrint[1]), "w") as fp:
    #     fp.write("")
    # with open("./Scores/" + "QLRZscoreCalibrated_pi_" + str(applicationWorkingPointsPrint[1]), "a") as fp:
    #     for score in list(calibratedScore):
    #         fp.write(str(score) + " ")
    #     fp.write("\n")
    #     for label in list(ltr):
    #         fp.write(str(label) + " ")
    # save_bayes_error_plot(calibratedScore, ltr, costs, "QLRZscoreCalibrated_pi_"+str(applicationWorkingPointsPrint[1]))
    #
    #
    #
    # ## PolySVM zscore 0_5
    save_bayes_error_plot(scores_qsvm, ltr, costs, "QSVMZscoreUncalibrated_pi_0_5_")
    save_ApplicationWorkingPoint_DCFs(scores_qsvm, ltr, costs, "QSVMZscoreUncalibrated_pi_0_5_", applicationWorkingPoints)
    calibratedScore = compute_calibration(scores_qsvm, ltr, applicationWorkingPoints[1])
    save_bayes_error_plot(calibratedScore, ltr, costs, "QSVMZscoreCalibrated_pi_0_5_")
    save_ApplicationWorkingPoint_DCFs(calibratedScore, ltr, costs, "QSVMZscoreCalibrated_pi_0_5_", applicationWorkingPoints)
    #
    # scores_qsvm, ltr = classifiers.compute_PolySVM_KFold_scores(DTROriginalNormalized, LTR, numFold,
    #                                                             applicationWorkingPoints[1], 10**2, 1, 2, 1)
    # with open("./Scores/" + "QSVMZscoreUncalibrated_pi_" + str(applicationWorkingPointsPrint[1]), "w") as fp:
    #     fp.write("")
    # with open("./Scores/" + "QSVMZscoreUncalibrated_pi_" + str(applicationWorkingPointsPrint[1]), "a") as fp:
    #     for score in scores_qsvm.tolist():
    #         fp.write(str(score) + " ")
    #     fp.write("\n")
    #     for label in list(ltr):
    #         fp.write(str(label) + " ")
    # save_bayes_error_plot(scores_qsvm, ltr, costs, "QSVMZscoreUncalibrated_pi_" + str(applicationWorkingPointsPrint[1]))
    #
    #
    # print("sto calibrando...")
    # calibratedScore = compute_calibration(scores_qsvm, ltr, applicationWorkingPoints[1])
    # with open("./Scores/" + "QSVMZscoreCalibrated_pi_" + str(applicationWorkingPointsPrint[1]), "w") as fp:
    #     fp.write(str(calibratedScore))
    # with open("./Scores/" + "QSVMZscoreCalibrated_pi_" + str(applicationWorkingPointsPrint[1]), "w") as fp:
    #     fp.write("")
    # with open("./Scores/" + "QSVMZscoreCalibrated_pi_" + str(applicationWorkingPointsPrint[1]), "a") as fp:
    #     for score in list(calibratedScore):
    #         fp.write(str(score) + " ")
    #     fp.write("\n")
    #     for label in list(ltr):
    #         fp.write(str(label) + " ")
    # save_bayes_error_plot(calibratedScore, ltr, costs, "QSVMZscoreCalibrated_pi_"+str(applicationWorkingPointsPrint[1]))
    #

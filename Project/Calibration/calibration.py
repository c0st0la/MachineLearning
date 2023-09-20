import matplotlib.pyplot as plt
import numpy
from Project import functions2, functions
from Project.Classifiers import classifiers
from utils import *

def save_bayes_error_plot(scores, L, costs, path):
    x = []
    y = []
    DCFsNormalized = []
    DCFsNormalized2 = []
    DCFsNormalizedMin = []
    effPriorLogOdds = numpy.linspace(-4, 4, 25)
    for effPriorLogOdd in effPriorLogOdds:
        print(str(effPriorLogOdd))
        x.append(effPriorLogOdd)
        effPrior = 1 / (1 + (numpy.exp(-effPriorLogOdd)))
        classPriorProbability = numpy.array([1 - effPrior, effPrior], dtype=float)
        optimalBayesDecisionPredictions = functions2.compute_optimal_bayes_decision(scores,
                                                                                    classPriorProbability, costs)

        confusionMatrix = functions2.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, L)
        DCFsNormalized.append(
            functions2.compute_normalized_detection_cost_function(confusionMatrix, classPriorProbability, costs))
        thresholds = [i for i in numpy.arange(-30, 30, 0.1)]
        for threshold in thresholds:
            optimalBayesDecisionPredictions = functions2.compute_optimal_bayes_decision_given_threshold(scores,
                                                                                                        threshold)
            confusionMatrix = functions2.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, L)
            DCFsNormalized2.append(
                functions2.compute_normalized_detection_cost_function(confusionMatrix, classPriorProbability, costs))
        DCFsNormalizedMin.append(min(DCFsNormalized2))
        DCFsNormalized2 = []
    mass = max(max(DCFsNormalizedMin), max(DCFsNormalized))
    with open("./DCFs/" + path, "w") as fp:
        fp.write("minDCF: "+str(min(DCFsNormalized))+"\n")
        fp.write("actDCF: " + str(min(DCFsNormalized)) + "\n")
    plt.figure()
    plt.plot(effPriorLogOdds, DCFsNormalized, label='actDCF', color='b')
    plt.plot(effPriorLogOdds, DCFsNormalizedMin,':', label='minDCF', color='r')
    plt.xlabel("$log \\frac{ \\tilde{\pi}}{1-\\tilde{\pi}}$")

    plt.legend()
    plt.ylim([0, mass])
    plt.xlim([-3, 3.0])
    plt.savefig(("./figures/"+path))
    plt.clf()
    plt.close()



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


    ## MVG

    scores_mvg, ltr = classifiers.compute_MVG_KFold_score(DTROriginalNormalized, LTR, numFold, labels)
    save_bayes_error_plot(scores_mvg, ltr, costs, "MVG")

    for i, applicationWorkingPoint in enumerate(applicationWorkingPoints):
        print("sto calibrando...")
        calibratedScore = compute_calibration(scores_mvg, ltr, applicationWorkingPoint)
        save_bayes_error_plot(calibratedScore, ltr, costs, "MVG_pi_"+str(applicationWorkingPointsPrint[i]))


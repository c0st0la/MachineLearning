import matplotlib.pyplot as plt
import numpy
from Project import functions2, functions
from Project.Classifiers import classifiers




def train_LRKFold(D, L, lambd, classPriorProbabilities, path, applicationWorkingPoints):
    num_samples = int(D.shape[1] / numFold)
    numpy.random.seed(27)
    perm = numpy.random.permutation(D.shape[1])
    D = D[:, perm]
    L = L[perm]
    scores = numpy.zeros(L.shape[0])
    for i in range(numFold):
        (DTR, LTR), (DTE, LTE) = functions.K_fold_generate_Training_and_Testing_samples(D, L, i, numFold, num_samples)
        scores[i * num_samples: (i + 1) * num_samples] = functions.compute_logistic_regression_binary_llr(DTR, LTR, DTE,
                                                                                        lambd, classPriorProbabilities)
    save_bayes_error_plot(scores, L, costs, path)
    save_ApplicationWorkingPoint_DCFs(scores, L, costs, path, applicationWorkingPoints)



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
    applicationWorkingPointsPrint = ['0_1', '0_5', '0_9']
    costs = numpy.array([1.0, 1.0], dtype=float)
    labels = [i for i in range(0, numpy.amax(LTR) + 1)]
    numFold = 5
    thresholds = [i for i in numpy.arange(-30, 30, 0.1)]

    DTROriginalNormalized = (DTROriginal - functions2.compute_mean(DTROriginal)) / functions2.to_column(
        DTROriginal.std(axis=1))
    DTEOriginalNormalized = (DTEOriginal - functions2.compute_mean(DTEOriginal)) / functions2.to_column(
        DTEOriginal.std(axis=1))

    with open("../Calibration/Scores/MVGZscoreUncalibrated_pi_0_5", "r") as fp:
        scores_mvg = fp.readline()
        ltr = fp.readline()
    ltr = [int(label) for label in ltr.rstrip().split(" ")]
    scores_mvg = [float(score) for score in scores_mvg.rstrip().split(" ")]
    with open("../Calibration/Scores/QLRZscoreUncalibrated_pi_0_5", "r") as fp:
        scores_qlr = fp.readline()
    scores_qlr = [float(score) for score in scores_qlr.rstrip().split(" ")]
    with open("../Calibration/Scores/QSVMZscoreUncalibrated_pi_0_5", "r") as fp:
        scores_qsvm = fp.readline()
    scores_qsvm = [float(score) for score in scores_qsvm.rstrip().split(" ")]

    scores1 = [scores_mvg, scores_qlr, scores_qsvm]

    scores = numpy.array(numpy.vstack(scores1))
    ltr= numpy.array(ltr)
    # scoreCalibrated = compute_calibration(scores, ltr, applicationWorkingPoints[1])
    # save_bayes_error_plot(scoreCalibrated, ltr, costs, "Fusion" + str(applicationWorkingPointsPrint[1]))
    train_LRKFold(scores, ltr, 10**-5, applicationWorkingPoints[1], "MVG_QLR_QSVM_Zscore_pt_"+str(applicationWorkingPointsPrint[1]),
                                                                                               applicationWorkingPoints)

    scores2 = [scores_mvg, scores_qlr]

    scores = numpy.array(numpy.vstack(scores2))
    ltr= numpy.array(ltr)
    # scoreCalibrated = compute_calibration(scores, ltr, applicationWorkingPoints[1])
    # save_bayes_error_plot(scoreCalibrated, ltr, costs, "Fusion" + str(applicationWorkingPointsPrint[1]))
    train_LRKFold(scores, ltr, 10**-5, applicationWorkingPoints[1], "MVG_QLR_Zscore_pt_"+str(applicationWorkingPointsPrint[1]),
                                                                                               applicationWorkingPoints)

    scores3 = [scores_qlr, scores_qsvm]

    scores = numpy.array(numpy.vstack(scores3))
    ltr= numpy.array(ltr)
    # scoreCalibrated = compute_calibration(scores, ltr, applicationWorkingPoints[1])
    # save_bayes_error_plot(scoreCalibrated, ltr, costs, "Fusion" + str(applicationWorkingPointsPrint[1]))
    train_LRKFold(scores, ltr, 10**-5, applicationWorkingPoints[1], "QLR_QSVM_Zscore_pt_"+str(applicationWorkingPointsPrint[1]),
                                                                                               applicationWorkingPoints)

    scores4 = [scores_mvg, scores_qsvm]

    scores = numpy.array(numpy.vstack(scores4))
    ltr= numpy.array(ltr)
    # scoreCalibrated = compute_calibration(scores, ltr, applicationWorkingPoints[1])
    # save_bayes_error_plot(scoreCalibrated, ltr, costs, "Fusion" + str(applicationWorkingPointsPrint[1]))
    train_LRKFold(scores, ltr, 10**-5, applicationWorkingPoints[1], "MVG_QSVM_Zscore_pt_"+str(applicationWorkingPointsPrint[1]),
                                                                                               applicationWorkingPoints)

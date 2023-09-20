import matplotlib.pyplot as plt
import numpy
from Project import functions2


def save_bayes_error_plot(scores, costs, LTE,path):
    x = []
    y = []
    DCFsNormalized = []
    DCFsNormalized2 = []
    DCFsNormalizedMin = []
    effPriorLogOdds = numpy.linspace(-4, 4,50)
    for effPriorLogOdd in effPriorLogOdds:
        x.append(effPriorLogOdd)
        effPrior = 1 / (1 + (numpy.exp(-effPriorLogOdd)))
        classPriorProbability = numpy.array([1 - effPrior, effPrior], dtype=float)
        optimalBayesDecisionPredictions = functions2.compute_optimal_bayes_decision(scores,
                                                                                    classPriorProbability, costs)

        confusionMatrix = functions2.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
        DCFsNormalized.append(
            functions2.compute_normalized_detection_cost_function(confusionMatrix, classPriorProbability, costs))
        thresholds = [i for i in numpy.arange(-30, 30, 0.1)]
        for threshold in thresholds:
            optimalBayesDecisionPredictions = functions2.compute_optimal_bayes_decision_given_threshold(scores,
                                                                                                        threshold)
            confusionMatrix = functions2.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
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
    plt.xlim([-3, 3.0])
    plt.show()
    plt.savefig(("./figures/"+path))


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

    DTROriginalNormalized = (DTROriginal - functions2.compute_mean(DTROriginal)) / functions2.to_column(
        DTROriginal.std(axis=1))
    DTEOriginalNormalized = (DTEOriginal - functions2.compute_mean(DTEOriginal)) / functions2.to_column(
        DTEOriginal.std(axis=1))

    score_mvg = functions2.compute_MVG_llrs(DTROriginalNormalized, LTR, DTEOriginalNormalized, labels)
    save_bayes_error_plot(score_mvg, costs,LTE,"MVG")

    for i in [[0.5, 0.5], [0.1, 0.9], [0.9, 0.1]]:
        scores_1, labels_1 = make_calibration(scores_rbsvm, ltr, i, "RadialBasedSVM_ZSCORE_C0_0.1_C1_0.9_K_1_C_5_g_0.1")
        save_bayes_error_plot(scores_1, costs,LTE,"MVG", labels=labels_1)
        scores_2, labels_2 = make_calibration(scores_svm, ltr, i, "SVM_ZSCORE_C0_0.1_C1_0.9_K_1_C_10.0.npy")
        plot_bayes_error(scores_2, Model.SVM, "Z-Score calibrated", labels=labels_2)
        scores_3, labels_3 = make_calibration(scores_lr, ltr, i, "LR_ZSCORE_C0_0.1_C1_0.9_l_1e-05.npy")
        plot_bayes_error(scores_3, Model.LR, "Z-Score calibrated", labels=labels_3)
        scores_4, labels_4 = make_calibration(scores_gmm, ltr, i, "GMM_ZSCORE_components_4_alpha_0.1_psi_0.01.npy")
        plot_bayes_error(scores_4, Model.GMM, "4 components - Z-Score calibrated", labels=labels_4)
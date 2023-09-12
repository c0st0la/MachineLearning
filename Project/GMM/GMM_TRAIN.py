import itertools
from Project import functions
from Project.Classifiers import classifiers
import numpy
from utils import *
from itertools import compress
from GMM_models import *
from gmmtrain import *



if __name__ == "__main__":
    DTRSplitted, LTR, DTROriginal = functions.read_file("../Train.txt")
    DTESplitted, LTE, DTEOriginal = functions.read_file("../Test.txt")
    classPriorProbabilities1 = numpy.array([9 / 10, 1 / 10], dtype=float)
    classPriorProbabilities2 = numpy.array([5 / 10, 5 / 10], dtype=float)
    classPriorProbabilities3 = numpy.array([1 / 10, 9 / 10], dtype=float)
    costs = numpy.array([1.0, 1.0], dtype=float)
    score=[]
    print("========================================================")
    for iteration in range(5):
        print("========================================================")
        print("========================================================")
        print("Training for GMM Model with " + str(2 ** iteration) + " components for all pre process in report...")
        #train_single_gmm(DTROriginal, LTR, DTEOriginal, LTE, iteration)

        score.append(make_train_with_K_fold(DTROriginal, LTR, DTEOriginal, LTE, iteration))
        print("========================================================")
        print("========================================================")
        print("========================================================")
    thresholds = [i for i in numpy.arange(-30, 30, 0.1)]
    DCFsNormalized1 = []
    totDCF = 0
    minDCF=[]
    for item in score:
        for threshold in thresholds:
            optimalBayesDecisionPredictions = functions.compute_optimal_bayes_decision_given_threshold(item,
                                                                                                       threshold)
            confusionMatrix = functions.compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
            DCFsNormalized1.append(
                functions.compute_normalized_detection_cost_function(confusionMatrix, classPriorProbabilities1, costs))
        minDCF.append(min(DCFsNormalized1))

print("suca")









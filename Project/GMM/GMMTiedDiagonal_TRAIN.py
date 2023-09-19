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
    for iteration in range(7):
        print("========================================================")
        print("========================================================")
        print("Training for GMM Model with " + str(2 ** iteration) + " components for all pre process in report...")


        score.append(GMMTiedDiagonal_train_with_K_fold(DTROriginal, LTR,iteration,"GmmTiedDiagonal"))
        print("========================================================")
        print("========================================================")
        print("========================================================")

print("suca")









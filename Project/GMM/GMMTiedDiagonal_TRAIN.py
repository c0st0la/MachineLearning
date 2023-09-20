import itertools
from Project import functions
from Project.Classifiers import classifiers
import numpy
from utils import *
from itertools import compress
from GMM_models import *
from gmmtrain import *
from Project import functions2



if __name__ == "__main__":
    DTRSplitted, LTR, DTROriginal = functions.read_file("../Train.txt")
    DTESplitted, LTE, DTEOriginal = functions.read_file("../Test.txt")
    DTROriginalNormalized = (DTROriginal - functions2.compute_mean(DTROriginal)) / functions2.to_column(
        DTROriginal.std(axis=1))
    classPriorProbabilities1 = numpy.array([9 / 10, 1 / 10], dtype=float)
    classPriorProbabilities2 = numpy.array([5 / 10, 5 / 10], dtype=float)
    classPriorProbabilities3 = numpy.array([1 / 10, 9 / 10], dtype=float)
    costs = numpy.array([1.0, 1.0], dtype=float)
    score=[]
    print("========================================================")
    # for iteration in range(7):
    #     print("========================================================")
    #     print("========================================================")
    #     print("Training for GMM Model with " + str(2 ** iteration) + " components for all pre process in report...")
    #
    #
    #     score.append(GMMTiedDiagonal_train_with_K_fold(DTROriginalNormalized, LTR,iteration,"GmmTiedDiagonal_Zscore"))
    #     print("========================================================")
    #     print("========================================================")
    #     print("========================================================")
    # for iteration in range(7):
    #     print("========================================================")
    #     print("========================================================")
    #     print("Training for GMM Model with " + str(2 ** iteration) + " components for all pre process in report...")
    #
    #
    #     score.append(GMMTiedDiagonal_train_with_K_fold(DTROriginal, LTR,iteration,"GmmTiedDiagonal"))
    #     print("========================================================")
    #     print("========================================================")
    #     print("========================================================")
    for subDimensionPCA in range(8, DTROriginal.shape[0]+1):
        toPrint = ""
        toPrint += "PCA with %d dimension" % subDimensionPCA + "\n"
        DTRNormalizedPCAOriginal, P = functions.compute_PCA(DTROriginalNormalized, subDimensionPCA)
        for iteration in range(7):
            print("========================================================")
            print("========================================================")
            print("Training for GMM Model with " + str(2 ** iteration) + " components for all pre process in report...")

            string="GmmTiedDiagonalPCA"+str(subDimensionPCA)
            score.append(GMM_train_with_K_fold(DTRNormalizedPCAOriginal, LTR,iteration,string))
            print("========================================================")
            print("========================================================")
            print("========================================================")










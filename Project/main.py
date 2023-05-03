  #10x10 --> 5x20
import numpy

from functions import *

if __name__ == "__main__":
    D,L=read_traintext("train.txt")
    DTE1,LTE=read_traintext("Test.txt")
    DataPca=compute_PCA(D,4)
    Data_merged=merge_dataset(DataPca)
    DatatestPca = compute_PCA(DTE1, 4)
    DTE = merge_dataset(DatatestPca)
    labels = [ i for i in range(0, numpy.amax(L)+1)]

    class_prior_probability = numpy.array([1/ 10,9/10], dtype=float).reshape(2, 1)

    log_MVG_class_conditional_probabilities = compute_MVG_log_likelihood_as_score_matrix(Data_merged, L, DTE, labels)
    MVG_class_conditional_probabilities = numpy.exp(log_MVG_class_conditional_probabilities)
    MVG_posterior_probability = compute_posterior_probability(MVG_class_conditional_probabilities,
                                                              class_prior_probability)
    log_MVG_posterior_probability = compute_log_posterior_probability(log_MVG_class_conditional_probabilities,
                                                                      class_prior_probability)
    MVG_predictions = numpy.argmax(numpy.exp(log_MVG_posterior_probability), axis=0)
    MVG_prediction_accuracy = compute_prediction_accuracy(MVG_predictions, LTE)
    MVG_error_rate = compute_error_rate(MVG_predictions, LTE)
    print("MultiVariateGaussian prediction accuracy : ", MVG_prediction_accuracy)
    print("MultiVariateGaussian error rate : %.2f" % MVG_error_rate)

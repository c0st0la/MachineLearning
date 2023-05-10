from functions import *


def compute_MVG_accuracy(D, L, DTE, LTE, labels, class_prior_probability):
    log_MVG_class_conditional_probabilities = compute_MVG_log_likelihood_as_score_matrix(D, L, DTE,
                                                                                         labels)
    log_MVG_posterior_probability = compute_log_posterior_probability(log_MVG_class_conditional_probabilities,
                                                                      class_prior_probability)
    MVG_predictions = numpy.argmax(numpy.exp(log_MVG_posterior_probability), axis=0)
    MVG_prediction_accuracy = compute_prediction_accuracy(MVG_predictions, LTE)
    return MVG_prediction_accuracy


def compute_NB_accuracy(D, L, DTE, LTE, labels, class_prior_probability):
    log_NB_class_conditional_probabilities = compute_NB_log_likelihood_as_score_matrix(D, L, DTE,
                                                                                       labels)

    NB_posterior_probability = compute_log_posterior_probability(log_NB_class_conditional_probabilities,
                                                             class_prior_probability)
    NB_predictions = numpy.argmax(numpy.exp(NB_posterior_probability), axis=0)
    NB_prediction_accuracy = compute_prediction_accuracy(NB_predictions, LTE)
    return NB_prediction_accuracy


def compute_TC_accuracy(D, L, DTE, LTE, labels, class_prior_probability):
    log_TC_class_conditional_probabilities = compute_TC_log_likelihood_as_score_matrix(D, L, DTE,
                                                                                       labels)

    TC_posterior_probability = compute_log_posterior_probability(log_TC_class_conditional_probabilities,
                                                             class_prior_probability)
    TC_predictions = numpy.argmax(numpy.exp(TC_posterior_probability), axis=0)
    TC_prediction_accuracy = compute_prediction_accuracy(TC_predictions, LTE)
    return TC_prediction_accuracy


def compute_TNB_accuracy(D, L, DTE, LTE, labels, class_prior_probability):
    log_TNB_class_conditional_probabilities = compute_TNB_log_likelihood_as_score_matrix(D, L, DTE,
                                                                                         labels)
    TNB_posterior_probability = compute_log_posterior_probability(log_TNB_class_conditional_probabilities,
                                                              class_prior_probability)
    TNB_predictions = numpy.argmax(numpy.exp(TNB_posterior_probability), axis=0)
    TNB_prediction_accuracy = compute_prediction_accuracy(TNB_predictions, LTE)
    return TNB_prediction_accuracy


def compute_PCA_data_and_test_merged(D, DTE, i):
    #train
    DataPca_unmerged = compute_PCA(D, i)
    DataPCA = merge_dataset(DataPca_unmerged)
    #test
    DatatestPca = compute_PCA(DTE, i)
    DTE = merge_dataset(DatatestPca)
    return DataPCA, DTE


def compute_PCA_data_and_test_unmerged(D, DTE, i):
    #train
    DPCA = compute_PCA(D, i)
    #test
    DTEPCA = compute_PCA(DTE, i)
    return DPCA, DTEPCA
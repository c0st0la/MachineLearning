import numpy

from functions import *

if __name__ == "__main__":

    # I am loading into D and L respectively: the dataset and the label to which each samples belong to
    D, L = load_iris()
    labels = [ i for i in range(0, numpy.amax(L)+1)]

    (DTR, LTR), (DTE, LTE) = split_db_to_train_test(D, L)

    class_prior_probability = numpy.array([1 / 3, 1 / 3, 1 / 3], dtype=float).reshape(3, 1)

    log_MVG_class_conditional_probabilities = compute_MVG_log_likelihood_as_score_matrix(DTR, LTR, DTE, labels)
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

    # ----------------------------------------------------- #


    # NOW I WILL USE NAIVE BAYES CLASSIFIER

    log_NB_class_conditional_probabilities = compute_NB_log_likelihood_as_score_matrix(DTR, LTR, DTE, labels)
    NB_class_conditional_probabilities = numpy.exp(log_NB_class_conditional_probabilities)
    NB_posterior_probability = compute_posterior_probability(NB_class_conditional_probabilities, class_prior_probability)
    NB_predictions = numpy.argmax(NB_posterior_probability, axis=0)
    NB_prediction_accuracy = compute_prediction_accuracy(NB_predictions, LTE)
    NB_error_rate = compute_error_rate(NB_predictions, LTE)
    print("NaiveBayes prediction accuracy: ", NB_prediction_accuracy)
    print("NaiveBayes error rate: %.2f" % NB_error_rate)

    # ----------------------------------------------------- #

    # NOW I WILL USE TIED COVARIANCE CLASSIFIER

    log_TC_class_conditional_probabilities = compute_TC_log_likelihood_as_score_matrix(DTR, LTR, DTE, labels)
    TC_class_conditional_probabilities = numpy.exp(log_TC_class_conditional_probabilities)
    TC_posterior_probability = compute_posterior_probability(TC_class_conditional_probabilities, class_prior_probability)
    TC_predictions = numpy.argmax(TC_posterior_probability, axis=0)
    TC_prediction_accuracy = compute_prediction_accuracy(TC_predictions, LTE)
    TC_error_rate = compute_error_rate(TC_predictions, LTE)
    print("TiedCovariance prediction accuracy: ", TC_prediction_accuracy)
    print("TiedCovariance error rate: %.2f" % TC_error_rate)

    # ----------------------------------------------------- #

    # NOW I WILL USE TIED NAIVE BAYES COVARIANCE CLASSIFIER

    log_TNB_class_conditional_probabilities = compute_TNB_log_likelihood_as_score_matrix(DTR, LTR, DTE, labels)
    TNB_class_conditional_probabilities = numpy.exp(log_TNB_class_conditional_probabilities)
    TNB_posterior_probability = compute_posterior_probability(TNB_class_conditional_probabilities,
                                                              class_prior_probability)
    TNB_predictions = numpy.argmax(TNB_posterior_probability, axis=0)
    TNB_prediction_accuracy = compute_prediction_accuracy(TNB_predictions, LTE)
    TNB_error_rate = compute_error_rate(TNB_predictions, LTE)
    print("TiedNaiveBayes prediction accuracy: ", TNB_prediction_accuracy)
    print("TiedNaiveBayes error rate: %.2f" % TNB_error_rate)

    # ----------------------------------------------------- #


    # NOW I WILL USE K-FOLD LOO (Leave One Out)

    MVG_accuracy, MVG_error_rate = K_fold_cross_validation(D, L, "MVG", D.shape[1], class_prior_probability, labels)
    NB_accuracy, NB_error_rate = K_fold_cross_validation(D, L, "NB", D.shape[1], class_prior_probability, labels)
    TC_accuracy, TC_error_rate = K_fold_cross_validation(D, L, "TC", D.shape[1], class_prior_probability, labels)
    TNB_accuracy, TNB_error_rate = K_fold_cross_validation(D, L, "TNB", D.shape[1], class_prior_probability, labels)
    print("These are the accuracy computed with K-fold cross validation with K = %d" % D.shape[1])
    print("K-fold MVG_error_rate : %.2f" % MVG_error_rate)
    print("K-fold NB_error_rate : %.2f" % NB_error_rate)
    print("K-fold TC_error_rate : %.2f" % TC_error_rate)
    print("K-fold TNB_error_rate : %.2f" % TNB_error_rate)
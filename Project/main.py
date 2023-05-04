  #10x10 --> 5x20
import numpy

from functions import *

if __name__ == "__main__":

    ##PCA+ MultivariateGaussian
    D,L,D_original=read_traintext("train.txt")
    DTE1,LTE,DTE_original=read_traintext("Test.txt")
    labels = [i for i in range(0, numpy.amax(L) + 1)]
    class_prior_probability = numpy.array([9/10, 1 / 10], dtype=float).reshape(2, 1)

    #no preprocessing
    print("----------------no processing-------------", )


    log_MVG_class_conditional_probabilities = compute_MVG_log_likelihood_as_score_matrix(D_original, L, DTE_original, labels)
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

   #NAIVE BAYES CLASSIFIER

    log_NB_class_conditional_probabilities = compute_NB_log_likelihood_as_score_matrix(D_original, L, DTE_original, labels)
    NB_class_conditional_probabilities = numpy.exp(log_NB_class_conditional_probabilities)
    NB_posterior_probability = compute_posterior_probability(NB_class_conditional_probabilities,
                                                             class_prior_probability)
    NB_predictions = numpy.argmax(NB_posterior_probability, axis=0)
    NB_prediction_accuracy = compute_prediction_accuracy(NB_predictions, LTE)
    NB_error_rate = compute_error_rate(NB_predictions, LTE)
    print("NaiveBayes prediction accuracy: ", NB_prediction_accuracy)
    print("NaiveBayes error rate: %.2f" % NB_error_rate)

    #   TIED COVARIANCE CLASSIFIER

    log_TC_class_conditional_probabilities = compute_TC_log_likelihood_as_score_matrix(D_original, L, DTE_original, labels)
    TC_class_conditional_probabilities = numpy.exp(log_TC_class_conditional_probabilities)
    TC_posterior_probability = compute_posterior_probability(TC_class_conditional_probabilities,
                                                             class_prior_probability)
    TC_predictions = numpy.argmax(TC_posterior_probability, axis=0)
    TC_prediction_accuracy = compute_prediction_accuracy(TC_predictions, LTE)
    TC_error_rate = compute_error_rate(TC_predictions, LTE)
    print("TiedCovariance prediction accuracy: ", TC_prediction_accuracy)
    print("TiedCovariance error rate: %.2f" % TC_error_rate)

    # TIED NAIVE BAYES COVARIANCE CLASSIFIER

    log_TNB_class_conditional_probabilities = compute_TNB_log_likelihood_as_score_matrix(D_original, L, DTE_original, labels)
    TNB_class_conditional_probabilities = numpy.exp(log_TNB_class_conditional_probabilities)
    TNB_posterior_probability = compute_posterior_probability(TNB_class_conditional_probabilities,
                                                              class_prior_probability)
    TNB_predictions = numpy.argmax(TNB_posterior_probability, axis=0)
    TNB_prediction_accuracy = compute_prediction_accuracy(TNB_predictions, LTE)
    TNB_error_rate = compute_error_rate(TNB_predictions, LTE)
    print("TiedCovariance  Naive Bayes prediction accuracy: ", TNB_prediction_accuracy)
    print("TiedCovariance  Naive Bayes  error rate: %.2f" % TNB_error_rate)


    #no preprocessing
    # LDA

    #DataPCA_false, DataPCA_true = filter_dataset_by_labels(DataPCA, L)

    print("-------------------LDA--------------------", 1)
    DP_LDA = compute_LDA_generalized_eigenvalue(D_original, L, directions=1, labels=labels)
    DTEST_LDA = compute_LDA_generalized_eigenvalue(DTE_original, LTE, directions=1, labels=labels)

    ##LDA+ MultivariateGaussian

    log_MVG_class_conditional_probabilities = compute_MVG_log_likelihood_as_score_matrix(DP_LDA, L, DTEST_LDA, labels)
    MVG_class_conditional_probabilities = numpy.exp(log_MVG_class_conditional_probabilities)
    MVG_posterior_probability = compute_posterior_probability(MVG_class_conditional_probabilities,
                                                              class_prior_probability)
    log_MVG_posterior_probability = compute_log_posterior_probability(log_MVG_class_conditional_probabilities,
                                                                      class_prior_probability)
    MVG_predictions = numpy.argmax(numpy.exp(log_MVG_posterior_probability), axis=0)
    MVG_prediction_accuracy = compute_prediction_accuracy(MVG_predictions, LTE)
    MVG_error_rate = compute_error_rate(MVG_predictions, LTE)
    print("LDA+MultiVariateGaussian prediction accuracy : ", MVG_prediction_accuracy)
    print("LDAMultiVariateGaussian error rate : %.2f" % MVG_error_rate)

    # Lda + NAIVE BAYES CLASSIFIER

    log_NB_class_conditional_probabilities = compute_NB_log_likelihood_as_score_matrix(DP_LDA, L, DTEST_LDA, labels)
    NB_class_conditional_probabilities = numpy.exp(log_NB_class_conditional_probabilities)
    NB_posterior_probability = compute_posterior_probability(NB_class_conditional_probabilities,
                                                             class_prior_probability)
    NB_predictions = numpy.argmax(NB_posterior_probability, axis=0)
    NB_prediction_accuracy = compute_prediction_accuracy(NB_predictions, LTE)
    NB_error_rate = compute_error_rate(NB_predictions, LTE)
    print("LDA+NaiveBayes prediction accuracy: ", NB_prediction_accuracy)
    print("LDA+ NaiveBayes error rate: %.2f" % NB_error_rate)

    # LDA +  TIED COVARIANCE CLASSIFIER

    log_TC_class_conditional_probabilities = compute_TC_log_likelihood_as_score_matrix(DP_LDA, L, DTEST_LDA, labels)
    TC_class_conditional_probabilities = numpy.exp(log_TC_class_conditional_probabilities)
    TC_posterior_probability = compute_posterior_probability(TC_class_conditional_probabilities,
                                                             class_prior_probability)
    TC_predictions = numpy.argmax(TC_posterior_probability, axis=0)
    TC_prediction_accuracy = compute_prediction_accuracy(TC_predictions, LTE)
    TC_error_rate = compute_error_rate(TC_predictions, LTE)
    print("LDA+TiedCovariance prediction accuracy: ", TC_prediction_accuracy)
    print("LDATiedCovariance error rate: %.2f" % TC_error_rate)

    # LDA+TIED NAIVE BAYES COVARIANCE CLASSIFIER

    log_TNB_class_conditional_probabilities = compute_TNB_log_likelihood_as_score_matrix(DP_LDA, L, DTEST_LDA, labels)
    TNB_class_conditional_probabilities = numpy.exp(log_TNB_class_conditional_probabilities)
    TNB_posterior_probability = compute_posterior_probability(TNB_class_conditional_probabilities,
                                                              class_prior_probability)
    TNB_predictions = numpy.argmax(TNB_posterior_probability, axis=0)
    TNB_prediction_accuracy = compute_prediction_accuracy(TNB_predictions, LTE)
    TNB_error_rate = compute_error_rate(TNB_predictions, LTE)
    print("LDA+TiedCovariance  Naive Bayes prediction accuracy: ", TNB_prediction_accuracy)
    print("LDA+TiedCovariance  Naive Bayes  error rate: %.2f" % TNB_error_rate)


for i in range(1,5):
    print("------------------PCA------------------",i)
    DataPca_unmerged=compute_PCA(D,i)
    DataPCA=merge_dataset(DataPca_unmerged)
    DatatestPca = compute_PCA(DTE1, i)
    DTE = merge_dataset(DatatestPca)



    log_MVG_class_conditional_probabilities = compute_MVG_log_likelihood_as_score_matrix(DataPCA, L, DTE, labels)
    MVG_class_conditional_probabilities = numpy.exp(log_MVG_class_conditional_probabilities)
    MVG_posterior_probability = compute_posterior_probability(MVG_class_conditional_probabilities,
                                                              class_prior_probability)
    log_MVG_posterior_probability = compute_log_posterior_probability(log_MVG_class_conditional_probabilities,
                                                                      class_prior_probability)
    MVG_predictions = numpy.argmax(numpy.exp(log_MVG_posterior_probability), axis=0)
    MVG_prediction_accuracy = compute_prediction_accuracy(MVG_predictions, LTE)
    MVG_error_rate = compute_error_rate(MVG_predictions, LTE)
    print("-------------------PCA--------------------")
    print("MultiVariateGaussian prediction accuracy : ", MVG_prediction_accuracy)
    print("MultiVariateGaussian error rate : %.2f" % MVG_error_rate)



# Pca + NAIVE BAYES CLASSIFIER

    log_NB_class_conditional_probabilities = compute_NB_log_likelihood_as_score_matrix(DataPCA, L, DTE, labels)
    NB_class_conditional_probabilities = numpy.exp(log_NB_class_conditional_probabilities)
    NB_posterior_probability = compute_posterior_probability(NB_class_conditional_probabilities, class_prior_probability)
    NB_predictions = numpy.argmax(NB_posterior_probability, axis=0)
    NB_prediction_accuracy = compute_prediction_accuracy(NB_predictions, LTE)
    NB_error_rate = compute_error_rate(NB_predictions, LTE)
    print("NaiveBayes prediction accuracy: ", NB_prediction_accuracy)
    print("NaiveBayes error rate: %.2f" % NB_error_rate)


# PCA+  TIED COVARIANCE CLASSIFIER

    log_TC_class_conditional_probabilities = compute_TC_log_likelihood_as_score_matrix(DataPCA, L, DTE, labels)
    TC_class_conditional_probabilities = numpy.exp(log_TC_class_conditional_probabilities)
    TC_posterior_probability = compute_posterior_probability(TC_class_conditional_probabilities, class_prior_probability)
    TC_predictions = numpy.argmax(TC_posterior_probability, axis=0)
    TC_prediction_accuracy = compute_prediction_accuracy(TC_predictions, LTE)
    TC_error_rate = compute_error_rate(TC_predictions, LTE)
    print("TiedCovariance prediction accuracy: ", TC_prediction_accuracy)
    print("TiedCovariance error rate: %.2f" % TC_error_rate)

# PCA+TIED NAIVE BAYES COVARIANCE CLASSIFIER

    log_TNB_class_conditional_probabilities = compute_TNB_log_likelihood_as_score_matrix(DataPCA, L, DTE, labels)
    TNB_class_conditional_probabilities = numpy.exp(log_TNB_class_conditional_probabilities)
    TNB_posterior_probability = compute_posterior_probability(TNB_class_conditional_probabilities, class_prior_probability)
    TNB_predictions = numpy.argmax(TNB_posterior_probability, axis=0)
    TNB_prediction_accuracy = compute_prediction_accuracy(TNB_predictions, LTE)
    TNB_error_rate = compute_error_rate(TNB_predictions, LTE)
    print("TiedCovariance  Naive Bayes prediction accuracy: ", TNB_prediction_accuracy)
    print("TiedCovariance  Naive Bayes  error rate: %.2f" % TNB_error_rate)



    # -------------------LDA+PCA----------------------------
    print("----------LDA+PCA")
    DP_LDA = compute_LDA_generalized_eigenvalue(DataPCA, L, directions=1, labels=labels)
    DTEST_LDA = compute_LDA_generalized_eigenvalue(DTE, LTE, directions=1, labels=labels)

    ##LDA+ MultivariateGaussian

    log_MVG_class_conditional_probabilities = compute_MVG_log_likelihood_as_score_matrix(DP_LDA, L, DTEST_LDA, labels)
    MVG_class_conditional_probabilities = numpy.exp(log_MVG_class_conditional_probabilities)
    MVG_posterior_probability = compute_posterior_probability(MVG_class_conditional_probabilities,
                                                              class_prior_probability)
    log_MVG_posterior_probability = compute_log_posterior_probability(log_MVG_class_conditional_probabilities,
                                                                      class_prior_probability)
    MVG_predictions = numpy.argmax(numpy.exp(log_MVG_posterior_probability), axis=0)
    MVG_prediction_accuracy = compute_prediction_accuracy(MVG_predictions, LTE)
    MVG_error_rate = compute_error_rate(MVG_predictions, LTE)
    print("LDA+MultiVariateGaussian prediction accuracy : ", MVG_prediction_accuracy)
    print("LDAMultiVariateGaussian error rate : %.2f" % MVG_error_rate)

    # Lda + NAIVE BAYES CLASSIFIER

    log_NB_class_conditional_probabilities = compute_NB_log_likelihood_as_score_matrix(DP_LDA, L, DTEST_LDA, labels)
    NB_class_conditional_probabilities = numpy.exp(log_NB_class_conditional_probabilities)
    NB_posterior_probability = compute_posterior_probability(NB_class_conditional_probabilities,
                                                             class_prior_probability)
    NB_predictions = numpy.argmax(NB_posterior_probability, axis=0)
    NB_prediction_accuracy = compute_prediction_accuracy(NB_predictions, LTE)
    NB_error_rate = compute_error_rate(NB_predictions, LTE)
    print("LDA+NaiveBayes prediction accuracy: ", NB_prediction_accuracy)
    print("LDA+ NaiveBayes error rate: %.2f" % NB_error_rate)

    # LDA +  TIED COVARIANCE CLASSIFIER

    log_TC_class_conditional_probabilities = compute_TC_log_likelihood_as_score_matrix(DP_LDA, L, DTEST_LDA, labels)
    TC_class_conditional_probabilities = numpy.exp(log_TC_class_conditional_probabilities)
    TC_posterior_probability = compute_posterior_probability(TC_class_conditional_probabilities,
                                                             class_prior_probability)
    TC_predictions = numpy.argmax(TC_posterior_probability, axis=0)
    TC_prediction_accuracy = compute_prediction_accuracy(TC_predictions, LTE)
    TC_error_rate = compute_error_rate(TC_predictions, LTE)
    print("LDA+TiedCovariance prediction accuracy: ", TC_prediction_accuracy)
    print("LDATiedCovariance error rate: %.2f" % TC_error_rate)

    # LDA+TIED NAIVE BAYES COVARIANCE CLASSIFIER

    log_TNB_class_conditional_probabilities = compute_TNB_log_likelihood_as_score_matrix(DP_LDA, L, DTEST_LDA, labels)
    TNB_class_conditional_probabilities = numpy.exp(log_TNB_class_conditional_probabilities)
    TNB_posterior_probability = compute_posterior_probability(TNB_class_conditional_probabilities,
                                                              class_prior_probability)
    TNB_predictions = numpy.argmax(TNB_posterior_probability, axis=0)
    TNB_prediction_accuracy = compute_prediction_accuracy(TNB_predictions, LTE)
    TNB_error_rate = compute_error_rate(TNB_predictions, LTE)
    print("LDA+TiedCovariance  Naive Bayes prediction accuracy: ", TNB_prediction_accuracy)
    print("LDA+TiedCovariance  Naive Bayes  error rate: %.2f" % TNB_error_rate)


#--------------------------PCA No Merge------------------------#
print("--------------------------PCA No Merge------------------------")

for i in range(1,5):
    print("------------------PCA------------------",i)
    DataPca_unmerged=compute_PCA(D_original,i)

    DatatestPca = compute_PCA(DTE_original, i)




    log_MVG_class_conditional_probabilities = compute_MVG_log_likelihood_as_score_matrix(DataPca_unmerged, L, DatatestPca, labels)
    MVG_class_conditional_probabilities = numpy.exp(log_MVG_class_conditional_probabilities)
    MVG_posterior_probability = compute_posterior_probability(MVG_class_conditional_probabilities,
                                                              class_prior_probability)
    log_MVG_posterior_probability = compute_log_posterior_probability(log_MVG_class_conditional_probabilities,
                                                                      class_prior_probability)
    MVG_predictions = numpy.argmax(numpy.exp(log_MVG_posterior_probability), axis=0)
    MVG_prediction_accuracy = compute_prediction_accuracy(MVG_predictions, LTE)
    MVG_error_rate = compute_error_rate(MVG_predictions, LTE)
    print("-------------------PCA--------------------")
    print("MultiVariateGaussian prediction accuracy : ", MVG_prediction_accuracy)
    print("MultiVariateGaussian error rate : %.2f" % MVG_error_rate)



# Pca + NAIVE BAYES CLASSIFIER

    log_NB_class_conditional_probabilities = compute_NB_log_likelihood_as_score_matrix(DataPca_unmerged, L, DatatestPca, labels)
    NB_class_conditional_probabilities = numpy.exp(log_NB_class_conditional_probabilities)
    NB_posterior_probability = compute_posterior_probability(NB_class_conditional_probabilities, class_prior_probability)
    NB_predictions = numpy.argmax(NB_posterior_probability, axis=0)
    NB_prediction_accuracy = compute_prediction_accuracy(NB_predictions, LTE)
    NB_error_rate = compute_error_rate(NB_predictions, LTE)
    print("NaiveBayes prediction accuracy: ", NB_prediction_accuracy)
    print("NaiveBayes error rate: %.2f" % NB_error_rate)


# PCA+  TIED COVARIANCE CLASSIFIER

    log_TC_class_conditional_probabilities = compute_TC_log_likelihood_as_score_matrix(DataPca_unmerged, L, DatatestPca, labels)
    TC_class_conditional_probabilities = numpy.exp(log_TC_class_conditional_probabilities)
    TC_posterior_probability = compute_posterior_probability(TC_class_conditional_probabilities, class_prior_probability)
    TC_predictions = numpy.argmax(TC_posterior_probability, axis=0)
    TC_prediction_accuracy = compute_prediction_accuracy(TC_predictions, LTE)
    TC_error_rate = compute_error_rate(TC_predictions, LTE)
    print("TiedCovariance prediction accuracy: ", TC_prediction_accuracy)
    print("TiedCovariance error rate: %.2f" % TC_error_rate)

# PCA+TIED NAIVE BAYES COVARIANCE CLASSIFIER

    log_TNB_class_conditional_probabilities = compute_TNB_log_likelihood_as_score_matrix(DataPca_unmerged, L, DatatestPca, labels)
    TNB_class_conditional_probabilities = numpy.exp(log_TNB_class_conditional_probabilities)
    TNB_posterior_probability = compute_posterior_probability(TNB_class_conditional_probabilities, class_prior_probability)
    TNB_predictions = numpy.argmax(TNB_posterior_probability, axis=0)
    TNB_prediction_accuracy = compute_prediction_accuracy(TNB_predictions, LTE)
    TNB_error_rate = compute_error_rate(TNB_predictions, LTE)
    print("TiedCovariance  Naive Bayes prediction accuracy: ", TNB_prediction_accuracy)
    print("TiedCovariance  Naive Bayes  error rate: %.2f" % TNB_error_rate)



    # -------------------LDA+PCA----------------------------
    print("----------LDA+PCA")
    DP_LDA = compute_LDA_generalized_eigenvalue(DataPca_unmerged, L, directions=1, labels=labels)
    DTEST_LDA = compute_LDA_generalized_eigenvalue(DatatestPca, LTE, directions=1, labels=labels)

    ##LDA+ MultivariateGaussian

    log_MVG_class_conditional_probabilities = compute_MVG_log_likelihood_as_score_matrix(DP_LDA, L, DTEST_LDA, labels)
    MVG_class_conditional_probabilities = numpy.exp(log_MVG_class_conditional_probabilities)
    MVG_posterior_probability = compute_posterior_probability(MVG_class_conditional_probabilities,
                                                              class_prior_probability)
    log_MVG_posterior_probability = compute_log_posterior_probability(log_MVG_class_conditional_probabilities,
                                                                      class_prior_probability)
    MVG_predictions = numpy.argmax(numpy.exp(log_MVG_posterior_probability), axis=0)
    MVG_prediction_accuracy = compute_prediction_accuracy(MVG_predictions, LTE)
    MVG_error_rate = compute_error_rate(MVG_predictions, LTE)
    print("LDA+MultiVariateGaussian prediction accuracy : ", MVG_prediction_accuracy)
    print("LDAMultiVariateGaussian error rate : %.2f" % MVG_error_rate)

    # Lda + NAIVE BAYES CLASSIFIER

    log_NB_class_conditional_probabilities = compute_NB_log_likelihood_as_score_matrix(DP_LDA, L, DTEST_LDA, labels)
    NB_class_conditional_probabilities = numpy.exp(log_NB_class_conditional_probabilities)
    NB_posterior_probability = compute_posterior_probability(NB_class_conditional_probabilities,
                                                             class_prior_probability)
    NB_predictions = numpy.argmax(NB_posterior_probability, axis=0)
    NB_prediction_accuracy = compute_prediction_accuracy(NB_predictions, LTE)
    NB_error_rate = compute_error_rate(NB_predictions, LTE)
    print("LDA+NaiveBayes prediction accuracy: ", NB_prediction_accuracy)
    print("LDA+ NaiveBayes error rate: %.2f" % NB_error_rate)

    # LDA +  TIED COVARIANCE CLASSIFIER

    log_TC_class_conditional_probabilities = compute_TC_log_likelihood_as_score_matrix(DP_LDA, L, DTEST_LDA, labels)
    TC_class_conditional_probabilities = numpy.exp(log_TC_class_conditional_probabilities)
    TC_posterior_probability = compute_posterior_probability(TC_class_conditional_probabilities,
                                                             class_prior_probability)
    TC_predictions = numpy.argmax(TC_posterior_probability, axis=0)
    TC_prediction_accuracy = compute_prediction_accuracy(TC_predictions, LTE)
    TC_error_rate = compute_error_rate(TC_predictions, LTE)
    print("LDA+TiedCovariance prediction accuracy: ", TC_prediction_accuracy)
    print("LDATiedCovariance error rate: %.2f" % TC_error_rate)

    # LDA+TIED NAIVE BAYES COVARIANCE CLASSIFIER

    log_TNB_class_conditional_probabilities = compute_TNB_log_likelihood_as_score_matrix(DP_LDA, L, DTEST_LDA, labels)
    TNB_class_conditional_probabilities = numpy.exp(log_TNB_class_conditional_probabilities)
    TNB_posterior_probability = compute_posterior_probability(TNB_class_conditional_probabilities,
                                                              class_prior_probability)
    TNB_predictions = numpy.argmax(TNB_posterior_probability, axis=0)
    TNB_prediction_accuracy = compute_prediction_accuracy(TNB_predictions, LTE)
    TNB_error_rate = compute_error_rate(TNB_predictions, LTE)
    print("LDA+TiedCovariance  Naive Bayes prediction accuracy: ", TNB_prediction_accuracy)
    print("LDA+TiedCovariance  Naive Bayes  error rate: %.2f" % TNB_error_rate)
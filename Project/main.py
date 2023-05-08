  #10x10 --> 5x20
import numpy

from functions import *
from functions2 import *
if __name__ == "__main__":

    ##PCA+ MultivariateGaussian
    D,L,D_original=read_file("train.txt")
    DTE1,LTE,DTE_original=read_file("Test.txt")

    #provo a normalizzare

    Dmean=numpy.mean(D,axis=1)
    Dvar=numpy.std(D,axis=1)
    D1 = (D- numpy.mean(D, axis=1).reshape(5,1)) / numpy.std(D, axis=1).reshape(5,1) +0.0000001
    D_original1= (D_original-numpy.mean(D_original,axis=1).reshape(10,1))/numpy.std(D_original,axis=1).reshape(10,1) + 0.0000001
    labels = [i for i in range(0, numpy.amax(L) + 1)]



    class_prior_probability = numpy.array([9/10, 1/10], dtype=float).reshape(2, 1)

    #no preprocessing
    print("----------------no processing-------------", )

    print("MultiVariateGaussian prediction accuracy : ",
          compute_MVG_accuracy(D_original, L, DTE_original, LTE, labels, class_prior_probability))


   #NAIVE BAYES CLASSIFIER


    print("NaiveBayes prediction accuracy: ",
          compute_NB_accuracy(D_original, L, DTE_original, LTE, labels, class_prior_probability))

    #   TIED COVARIANCE CLASSIFIER


    print("TiedCovariance prediction accuracy: ",
          compute_TC_accuracy(D_original, L, DTE_original, LTE, labels, class_prior_probability))

    # TIED NAIVE BAYES COVARIANCE CLASSIFIER


    print("TiedCovariance  Naive Bayes prediction accuracy: ",
          compute_TNB_accuracy(D_original, L, DTE_original, LTE, labels, class_prior_probability))


    #no preprocessing
    # LDA

    #DataPCA_false, DataPCA_true = filter_dataset_by_labels(DataPCA, L)

    print("-------------------LDA--------------------", 1)
    D_LDA = compute_LDA_generalized_eigenvalue(D_original, L, directions=1, labels=labels)
    DTEST_LDA = compute_LDA_generalized_eigenvalue(DTE_original, LTE, directions=1, labels=labels)

    ##LDA+ MultivariateGaussian


    print("LDA+MultiVariateGaussian prediction accuracy : ",
          compute_MVG_accuracy(D_LDA, L, DTEST_LDA, LTE, labels, class_prior_probability))

    # Lda + NAIVE BAYES CLASSIFIER

    print("LDA+NaiveBayes prediction accuracy: ",
          compute_NB_accuracy(D_LDA, L, DTEST_LDA, LTE, labels, class_prior_probability))

    # LDA +  TIED COVARIANCE CLASSIFIER

    print("LDA+TiedCovariance prediction accuracy: ",
          compute_TC_accuracy(D_LDA, L, DTEST_LDA, LTE, labels, class_prior_probability))


    # LDA+TIED NAIVE BAYES COVARIANCE CLASSIFIER

    print("LDA+TiedCovariance  Naive Bayes prediction accuracy: ",
          compute_TNB_accuracy(D_LDA, L, DTEST_LDA, LTE, labels, class_prior_probability))


    for i in range(1,5):
        print("------------------PCA------------------",i)

        DPCA, DTEPCA = compute_PCA_data_and_test_merged(D, DTE1, i)

        print("MultiVariateGaussian prediction accuracy : ",
              compute_MVG_accuracy(DPCA, L, DTEPCA, LTE, labels, class_prior_probability))

        # NAIVE BAYES CLASSIFIER

        print("NaiveBayes prediction accuracy: ",
              compute_NB_accuracy(DPCA, L, DTEPCA, LTE, labels, class_prior_probability))

        #   TIED COVARIANCE CLASSIFIER

        print("TiedCovariance prediction accuracy: ",
              compute_TC_accuracy(DPCA, L, DTEPCA, LTE, labels, class_prior_probability))

        # TIED NAIVE BAYES COVARIANCE CLASSIFIER

        print("TiedCovariance  Naive Bayes prediction accuracy: ",
              compute_TNB_accuracy(DPCA, L, DTEPCA, LTE, labels, class_prior_probability))

        # -------------------LDA+PCA----------------------------

        print("----------LDA+PCA-------------")
        DP_LDA = compute_LDA_generalized_eigenvalue(DPCA, L, directions=1, labels=labels)
        DTEST_LDA = compute_LDA_generalized_eigenvalue(DTEPCA, LTE, directions=1, labels=labels)

        print("MultiVariateGaussian prediction accuracy : ",
              compute_MVG_accuracy(DP_LDA, L, DTEST_LDA, LTE, labels, class_prior_probability))

        # NAIVE BAYES CLASSIFIER

        print("NaiveBayes prediction accuracy: ",
              compute_NB_accuracy(DP_LDA, L, DTEST_LDA, LTE, labels, class_prior_probability))

        #   TIED COVARIANCE CLASSIFIER

        print("TiedCovariance prediction accuracy: ",
              compute_TC_accuracy(DP_LDA, L, DTEST_LDA, LTE, labels, class_prior_probability))

        # TIED NAIVE BAYES COVARIANCE CLASSIFIER

        print("TiedCovariance  Naive Bayes prediction accuracy: ",
              compute_TNB_accuracy(DP_LDA, L, DTEST_LDA, LTE, labels, class_prior_probability))


    #--------------------------PCA No Merge--------------------------------------------------#
    print("--------------------------PCA No Merge------------------------")

    for i in range(1,10):
        print("------------------PCA------------------",i)
        DPCA, DTEPCA = compute_PCA_data_and_test_unmerged(D_original, DTE_original, i)

        print("MultiVariateGaussian prediction accuracy : ",
              compute_MVG_accuracy(DPCA, L, DTEPCA, LTE, labels, class_prior_probability))

        # NAIVE BAYES CLASSIFIER

        print("NaiveBayes prediction accuracy: ",
              compute_NB_accuracy(DPCA, L, DTEPCA, LTE, labels, class_prior_probability))

        #   TIED COVARIANCE CLASSIFIER

        print("TiedCovariance prediction accuracy: ",
              compute_TC_accuracy(DPCA, L, DTEPCA, LTE, labels, class_prior_probability))

        # TIED NAIVE BAYES COVARIANCE CLASSIFIER

        print("TiedCovariance  Naive Bayes prediction accuracy: ",
              compute_TNB_accuracy(DPCA, L, DTEPCA, LTE, labels, class_prior_probability))






        # -------------------LDA+PCA----------------------------
        print("----------LDA+PCA")
        DP_LDA = compute_LDA_generalized_eigenvalue(DPCA, L, directions=1, labels=labels)
        DTEST_LDA = compute_LDA_generalized_eigenvalue(DTEPCA, LTE, directions=1, labels=labels)

        ##LDA+ MultivariateGaussian

        print("MultiVariateGaussian prediction accuracy : ",
              compute_MVG_accuracy(DP_LDA, L, DTEST_LDA, LTE, labels, class_prior_probability))

        # NAIVE BAYES CLASSIFIER

        print("NaiveBayes prediction accuracy: ",
              compute_NB_accuracy(DP_LDA, L, DTEST_LDA, LTE, labels, class_prior_probability))

        #   TIED COVARIANCE CLASSIFIER

        print("TiedCovariance prediction accuracy: ",
              compute_TC_accuracy(DP_LDA, L, DTEST_LDA, LTE, labels, class_prior_probability))

        # TIED NAIVE BAYES COVARIANCE CLASSIFIER

        print("TiedCovariance  Naive Bayes prediction accuracy: ",
              compute_TNB_accuracy(DP_LDA, L, DTEST_LDA, LTE, labels, class_prior_probability))
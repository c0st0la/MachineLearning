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


def compute_SVM_vector_accuracy(DTR, LTR, DTE, LTE, K, C, threshold=0):
        svm = SVMClass(DTR, LTR, DTE, K, C)
        bounds = []
        for i in range(svm.DTR.shape[1]):
            bounds.append((0, svm.C))
        x, f, d = scipy.optimize.fmin_l_bfgs_b(svm.svm_dual_obj, x0=numpy.zeros(svm.DTR.shape[1]), fprime=None,
                                               bounds=bounds, factr=1.0)

        # print("The objective value at the minimum is %f" % f)

        alpha = x
        z = []
        for index in range(svm.LTR.size):
            if LTR[index] == 1:
                z.append(1)
            else:
                z.append(-1)
        w_optimal = 0
        for i in range(svm.DTR.shape[1]):
            w_optimal += alpha[i] * z[i] * svm.DTR[:, i]
        scores = []
        for index in range(svm.DTE.shape[1]):
            scores.append(numpy.dot(w_optimal.T, svm.DTE[:, index]))
        predictions = []
        for score in scores:
            if score > threshold:
                predictions.append(1)
            else:
                predictions.append(0)
        primalObjFunc = svm.compute_primal_obj(w_optimal)
        dualObjFunc = f
       # print("Primal Obj Func %.6f" % primalObjFunc)
        #print("Dual Obj Func %.6f" % dualObjFunc)
        #print("Duality gap is %.10f" % ((primalObjFunc + dualObjFunc) * 10 ** 5))
        print("The SVM accuracy is %.3f" % compute_prediction_accuracy(predictions, LTE))

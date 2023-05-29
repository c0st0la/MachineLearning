from functions import *
from discreteFunctions import *
from Data import load
import numpy as np

if __name__ == "__main__":
    D, L = load_iris()
    labels = [i for i in range(0, numpy.amax(L) + 1)]

    (DTR, LTR), (DTE, LTE) = split_db_to_train_test(D, L)

    class_prior_probability = numpy.array([1 / 3, 1 / 3, 1 / 3], dtype=float).reshape(3, 1)
    missClassificationCosts = numpy.array([1, 1, 1], dtype=float).reshape(3, 1)
    log_MVG_class_conditional_probabilities = compute_MVG_log_likelihood_as_score_matrix(DTR, LTR, DTE, labels)
    MVG_class_conditional_probabilities = numpy.exp(log_MVG_class_conditional_probabilities)
    MVG_posterior_probability = compute_posterior_probability(MVG_class_conditional_probabilities,
                                                              class_prior_probability)
    log_MVG_posterior_probability = compute_log_posterior_probability(log_MVG_class_conditional_probabilities,
                                                                      class_prior_probability)
    MVG_predictions = numpy.argmax(numpy.exp(log_MVG_posterior_probability), axis=0)
    confusionMatrix = compute_confusion_matrix(MVG_predictions, LTE)


    # I REPEAT THE SAME FOR THE TIED CLASSIFIER #

    log_TC_class_conditional_probabilities = compute_TC_log_likelihood_as_score_matrix(DTR, LTR, DTE, labels)
    TC_class_conditional_probabilities = numpy.exp(log_TC_class_conditional_probabilities)
    TC_posterior_probability = compute_posterior_probability(TC_class_conditional_probabilities,
                                                             class_prior_probability)
    TC_predictions = numpy.argmax(TC_posterior_probability, axis=0)

    confusionMatrix = compute_confusion_matrix(TC_predictions, LTE)
    1



    # I COMPUTE THE CONFUSION MATRIX WITH THE DIVINA COMMEDIA #

    inf_text, pur_text, par_text = load.load_data()

    inf_training, inf_evaluation = load.split_data(inf_text, 4)
    pur_training, pur_evaluation= load.split_data(pur_text, 4)
    par_training, par_evaluation = load.split_data(par_text, 4)

    ### Solution 1 ###
    ### Multiclass ###

    class2Index = {'inferno': 0, 'purgatorio': 1, 'paradiso': 2}

    dictTextTrain = {
        'inferno': inf_training,
        'purgatorio': pur_training,
        'paradiso': par_training
    }

    tercets_list_evaluation = inf_evaluation + pur_evaluation + par_evaluation
    class_prior_probability = numpy.array([1. / 3., 1. / 3., 1. / 3.]).reshape((3, 1))
    log_class_conditional_probabilities = compute_class_conditional_log_likelihoods_as_score_matrix(
        dictTextTrain,
        tercets_list_evaluation,
        class2Index
    )
    multiclass_posterior_probability = numpy.exp(compute_log_posterior_probability(log_class_conditional_probabilities,
                                                                            numpy.array([1. / 3., 1. / 3., 1. / 3.])))
    LTE_inf = numpy.zeros(len(inf_evaluation), dtype=int)
    LTE_inf[:] = class2Index['inferno']

    LTE_pur = numpy.zeros(len(pur_evaluation), dtype=int)
    LTE_pur[:] = class2Index['purgatorio']

    LTE_par = numpy.zeros(len(par_evaluation), dtype=int)
    LTE_par[:] = class2Index['paradiso']

    LTE = numpy.hstack([LTE_inf, LTE_pur, LTE_par])

    multiclass_predictions = numpy.argmax(multiclass_posterior_probability, axis=0)

    confusionMatrix = compute_confusion_matrix(multiclass_predictions, LTE)
    classPriorProbability = numpy.array([0.5, 0.5], dtype=float)
    costs = numpy.array([1, 1], dtype=float)
    logLikelihoodRatioInfPar = np.load('Data/commedia_llr_infpar.npy')
    LInfPar = np.load('Data/commedia_labels_infpar.npy')
    optimalBayesDecisionPredictions = compute_optimal_bayes_decision(logLikelihoodRatioInfPar, classPriorProbability, costs, LInfPar)

    confusionMatrix = compute_confusion_matrix(optimalBayesDecisionPredictions, LInfPar)
    1

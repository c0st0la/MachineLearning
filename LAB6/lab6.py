from data import load
from utilities import *
import numpy


def load_data():
    lInf = []

    f = open('data/inferno.txt', encoding="ISO-8859-1")

    for line in f:
        lInf.append(line.strip())
    f.close()

    lPur = []

    f = open('data/purgatorio.txt', encoding="ISO-8859-1")

    for line in f:
        lPur.append(line.strip())
    f.close()

    lPar = []

    f = open('data/paradiso.txt', encoding="ISO-8859-1")

    for line in f:
        lPar.append(line.strip())
    f.close()

    return lInf, lPur, lPar


def split_data(l, n):
    lTrain, lTest = [], []
    for i in range(len(l)):
        if i % n == 0:
            lTest.append(l[i])
        else:
            lTrain.append(l[i])

    return lTrain, lTest


if __name__ == "__main__":
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
    multiclass_prediction_accuracy = compute_prediction_accuracy(multiclass_predictions, LTE)
    multiclass_error_rate = compute_error_rate(multiclass_predictions, LTE)
    print("Multiclass prediction accuracy : %.2f" % multiclass_prediction_accuracy)

    # Per-class accuracy
    inf_prediction = numpy.argmax(multiclass_posterior_probability[:, LTE == class2Index['inferno']], axis = 0)
    pur_prediction = numpy.argmax(multiclass_posterior_probability[:, LTE == class2Index['purgatorio']], axis = 0)
    par_prediction = numpy.argmax(multiclass_posterior_probability[:, LTE == class2Index['paradiso']], axis = 0)
    inf_prediction_accuracy = compute_prediction_accuracy(inf_prediction, LTE[LTE == class2Index['inferno']])
    pur_prediction_accuracy = compute_prediction_accuracy(pur_prediction, LTE[LTE == class2Index['purgatorio']])
    par_prediction_accuracy = compute_prediction_accuracy(par_prediction, LTE[LTE == class2Index['paradiso']])
    print("Inferno prediction accuracy %.2f" % inf_prediction_accuracy)
    print("Purgatorio prediction accuracy  %.2f" % pur_prediction_accuracy)
    print("Paradiso prediction accuracy  %.2f" % par_prediction_accuracy)

    # Per-2-classes accuracy
    # Inferno-Paradiso

    inf_par_evaluation = inf_evaluation + par_evaluation

    inf_par_log_class_conditional_probabilities = compute_class_conditional_log_likelihoods_as_score_matrix(
        dictTextTrain,
        inf_par_evaluation,
        class2Index
    )
    inf_par_log_class_conditional_probabilities = numpy.vstack([inf_par_log_class_conditional_probabilities[0:1, :],
                                                                inf_par_log_class_conditional_probabilities[2:3, :]])
    inf_par_multiclass_posterior_probability = numpy.exp(compute_log_posterior_probability(inf_par_log_class_conditional_probabilities,
                                                                                   numpy.array(
                                                                                       [1. / 2., 1. / 2.])))
    LTE_inf_par = numpy.hstack([LTE_inf, LTE_par])
    inf_mask = (LTE_inf_par == class2Index['inferno'])
    par_mask = (LTE_inf_par == class2Index['paradiso'])
    LTE_inf_par[par_mask] = 1
    inf_par_prediction = numpy.argmax(inf_par_multiclass_posterior_probability, axis=0)
    inf_par_prediction_accuracy = compute_prediction_accuracy(inf_par_prediction, LTE_inf_par)
    print("Inferno-Paradiso prediction accuracy %.2f" % inf_par_prediction_accuracy)



    # Per-2-classes accuracy
    # Paradiso-Purgatorio

    par_pur_evaluation = pur_evaluation + par_evaluation

    par_pur_log_class_conditional_probabilities = compute_class_conditional_log_likelihoods_as_score_matrix(
        dictTextTrain,
        par_pur_evaluation,
        class2Index
    )
    par_pur_log_class_conditional_probabilities = numpy.vstack([par_pur_log_class_conditional_probabilities[1:2, :],
                                                                par_pur_log_class_conditional_probabilities[2:3, :]])
    par_pur_posterior_probability = numpy.exp(compute_log_posterior_probability(par_pur_log_class_conditional_probabilities,
                                                                                   numpy.array(
                                                                                       [1. / 2., 1. / 2.])))
    LTE_par_pur = numpy.hstack([LTE_pur, LTE_par])
    pur_mask = (LTE_par_pur == class2Index['purgatorio'])
    par_mask = (LTE_par_pur == class2Index['paradiso'])
    LTE_par_pur[pur_mask] = 0
    LTE_par_pur[par_mask] = 1
    par_pur_prediction = numpy.argmax(par_pur_posterior_probability, axis=0)
    par_pur_prediction_accuracy = compute_prediction_accuracy(par_pur_prediction, LTE_par_pur)
    print("Paradiso-Purgatorio prediction accuracy %.2f" % par_pur_prediction_accuracy)

    # Per-2-classes accuracy
    # Inferno-Purgatorio

    inf_pur_evaluation = inf_evaluation + pur_evaluation
    inf_pur_log_class_conditional_probabilities = compute_class_conditional_log_likelihoods_as_score_matrix(
        dictTextTrain,
        inf_pur_evaluation,
        class2Index
    )
    inf_pur_log_class_conditional_probabilities = numpy.vstack([inf_pur_log_class_conditional_probabilities[0:1, :],
                                                                inf_pur_log_class_conditional_probabilities[1:2, :]])
    inf_pur_posterior_probability = numpy.exp(
        compute_log_posterior_probability(inf_pur_log_class_conditional_probabilities,
                                          numpy.array(
                                              [1. / 2., 1. / 2.])))

    LTE_inf_pur = numpy.hstack([LTE_inf, LTE_pur])
    inf_pur_prediction = numpy.argmax(inf_pur_posterior_probability, axis=0)
    inf_pur_prediction_accuracy = compute_prediction_accuracy(inf_pur_prediction, LTE_inf_pur)
    print("Inferno-Purgatorio prediction accuracy %.2f" % inf_pur_prediction_accuracy)






import itertools
from Project import functions
from Project.Classifiers import classifiers
import numpy
from utils import *
from itertools import compress
from GMM_models import *



def GMM_train_with_K_fold(DTR, LTR,iterations,path, K=5, seed=27):
    # Setting up all data needed
    D, L, idx = split_data(DTR, LTR, K, seed)
    mask = numpy.array([False for _ in range(K)])
    scores = numpy.zeros(LTR.shape[0])
    n_folds = LTR.shape[0] // K
    labels_training = LTR[idx]

    # Setting up the folder name
    # folder_descr, path = self.__define_folder_name(True)

    for i in range(K):
        mask[i] = True

        DTE = numpy.array(list(compress(D, mask))).reshape(-1, D[0].shape[1])
        DTR = numpy.hstack(numpy.array(list(compress(D, ~mask))))
        LTE = numpy.array(list(compress(L, mask))).ravel()
        LTR = numpy.hstack(numpy.array(list(compress(L, ~mask))))

        # Apply the model selected for the pipe with training and save the score
        model = GMMclass(iterations)
        model.set_attributes(DTR, LTR, DTE, LTE)
        model.train()
        model.compute_scores()
        scores[i * n_folds: (i + 1) * n_folds] = model.scores
        mask[i] = False
    model.scores = scores

    debug_print_information(model, labels_training,path,iterations)
    return model.scores


def GMMTied_train_with_K_fold(DTR, LTR,iterations,path, K=5, seed=27):
    # Setting up all data needed
    D, L, idx = split_data(DTR, LTR, K, seed)
    mask = numpy.array([False for _ in range(K)])
    scores = numpy.zeros(LTR.shape[0])
    n_folds = LTR.shape[0] // K
    labels_training = LTR[idx]

    # Setting up the folder name
    # folder_descr, path = self.__define_folder_name(True)

    for i in range(K):
        mask[i] = True

        DTE = numpy.array(list(compress(D, mask))).reshape(-1, D[0].shape[1])
        DTR = numpy.hstack(numpy.array(list(compress(D, ~mask))))
        LTE = numpy.array(list(compress(L, mask))).ravel()
        LTR = numpy.hstack(numpy.array(list(compress(L, ~mask))))

        # Apply the model selected for the pipe with training and save the score
        model = GMMTied(iterations)
        model.set_attributes(DTR, LTR, DTE, LTE)
        model.train()
        model.compute_scores()
        scores[i * n_folds: (i + 1) * n_folds] = model.scores
        mask[i] = False
    model.scores = scores

    debug_print_information(model, labels_training, path, iterations)
    return model.scores


def GMMDiagonal_train_with_K_fold(DTR, LTR,iterations,path, K=5, seed=27):
    # Setting up all data needed
    D, L, idx = split_data(DTR, LTR, K, seed)
    mask = numpy.array([False for _ in range(K)])
    scores = numpy.zeros(LTR.shape[0])
    n_folds = LTR.shape[0] // K
    labels_training = LTR[idx]

    # Setting up the folder name
    # folder_descr, path = self.__define_folder_name(True)

    for i in range(K):
        mask[i] = True

        DTE = numpy.array(list(compress(D, mask))).reshape(-1, D[0].shape[1])
        DTR = numpy.hstack(numpy.array(list(compress(D, ~mask))))
        LTE = numpy.array(list(compress(L, mask))).ravel()
        LTR = numpy.hstack(numpy.array(list(compress(L, ~mask))))

        # Apply the model selected for the pipe with training and save the score
        model = GMMDiagonal(iterations)
        model.set_attributes(DTR, LTR, DTE, LTE)
        model.train()
        model.compute_scores()
        scores[i * n_folds: (i + 1) * n_folds] = model.scores
        mask[i] = False
    model.scores = scores

    debug_print_information(model, labels_training,path,iterations)
    return model.scores

def GMMTiedDiagonal_train_with_K_fold(DTR, LTR,iterations,path, K=5, seed=27):
    # Setting up all data needed
    D, L, idx = split_data(DTR, LTR, K, seed)
    mask = numpy.array([False for _ in range(K)])
    scores = numpy.zeros(LTR.shape[0])
    n_folds = LTR.shape[0] // K
    labels_training = LTR[idx]

    # Setting up the folder name
    # folder_descr, path = self.__define_folder_name(True)

    for i in range(K):
        mask[i] = True

        DTE = numpy.array(list(compress(D, mask))).reshape(-1, D[0].shape[1])
        DTR = numpy.hstack(numpy.array(list(compress(D, ~mask))))
        LTE = numpy.array(list(compress(L, mask))).ravel()
        LTR = numpy.hstack(numpy.array(list(compress(L, ~mask))))

        # Apply the model selected for the pipe with training and save the score
        model = GMMTiedDiagonal(iterations)
        model.set_attributes(DTR, LTR, DTE, LTE)
        model.train()
        model.compute_scores()
        scores[i * n_folds: (i + 1) * n_folds] = model.scores
        mask[i] = False
    model.scores = scores

    debug_print_information(model, labels_training, path, iterations)
    return model.scores

def debug_print_information(model, labels,path,iterations):
    predicted_labels = np.where(model.scores > 0, 1, 0)
    err = (1 - (labels == predicted_labels).sum() / labels.size) * 100
    cost_0_5 = str(round(compute_minimum_NDCF(model.scores, labels, 0.5, 1, 1)[0], 3))
    cost_0_1 = str(round(compute_minimum_NDCF(model.scores, labels, 0.1, 1, 1)[0], 3))
    cost_0_9 = str(round(compute_minimum_NDCF(model.scores, labels, 0.9, 1, 1)[0], 3))

    with open("./dati/"+path+"_"+str(2 ** iterations), "w") as fp:
        fp.write("Error rate for this training is "+ str(round(err, 2)) + "%")
        fp.write('\n')
        fp.write("minDCF with pi=0.5 "+ cost_0_5)
        fp.write('\n')
        fp.write("minDCF with pi=0.1 "+ cost_0_1)
        fp.write('\n')
        fp.write("minDCF with pi=0.9 "+ cost_0_9 )


def plot_graph(scores, labels):

    cost_0_5 = str(round(compute_minimum_NDCF(scores, labels, 0.5, 1, 1)[0], 3))
    cost_0_1 = str(round(compute_minimum_NDCF(scores, labels, 0.1, 1, 1)[0], 3))
    cost_0_9 = str(round(compute_minimum_NDCF(scores, labels, 0.9, 1, 1)[0], 3))



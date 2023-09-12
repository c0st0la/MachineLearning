
import itertools
from Project import functions
from Project.Classifiers import classifiers
import numpy
from utils import *
from itertools import compress
from GMM_models import *

def train_single_gmm(DTR, LTR, DTE, LTE, iterations):
    print("========================RAW============================")
    make_train_with_K_fold(DTR, LTR, DTE, LTE, iterations)




def make_train_with_K_fold(DTR, LTR, DTE, LTE, iterations, K=5, seed=27, calibration=False, fusion=False,
                           model_calibrated=None, fusion_desc=None):
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
    return model.scores

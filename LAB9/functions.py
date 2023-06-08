import numpy
import matplotlib.pyplot as plt
from itertools import combinations

import numpy as np
import scipy
import numpy
import sklearn.datasets

FILEPATH = 'Solution/iris.csv'


def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L


def to_column(array):
    return array.reshape((array.size, 1))

def to_row(array):
    return array.reshape((1, array.size))


def load_iris_datasets_from_file(filename):
    """

    :param filename: the csv file
    :return: It returns a pair. The first element is the dataset, the second one is the labels
    """
    class_mapping = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    sample_mapping = list()
    D_list = list()
    with open(filename, "r") as fp:
        for line in fp:
            attrs = line.rstrip().split(',')[0:4]
            D_list.append(to_column(numpy.array([float(i) for i in attrs])))
            sample_mapping.append(class_mapping[line.rstrip().split(',')[4]])
    sample_mapping = numpy.array(sample_mapping, dtype=numpy.int32)
    return numpy.hstack(D_list), sample_mapping


def filter_dataset_by_labels(D, labels):
    """

    :param D: It is the dataset to filter
    :param labels: It is the array containing the labels of the dataset
    :return: the dataset D filtered by the labels provided
    """
    mask_setosa = (labels == 0)
    mask_versicolor = (labels == 1)
    mask_virginica = (labels == 2)
    return D[:, mask_setosa], D[:, mask_versicolor], D[:, mask_virginica]


def plot_scatter_attributes_X_label(D_setosa, D_versicolor, D_virginica, title=""):
    plt.figure()
    plt.scatter(D_setosa[0, :], D_setosa[1, :], label="Iris_setosa")
    plt.scatter(D_versicolor[0, :], D_versicolor[1, :], label="Iris_versicolor")
    plt.scatter(D_virginica[0, :], D_virginica[1, :], label="Iris_virginica")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_hist_attributes_X_label(D_setosa, D_versicolor, D_virginica, attributes):
    """
    It plots the datasets using histograms plotting type for each of the attribute present in the
    attributes parameter
    :param D1_setosa:
    :param D1_versicolor:
    :param D1_virginica:
    :param attributes: It contains the attribute of the dataset
    :return: None
    """
    for i in range(len(attributes)):
        plt.figure()
        plt.hist(D_setosa[i, :], bins=10, ec='black', density=True, alpha=0.3, label='Iris_Setosa')
        plt.hist(D_versicolor[i, :], bins=10, ec='black', density=True, alpha=0.3, label='Iris_Versicolor')
        plt.hist(D_virginica[i, :], bins=10, ec='black', density=True, alpha=0.3, label='Iris_Virginica')
        plt.legend()
        plt.xlabel(attributes[i])
        plt.show()


def plot_scatter_pair_attributes_values(D_setosa, D_versicolor, D_virginica, attributes):
    comb = list(combinations([i for i in range(len(attributes))], 2))
    for x, y in comb:
        plt.figure()
        plt.scatter(D_setosa[x, :], D_setosa[y, :], label='Iris_Setosa')
        plt.scatter(D_versicolor[x, :], D_versicolor[y, :], label='Iris_Versicolor')
        plt.scatter(D_virginica[x, :], D_virginica[y, :], label='Iris_Virginica')
        plt.legend()
        plt.xlabel(attributes[x])
        plt.ylabel(attributes[y])
        plt.show()


def center_data(D):
    mu = to_column(D.mean(axis=1))
    DC = D - mu
    return DC


def compute_PCA(D, sub_dimension):
    """
    Allows reducing the dimensionality of a dataset by projecting the data over the principal components (PCA)
    :param D: It is the dataset. If  D has no zero mean it must be centered
    :param sub_dimension: is the dimension of the subspace. It can be chosen between 1 <= x <= D.shape[0]
    :return: the reduced dimension dataset
    """
    DC = center_data(D)
    # The dataset has D.shape=(n,m)
    # C is the covariance and will have a C.shape=(n,n)
    C = compute_covariance(DC)
    # These are two different but equivalent ways to compute the subspace given...
    # ...the covariance matrix C. You call one or the other depending on the...
    # ...features of the matrix C (see the description inside the function implementation)
    P1 = compute_subspace(C, sub_dimension)
    P2 = compute_subspace_svd(C, sub_dimension)
    # numpy.dot is a matrix multiplication. I am projecting the dataset on the subspace
    DP = numpy.dot(P1.T, D)
    return DP


def compute_covariance(D):
    """
    :param D: dataset
    :return: the covariance of the dataset given as input
    """
    covariance = numpy.dot(D, D.T) / float(D.shape[1])
    return covariance


def compute_empirical_covariance(D):
    """
     It performs the covariance of the matrix D after it has been centered on its mean
     :param D: dataset
     :return: the empirical covariance of the dataset given as input
     """
    D = center_data(D)
    covariance = numpy.dot(D, D.T) / float(D.shape[1])
    return covariance


def compute_mean(D):
    mu = to_column(D.mean(axis=1))
    return mu


def compute_subspace(C, sub_dimension):
    """

    :param C: the covariance matrix. Must be a square matrix with C.shape = (n,n). C must be a square matrix with C.shape = (n,n)
    :param sub_dimension: the sub_dimension desired
    :return: the subspace
    """

    # s is an array with the eigenvalues (sorted from smallest to largest) of the matrix C...
    # ...each eigenvalue corresponds to the variance of the corresponding axis
    # linalg.eigh can be called only a square symmetric matrix...
    # ...if you have a generic square matrix you have to call linalg.eig
    s, U = numpy.linalg.eigh(C)
    # U is a square matrix U.shape=(n,n)
    # The subspace solution corresponds to the sub_dimension columns of U corresponding to the...
    # ...sub_dimension highest eigenvalue
    # s is sorted in ascending order. That's why U is first column reversed
    P = U[:, ::-1][:, 0:sub_dimension]
    return P


def compute_subspace_svd(C, sub_dimension):
    """
    Singular Value Decomposition can be used also on rectangular matrix (n,m)
    Since C is semi-definitive positive we can use the Singular Value Decomposition
    with linalg.svd. the vector of eigenvalues 's' is sorted in descending order
    :param C: the covariance matrix
    :param sub_dimension:
    :return: the subspace
    """
    # The vector of eigenvalues 's' is sorted in descending order
    U, s, V_transposed = numpy.linalg.svd(C)
    # U is orthogonal and U.shape = (n, n)
    # V is orthogonal and V.shape = (m, m)
    # s has sub_dimension eigenvalues
    P = U[:, 0:sub_dimension]
    return P


def compute_LDA_generalized_eigenvalue(D, class_mapping, L, directions):
    """

    :param D: dataset
    :param class_mapping: It is a dictionary where each class is mapped to a number
    :param L: labels
    :param directions: the number of directions of reduction
    :return: the reduced database
    """

    Covariance_between = compute_between_covariance(D, class_mapping, L)
    Covariance_within = compute_within_covariance(D, class_mapping, L)
    # scipy.linalg.eigh solves the generalized eigenvalue problem for hermitian...
    # ...(including real symmetric) matrix
    # We can use scipy.linalg.eigh on positive definite matrices...
    # ...(those matrix which have all eigenvalues > 0)...
    # ...You can verify it calling --> s, U = numpy.linalg.eigh(Covariance_within)
    # ...and performing a check over the vector of eigenvalues 's'
    s, U = scipy.linalg.eigh(Covariance_between, Covariance_within)
    W = U[:, ::-1][:, 0:directions]
    # This is a way to find the basis of the space W. This step is not mandatory
    # UW is the basis of W
    UW, _, _ = numpy.linalg.svd(W)
    U = UW[:, 0:directions]
    DP = numpy.dot(W.T, D)
    return DP


def compute_within_covariance(D, L, labels):
    Covariance_within = 0
    tot_samples = 0
    for label in labels:
        # D_class is the dataset filtered by class
        D_class = D[:, L == label]
        DC_class = center_data(D_class)
        num_samples = D_class.shape[1]
        tot_samples += num_samples
        Covariance_within = Covariance_within + numpy.dot(DC_class, DC_class.T)
    Covariance_within = Covariance_within / float(tot_samples)
    return Covariance_within


def compute_between_covariance(D, class_mapping, samples_mapped_to_class):
    mu = to_column(D.mean(1))
    labels = class_mapping.values()
    Covariance_between = 0
    tot_samples = 0
    for label in labels:
        D_class = D[:, samples_mapped_to_class == label]
        mu_class = to_column(D_class.mean(1))
        num_samples = D_class.shape[1]
        tot_samples += num_samples
        Covariance_between = Covariance_between + num_samples * numpy.dot((mu_class - mu), (mu_class - mu).T)
    Covariance_between = Covariance_between / float(tot_samples)
    return Covariance_between


def compute_LDA_generalized_eigenvalue_by_joint_diagonalization(D, class_mapping, L, directions):
    """
    It solves the generalized eigenvalue by joint diagonalization of the betweeen and within covariance matrices
    :param D: Dataset
    :param class_mapping: It is a dictionary where each class is mapped to a number
    :param L: labels
    :param directions: the number of directions of reduction
    :return: the reduced database
    """
    Covariance_between = compute_between_covariance(D, class_mapping, L)
    Covariance_within = compute_within_covariance(D, class_mapping, L)
    # Since Covariance_within is semi-definitive positive we can use the Singular Value Decomposition
    # with linalg.svd the vector of eigenvalues is sorted in descending order
    Uw, sw, _ = numpy.linalg.svd(Covariance_within)
    Pw = numpy.dot(numpy.dot(Uw, numpy.diag(1.0 / (sw ** 0.5))), Uw.T)
    Covariance_between_transformed = numpy.dot(numpy.dot(Pw, Covariance_between), Pw.T)
    Ub, sb, _ = numpy.linalg.svd(Covariance_between_transformed)
    Pb = Ub[:, 0:directions]
    W = numpy.dot(Pw.T, Pb)
    # This is a way to find the basis of the space W. This step is not mandatory
    # U is the basis
    UW, _, _ = numpy.linalg.svd(W)
    U = UW[:, 0:directions]
    DP = numpy.dot(W.T, D)

    # sw, Uw = numpy.linalg.eigh(Covariance_within)
    # Pw = numpy.dot(numpy.dot(Uw, numpy.diag(1.0/(sw**0.5))), Uw.T)
    # Covariance_between_transformed = numpy.dot(numpy.dot(Pw, Covariance_between), Pw.T)
    # sb, Ub = numpy.linalg.eigh(Covariance_between_transformed)
    # Pb = Ub[:, ::-1][:, 0:directions]
    # W = numpy.dot(Pw.T, Pb)
    # DP = numpy.dot(W.T, D)
    return DP


def log_probab_distr_func_GAU_matrix(D, mu, C):
    """
    It computes the log Multivariate Gaussina densitiy given a matrix and its model parameter (mean and covariance)
    :param D: is the Dataset
    :param mu: is the means of the dataset D
    :param C: is the covariance
    :return: log Multivariate Gaussina densitiy
    """
    Y = list()
    samples = [to_column(D[:, i]) for i in range(0, D.shape[1])]
    featureVectorSize = D.shape[0]
    for sample in samples:
        _, logDeterminantCovariance = numpy.linalg.slogdet(C)
        inverseCovariance = numpy.linalg.inv(C)
        log_MVG = -(featureVectorSize / 2) * numpy.log(2 * numpy.pi) - 0.5 * logDeterminantCovariance - \
                  0.5 * numpy.dot(numpy.dot((sample - mu).T, inverseCovariance), (sample - mu))
        Y.append(log_MVG)
    Y = numpy.array(Y)
    Y = Y.ravel()
    return Y


def log_likelihood(D):
    muML = compute_mean(D)
    CML = compute_empirical_covariance(D)
    arrayLogLikelihood = log_probab_distr_func_GAU_matrix(D ,muML, CML)
    LogLikelihood = numpy.sum(arrayLogLikelihood)
    return LogLikelihood


def split_db_to_train_test(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def maximum_likelihood_MVG_parameter_estimator(D):
    # This function returns the parameters that maximize the likelihood
    # it returns a tuple : the first param is the empirical mean...
    # ... the second param is the empirical covariance matrix
    mu = to_column(D.mean(1))
    DC = center_data(D)
    C = compute_covariance(DC)
    return mu, C


def logpdf_MVG_vector(x, mu, C):
    # x is the sample vector with size N
    # mu is the mean of the matrix from which the vector belongs to. mu.shape = (N, 1)
    # C is the covariance matrix from which the vector belongs to. C.shape = (N, N)
    N = x.size
    _, logC_determinant = numpy.linalg.slogdet(C)
    # C_inverse is also called precision matrix
    C_inverse = numpy.linalg.inv(C)
    log_pdf = - ((N / 2) * numpy.log(2 * numpy.pi))
    log_pdf = log_pdf - ((1 / 2) * logC_determinant)
    log_pdf = log_pdf - ((1 / 2) * numpy.dot((x - mu).T, numpy.dot(C_inverse, (x - mu))))
    return log_pdf


def logpdf_MVG_matrix(D, mu, C, label, class_conditional_probabilities):
    num_columns = D.shape[1]
    for column_index in range(num_columns):
        a = logpdf_MVG_vector(to_column(D[:, column_index]), mu, C)
        class_conditional_probabilities[label, column_index] = a


def compute_MVG_log_likelihood_as_score_matrix(DTR, LTR, DTE, labels):
    # This function return the score (e.g the class conditional probabilities/log_likehood f(x|C) )...
    # ...Each row of the score matrix corresponds to a class, and contains...
    # ...the conditional log-likelihoods for all the samples for that class

    # The variable class_conditional_probabilities is built on the TEST dataset
    class_conditional_probabilities = numpy.zeros((len(labels), DTE.shape[1]))
    for label in labels:
        D_filtered = DTR[:, LTR == label]
        # I have to estimate the parameter using the TRAINING dataset
        mu, C = maximum_likelihood_MVG_parameter_estimator(D_filtered)
        logpdf_MVG_matrix(DTE, mu, C, label, class_conditional_probabilities)
    return class_conditional_probabilities


def compute_posterior_probability(class_conditional_probabilities, class_prior_probability):
    """
        This function compute the posterior probability given the class conditional probabilities
        and the class prior probabilities
    """
    # SJoint is the joint distribution for each sample of each class
    SJoint = class_conditional_probabilities * class_prior_probability
    # We are summing all the joint probability belonging to the sum class. I am summing over the horizontal axis...
    # In each row we have the joint density belonging to a specific class...
    # ...(row 0 -> class 0, row 1 -> class 1 , .... row n -> class n)
    SMarginal = to_row(SJoint.sum(0))
    # Now we compute the class_posterior probability
    SPosterior = SJoint / SMarginal
    return SPosterior


def compute_log_posterior_probability(log_class_conditional_probabilities, class_prior_probability):
    """
            This function compute the log posterior probability given the log class conditional probabilities
            and the class prior probabilities
        """
    log_SJoint = 0
    for i in range(class_prior_probability.size):
        if i == 0:
            log_SJoint = to_row(log_class_conditional_probabilities[i] + numpy.log(class_prior_probability[i]))
        else:
            log_SJoint = numpy.vstack(
                (log_SJoint, log_class_conditional_probabilities[i] + numpy.log(class_prior_probability[i])))
    log_SMarginal = to_row(scipy.special.logsumexp(log_SJoint, axis=0))
    log_posterior_probability = log_SJoint - log_SMarginal
    return log_posterior_probability


def compute_NB_log_likelihood_as_score_matrix(DTR, LTR, DTE, labels):
    # This function return the score (e.g the class conditional probabilities/log_likehood f(x|C) )...
    # ...Each row of the score matrix corresponds to a class, and contains...
    # ...the conditional log-likelihoods for all the samples for that class

    # The variable class_conditional_probabilities is built on the TEST dataset
    class_conditional_probabilities = numpy.zeros((len(labels), DTE.shape[1]))
    for label in labels:
        D_filtered = DTR[:, LTR == label]
        # I have to estimate the parameter using the TRAINING dataset
        mu, C = maximum_likelihood_NB_parameter_estimator(D_filtered)
        logpdf_MVG_matrix(DTE, mu, C, label, class_conditional_probabilities)
    return class_conditional_probabilities


def compute_TC_log_likelihood_as_score_matrix(DTR, LTR, DTE, labels):
    """
     This function return the score (e.g the class conditional probabilities/log_likehood f(x|C) )
     Each row of the score matrix corresponds to a class, and contains
     the conditional log-likelihoods for all the samples for that class
    """

    # The variable class_conditional_probabilities is built on the TEST dataset
    class_conditional_probabilities = numpy.zeros((len(labels), DTE.shape[1]))
    # C is the within covariance matrix (one of the model parameters)
    C = compute_within_covariance(DTR, LTR, labels)
    for label in labels:
        D_filtered = DTR[:, LTR == label]
        mu = to_column(D_filtered.mean(1))
        logpdf_MVG_matrix(DTE, mu, C, label, class_conditional_probabilities)
    return class_conditional_probabilities


def maximum_likelihood_NB_parameter_estimator(D):
    """
        This function returns the parameters that maximize the likelihood
        it returns a tuple : the first param is the empirical mean
        the second param is the empirical covariance matrix
    """

    mu = to_column(D.mean(1))
    DC = center_data(D)
    C = compute_covariance(DC)
    # The covariance matrix is the diagonal of the original covariance matrix obtained just before...
    # ...since the features are small we can compute it multiplying by the identity matrix.
    # Otherwise, you have to implement ad-hoc function
    C = C * numpy.identity(C.shape[1])
    return mu, C


def compute_TNB_log_likelihood_as_score_matrix(DTR, LTR, DTE, labels):
    """
         This function return the score (e.g the class conditional probabilities/log_likehood f(x|C) )
         Each row of the score matrix corresponds to a class, and contains
         the conditional log-likelihoods for all the samples for that class
    """

    # The variable class_conditional_probabilities is built on the TEST dataset
    class_conditional_probabilities = numpy.zeros((len(labels), DTE.shape[1]))
    # C is the within covariance matrix (one of the model parameters)
    C = compute_within_covariance(DTR, LTR, labels)
    # The covariance matrix is the diagonal of the original covariance matrix obtained just before...
    # ...since the features are small we can compute it multiplying by the identity matrix.
    # Otherwise, you have to implement ad-hoc function
    C = C * numpy.identity(C.shape[1])
    for label in labels:
        D_filtered = DTR[:, LTR == label]
        mu = to_column(D_filtered.mean(1))
        logpdf_MVG_matrix(DTE, mu, C, label, class_conditional_probabilities)
    return class_conditional_probabilities


def K_fold_cross_validation(D, L, classifier, k, class_prior_probability, labels):
    """
        This function perform a K-fold cross validation.
        You have to indicate the name of the classifier you want to use between:
        MVG, NB, TC, TNB.
        k is the number of folds
        D is the dataset
        L is the class mapping of the dataset
    """
    num_samples = int(D.shape[1] / k)
    tot_accuracy = 0
    tot_error_rate = 0
    if classifier == "MVG":
        for i in range(k):
            (DTR, LTR), (DTE, LTE) = K_fold_generate_Training_and_Testing_samples(D, L, i, k, num_samples)
            log_MVG_class_conditional_probabilities = compute_MVG_log_likelihood_as_score_matrix(DTR, LTR, DTE, labels)
            log_MVG_posterior_probability = compute_log_posterior_probability(log_MVG_class_conditional_probabilities, class_prior_probability)
            MVG_posterior_probability = numpy.exp(log_MVG_posterior_probability)
            MVG_predictions = numpy.argmax(MVG_posterior_probability, axis=0)
            MVG_prediction_accuracy = compute_prediction_accuracy(MVG_predictions, LTE)
            MVG_error_rate = compute_error_rate(MVG_predictions, LTE)
            tot_accuracy += MVG_prediction_accuracy
            tot_error_rate += MVG_error_rate

    elif classifier == "NB":
        for i in range(k):
            (DTR, LTR), (DTE, LTE) = K_fold_generate_Training_and_Testing_samples(D, L, i, k, num_samples)
            log_NB_class_conditional_probabilities = compute_NB_log_likelihood_as_score_matrix(DTR, LTR, DTE, labels)
            log_NB_posterior_probability = compute_log_posterior_probability(log_NB_class_conditional_probabilities,
                                                                              class_prior_probability)
            NB_posterior_probability = numpy.exp(log_NB_posterior_probability)
            NB_predictions = numpy.argmax(NB_posterior_probability, axis=0)
            NB_prediction_accuracy = compute_prediction_accuracy(NB_predictions, LTE)
            NB_error_rate = compute_error_rate(NB_predictions, LTE)
            tot_accuracy += NB_prediction_accuracy
            tot_error_rate += NB_error_rate

    elif classifier == "TC":
        for i in range(k):
            (DTR, LTR), (DTE, LTE) = K_fold_generate_Training_and_Testing_samples(D, L, i, k, num_samples)
            log_TC_class_conditional_probabilities = compute_TC_log_likelihood_as_score_matrix(DTR, LTR, DTE, labels)
            log_TC_posterior_probability = compute_log_posterior_probability(log_TC_class_conditional_probabilities,
                                                                             class_prior_probability)
            TC_posterior_probability = numpy.exp(log_TC_posterior_probability)
            TC_predictions = numpy.argmax(TC_posterior_probability, axis=0)
            TC_prediction_accuracy = compute_prediction_accuracy(TC_predictions, LTE)
            TC_error_rate = compute_error_rate(TC_predictions, LTE)
            tot_accuracy += TC_prediction_accuracy
            tot_error_rate += TC_error_rate

    elif classifier == "TNB":
        for i in range(k):
            (DTR, LTR), (DTE, LTE) = K_fold_generate_Training_and_Testing_samples(D, L, i, k, num_samples)
            log_TNB_class_conditional_probabilities = compute_TNB_log_likelihood_as_score_matrix(DTR, LTR, DTE, labels)
            log_TNB_posterior_probability = compute_log_posterior_probability(log_TNB_class_conditional_probabilities,
                                                                             class_prior_probability)
            TNB_posterior_probability = numpy.exp(log_TNB_posterior_probability)
            TNB_predictions = numpy.argmax(TNB_posterior_probability, axis=0)
            TNB_prediction_accuracy = compute_prediction_accuracy(TNB_predictions, LTE)
            TNB_error_rate = compute_error_rate(TNB_predictions, LTE)
            tot_accuracy += TNB_prediction_accuracy
            tot_error_rate += TNB_error_rate

    else:
        print("The given classifier %s is not recognized!" % classifier)
        exit(-1)
    tot_accuracy = tot_accuracy / k
    tot_error_rate = tot_error_rate / k
    return tot_accuracy, tot_error_rate



def K_fold_generate_Training_and_Testing_samples(D, L, i, k, num_samples, seed=0):
    numpy.random.seed(seed)
    idx = numpy.arange(0, D.shape[1])
    if i == 0:
        idxTest = idx[:num_samples]
        idxTrain = idx[num_samples:]
    elif 0 < i < k - 1:
        idxTrain1 = idx[:(i * num_samples)]
        idxTest = idx[(i * num_samples):((i + 1) * num_samples)]
        idxTrain2 = idx[((i + 1) * num_samples):]
        idxTrain = numpy.hstack((idxTrain1, idxTrain2))
    elif i == k - 1:
        idxTrain = idx[:i * num_samples]
        idxTest = idx[i * num_samples:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def compute_prediction_accuracy(predictions, LTE):
    correct_predictions = (LTE == predictions).sum()
    accuracy = correct_predictions / float(LTE.size)
    return accuracy


def compute_error_rate(predictions, LTE):
    correct_predictions = (LTE == predictions).sum()
    accuracy = correct_predictions / float(LTE.size)
    return 1 - accuracy


def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


class logRegClass:

    def __init__(self, DTR, LTR, l, feature_space_dimension=None, num_classes=None):
        self.DTR = DTR
        self.LTR = LTR
        # This is lambda (the hyper parameter)
        self.l = l
        self.feature_space_dimension = feature_space_dimension
        self.num_classes = num_classes

    def log_reg_obj_bin(self, v):
        w = v[0:-1]
        b = v[-1]
        regularization_term = (self.l / 2) * (numpy.power(numpy.linalg.norm(w), 2))
        objective_function = 0
        for i in range(self.DTR.shape[1]):
            sample = self.DTR[:, i]
            label = self.LTR[i]
            objective_function = objective_function + numpy.logaddexp(0,
                                                                      -(2 * label - 1) * (numpy.dot(w.T, sample) + b))
        objective_function = objective_function / self.DTR.shape[1]
        objective_function = regularization_term + objective_function
        return objective_function

    def log_reg_obj_multiclass(self, v):
        W = v[0:-self.num_classes]
        b = v[-self.num_classes:]
        W = W.reshape((self.feature_space_dimension, self.num_classes))
        regularization_term = (self.l / 2.0) * (W * W).sum()
        z = numpy.zeros((self.num_classes, self.LTR.size))
        for row_index in range(z.shape[0]):
            for col_index in range(z.shape[1]):
                if self.LTR[col_index] == row_index:
                    z[row_index, col_index] = 1
        score_matrix = numpy.dot(W.T, self.DTR) + to_column(b)
        log_sum_exp = scipy.special.logsumexp(score_matrix, axis=0)
        Y_log = score_matrix - log_sum_exp
        objective_function = 0
        for col_index in range(self.DTR.shape[1]):
            for row_index in range(self.num_classes):
                objective_function = objective_function + (z[row_index, col_index] * Y_log[row_index, col_index])
        objective_function = objective_function / self.DTR.shape[1]
        return regularization_term - objective_function


def logistic_regression_binary(DTR, LTR, DTE, LTE, l):
    logReg = logRegClass(DTR, LTR, l)

    x, f, d = scipy.optimize.fmin_l_bfgs_b(logReg.log_reg_obj_bin, x0=numpy.zeros(DTR.shape[0] + 1), approx_grad=True)
    print("The objective value at the minimum is %f" % f)
    w = x[0:-1]
    b = x[-1]
    posterior_log_likelihood_ratio = numpy.dot(w.T, DTE) + b
    predictions = numpy.zeros(posterior_log_likelihood_ratio.size)
    for index in range(posterior_log_likelihood_ratio.size):
        if posterior_log_likelihood_ratio[index] > 0:
            predictions[index] = 1
    predictions_accuracy = compute_prediction_accuracy(predictions, LTE)
    print("The accuracy is %.3f" % predictions_accuracy)
    print("The error rate is %.3f" % (1 - predictions_accuracy))


def logistic_regression_multiclass(DTR, LTR, DTE, LTE, feature_space_dimension, num_classes, l):
    # l is lambda
    logReg = logRegClass(DTR, LTR, l, feature_space_dimension, num_classes)
    x0 = numpy.zeros(feature_space_dimension * num_classes + num_classes)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(logReg.log_reg_obj_multiclass,
                                           x0=x0,
                                           approx_grad=True)
    print("The objective value at the minimum is %.6f" % f)
    W = x[0:-num_classes]
    W = W.reshape((4, 3))
    b = x[-num_classes:]
    class_posterior_probability = numpy.zeros((num_classes, DTE.shape[1]))
    for col_index in range(class_posterior_probability.shape[1]):
        sample_test = DTE[:, col_index]
        class_posterior_probability[:, col_index] = numpy.dot(W.T, sample_test) + b
    predictions = numpy.argmax(class_posterior_probability, axis=0)
    predictions_accuracy = compute_prediction_accuracy(predictions, LTE)
    print("The accuracy is %.3f" % predictions_accuracy)
    print("The error rate is %.3f" % (1 - predictions_accuracy))


## NEW FROM HERE

def compute_confusion_matrix(predictions, L):
    numLabels = numpy.max(L)+1
    matrix = numpy.zeros((numLabels, numLabels))
    for index, prediction in enumerate(predictions):
        matrix[prediction, L[index]] += 1
    return matrix


def compute_optimal_bayes_decision(logLikelihoodRatios, priorsProbability, costs):
    """

    :param logLikelihoodRatio:  ll of class 1 over ll of class 0
    :param priorsProbability: index 0 contains priors of class 0, index 1 contatins priors of class 1
    :param costs: index 0 contains cost of false negative, index 1 contains cost of false postive
    :return:
    """
    predictions = np.zeros((logLikelihoodRatios.size), dtype=np.int8)
    threshold = - np.log((priorsProbability[1]*costs[0])/(priorsProbability[0]*costs[1]))
    for index, llr in enumerate(logLikelihoodRatios):
        if llr > threshold:
            # it is predicted as class 1
            predictions[index] = 1
        else:
            # it is predicted as class 0
            predictions[index] = 0
    return predictions


def compute_optimal_bayes_decision_given_threshold(logLikelihoodRatios, threshold):
    """

    :param logLikelihoodRatio:  ll of class 1 over ll of class 0
    :param priorsProbability: index 0 contains priors of class 0, index 1 contatins priors of class 1
    :param costs: index 0 contains cost of false negative, index 1 contains cost of false postive
    :return:
    """
    predictions = np.zeros((logLikelihoodRatios.size), dtype=np.int8)
    for index, llr in enumerate(logLikelihoodRatios):
        if llr > threshold:
            # it is predicted as class 1
            predictions[index] = 1
        else:
            # it is predicted as class 0
            predictions[index] = 0
    return predictions



def compute_binary_prediction_rates(confusionMatrix):
    FNR = confusionMatrix[0, 1]/(confusionMatrix[0, 1] + confusionMatrix[1, 1])
    FPR = confusionMatrix[1, 0]/(confusionMatrix[0, 0] + confusionMatrix[1, 0])
    TNR = confusionMatrix[0, 0]/(confusionMatrix[0, 0] + confusionMatrix[1, 0])
    TPR = confusionMatrix[1, 1] / (confusionMatrix[0, 1] + confusionMatrix[1, 1])
    return FNR, FPR, TNR, TPR


def compute_detection_cost_function(confusionMatrix, classPriorsProbability, costs):
    """

    :param confusionMatrix: confusionMatrix
    param priorsProbability: index 0 contains priors of class 0, index 1 contatins priors of class 1
    :param costs: index 0 contains cost of false negative, index 1 contains cost of false postive
    :return:
    """
    FNR, FPR, TNR, TPR = compute_binary_prediction_rates(confusionMatrix)
    return classPriorsProbability[1]*costs[0]*FNR + (1-classPriorsProbability[1])*costs[1]*FPR


def compute_normalized_detection_cost_function(confusionMatrix, classPriorsProbability, costs):
    """

    :param confusionMatrix: confusionMatrix
    param priorsProbability: index 0 contains priors of class 0, index 1 contatins priors of class 1
    :param costs: index 0 contains cost of false negative, index 1 contains cost of false postive
    :return:
    """
    DCF = compute_detection_cost_function(confusionMatrix, classPriorsProbability, costs)
    return DCF/min(classPriorsProbability[1]*costs[0], (1-classPriorsProbability[1])*costs[1])


def compute_missclassification_ratios(confusionMatrix):
    misClassification_ratios = numpy.zeros((confusionMatrix.shape[0],  confusionMatrix.shape[1]))
    for i in range(confusionMatrix.shape[0]):
            for j in range(confusionMatrix.shape[1]):
                misClassification_ratios[i ,j] = confusionMatrix[i, j]/(numpy.sum(confusionMatrix[:, j]))
    return misClassification_ratios


def compute_detection_cost_functio_by_misclassificationRatio(costs, misClassificationRatios, classPriorsProbability):
    DCF = 0
    for j in range(classPriorsProbability.size):
        sum = 0
        for i in range(classPriorsProbability.size):
            sum += costs[i, j]*misClassificationRatios[i, j]
        DCF += classPriorsProbability[j] * sum
    return DCF

class SVMClass:

    def __init__(self, DTR, LTR, DTE, K, C):
        self.DTR = numpy.vstack((DTR,K*numpy.ones((1,DTR.shape[1]))))
        self.LTR = LTR
        self.DTE = numpy.vstack((DTE,K*numpy.ones((1, DTE.shape[1]))))
        self.C = C
        self.K = K
        self.z = []
        for index in range(LTR.size):
            if LTR[index] == 1:
                self.z.append(1)
            else:
                self.z.append(-1)
        self.H = numpy.zeros((self.DTR.shape[1], self.DTR.shape[1]))
        for i in range(self.DTR.shape[1]):
            for j in range(self.DTR.shape[1]):
                self.H[i, j] = self.z[i] * self.z[j] * numpy.dot(self.DTR[:, i].T, self.DTR[:, j])


    def svm_dual_obj(self, alpha):
        ones = np.ones(self.DTR.shape[1])
        dualObjectiveFunction = 1/2 * np.dot(np.dot(alpha.T, self.H), alpha) - np.dot(alpha.T, ones)
        gradient = np.dot(self.H, alpha) - ones
        gradient = np.reshape(gradient, (alpha.size,))
        return dualObjectiveFunction, gradient

    def compute_primal_obj(self, w_optimal):
        primalObjectiveFunction = 0
        regularitazionTerm = 0.5 * (np.power(np.linalg.norm(w_optimal), 2))
        for i in range(self.DTR.shape[1]):
            primalObjectiveFunction += max(0, 1-self.z[i]*np.dot(w_optimal.T, self.DTR[:, i]))
        primalObjectiveFunction = regularitazionTerm + self.C * primalObjectiveFunction
        return primalObjectiveFunction

def support_vector_machine(DTR, LTR, DTE, LTE, K, C, threshold=0):
    svm = SVMClass(DTR, LTR, DTE, K, C)
    bounds = []
    for i in range(svm.DTR.shape[1]):
        bounds.append((0, svm.C))
    x, f, d = scipy.optimize.fmin_l_bfgs_b(svm.svm_dual_obj, x0= numpy.zeros(svm.DTR.shape[1]), fprime=None,
                                           bounds=bounds, factr=1.0)

    #print("The objective value at the minimum is %f" % f)

    alpha = x
    z = []
    for index in range(svm.LTR.size):
        if LTR[index] == 1:
            z.append(1)
        else:
            z.append(-1)
    w_optimal = 0
    for i in range(svm.DTR.shape[1]):
        w_optimal += alpha[i]*z[i]*svm.DTR[:, i]
    scores = []
    for index in range(svm.DTE.shape[1]):
        scores.append(np.dot(w_optimal.T, svm.DTE[:, index]))
    predictions=[]
    for score in scores:
        if score > threshold:
            predictions.append(1)
        else:
            predictions.append(0)
    primalObjFunc = svm.compute_primal_obj(w_optimal)
    dualObjFunc = f
    print("Primal Obj Func %.6f" % primalObjFunc)
    print("Dual Obj Func %.6f" % dualObjFunc)
    print("Duality gap is %.10f" %((primalObjFunc + dualObjFunc)*10**5))
    print("The error rate is %.3f" % (compute_error_rate(predictions, LTE)*100))


class SVMKernelClass:

    def __init__(self, DTR, LTR, DTE, K, C, kernelFunction, c= None, d=None, gamma=None):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.K = K
        self.C = C
        self.c = c
        self.d = d
        self.gamma = gamma
        self.z = []
        if kernelFunction == 'p':
            self.kernelFunction = polyKernelFunction
        elif kernelFunction == 'r':
            self.kernelFunction = radialBassKernelFunction
        for index in range(LTR.size):
            if LTR[index] == 1:
                self.z.append(1)
            else:
                self.z.append(-1)
        self.H = numpy.zeros((self.DTR.shape[1], self.DTR.shape[1]))
        for i in range(self.DTR.shape[1]):
            for j in range(self.DTR.shape[1]):
                if kernelFunction == 'p':
                    self.H[i, j] = self.z[i] * self.z[j] * self.kernelFunction(DTR[:, i], DTR[:, j], self.c, self.d, self.K)
                elif kernelFunction == 'r':
                    self.H[i, j] = self.z[i] * self.z[j] * self.kernelFunction(DTR[:, i], DTR[:, j], self.gamma, self.K)



    def svm_dual_obj(self, alpha):
        ones = np.ones(self.DTR.shape[1])
        dualObjectiveFunction = 1/2 * np.dot(np.dot(alpha.T, self.H), alpha) - np.dot(alpha.T, ones)
        gradient = np.dot(self.H, alpha) - ones
        gradient = np.reshape(gradient, (alpha.size,))
        return dualObjectiveFunction, gradient


def support_vector_machine_kernel(DTR, LTR, DTE, LTE, K, C, kernelFunction, c=None, d=None, gamma=None, threshold=0):
    svm = SVMKernelClass(DTR, LTR, DTE, K, C, kernelFunction, c, d, gamma)
    bounds = []
    for i in range(svm.DTR.shape[1]):
        bounds.append((0, svm.C))
    x, f, d = scipy.optimize.fmin_l_bfgs_b(svm.svm_dual_obj, x0= numpy.zeros(svm.DTR.shape[1]), fprime=None,
                                           bounds=bounds, factr=1.0)
    alpha = x
    scores = []
    for i in range(svm.DTE.shape[1]):
        tmp=0
        for j in range(svm.DTR.shape[1]):
            if alpha[j] > 0:
                if svm.kernelFunction == polyKernelFunction:
                    tmp = tmp + alpha[j] * svm.z[j] * svm.kernelFunction(DTR[:, j], DTE[:, i], svm.c, svm.d, svm.K)
                elif svm.kernelFunction == radialBassKernelFunction:
                    tmp=tmp+ alpha[j]*svm.z[j]* svm.kernelFunction(DTR[:,j], DTE[:, i], svm.gamma, svm.K)

        scores.append(tmp)

    predictions=[]
    for score in scores:
        if score > threshold:
            predictions.append(1)
        else:
            predictions.append(0)

    dualObjFunc = -f

    print("Dual Obj Func %.6f" % dualObjFunc)
    print("The error rate is %.3f" % (compute_error_rate(predictions, LTE)*100))


def polyKernelFunction(x1, x2, c, d, K):
    return  np.power((np.dot(x1.T, x2) + c), d) + K*K

def radialBassKernelFunction(x1, x2, gamma, K):
    return np.exp((-gamma)*np.power(np.linalg.norm(x1-x2), 2)) + K*K
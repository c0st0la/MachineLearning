import scipy
import matplotlib.pyplot as plt
from itertools import combinations
import numpy
import sklearn.datasets

FILEPATH = 'Solution/iris.csv'
pairs = list()


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


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")


def filter_dataset_by_labels(D, L):
    """

    :param D: It is the dataset to filter
    :param labels: It is the array containing the labels of the dataset
    :return: the dataset D filtered by the labels provided
    """

    return D[:, L == 0], D[:, L == 1]


def compute_binary_index_array(L):
    indexesTrue = list()
    indexesFalse = list()

    for position, item in enumerate((L == 0).tolist()):
        if item:
            indexesTrue.append(position)
        else:
            indexesFalse.append(position)
    return numpy.array(indexesTrue, dtype=numpy.int), numpy.array(indexesFalse, dtype=numpy.int)


def plot_scatter_attributes_X_label(D, filepath, title=""):
    dimension = D.shape[0]
    # color = iter(cm.rainbow(numpy.linspace(0, 1, dimension*dimension)))

    for i in range(dimension):
        for j in range(i + 1, dimension):
            if (i, j) not in pairs:
                pairs.append((i, j))
                # c = next(color)
                plt.scatter(D[i, :], D[j, :], s=4, label=f"Dimension {i + 1}-{j + 1}")
                plt.title(title)
                plt.legend()
                plt.xlim([int(D.min() - 1), int(D.max() + 1)])
                plt.ylim([int(D.min() - 1), int(D.max() + 1)])
                plt.savefig(f"{filepath}{title} Dimension {i + 1}-{j + 1}")
                plt.clf()
    pairs.clear()


def plot_hist_attributes_X_label_binary(DTrue, DFalse, filepath, title=""):
    for i in range(DTrue.shape[0]):
        plt.figure()
        plt.hist(DTrue[i, :], bins=10, ec='black', density=True, alpha=0.3, label='DTrue')
        plt.hist(DFalse[i, :], bins=10, ec='black', density=True, alpha=0.3, label='DFalse')
        plt.legend()
        plt.ylim([0, 0.5])
        plt.xlabel(f"Values dimension {i}")
        plt.savefig(f"{filepath}{title} Dimension {i}")
        plt.clf()


def plot_scatter_attributes_X_label_binary(DTrue, DFalse, filepath, title=""):
    for i in range(DTrue.shape[0]):
        plt.figure()
        plt.scatter(DTrue[i, :], DFalse[i, :], label='DTrue')
        plt.legend()
        plt.xlabel(f"Values dimension {i}")
        plt.savefig(f"{filepath}{title} Dimension {i}")
        plt.clf()


def plot_scatter_attributes_X_label_True_False(DTrue, DFalse, filepath, title=""):
    dimension = DTrue.shape[0]
    # color = iter(cm.rainbow(numpy.linspace(0, 1, dimension*dimension)))

    for i in range(dimension):
        for j in range(i + 1, dimension):
            if (i, j) not in pairs:
                pairs.append((i, j))
                # c = next(color)
                plt.scatter(DTrue[i, :], DTrue[j, :], s=4, label=f"True Dimension {i + 1}-{j + 1}")
                plt.scatter(DFalse[i, :], DFalse[j, :], s=4, label=f"False Dimension {i + 1}-{j + 1}")
                plt.title(title)
                plt.legend()
                plt.xlim([min(int(DTrue.min() - 1), int(DFalse.min() - 1)),
                          max(int(DTrue.max() + 1), int(DFalse.max() + 1))])
                plt.ylim([min(int(DTrue.min() - 1), int(DFalse.min() - 1)),
                          max(int(DTrue.max() + 1), int(DFalse.max() + 1))])
                plt.savefig(f"{filepath}{title} Dimension {i + 1}-{j + 1}")
                plt.clf()
    pairs.clear()


def plot_hist_attributes_X_label_True_False(DTrue, DFalse, filepath, title=""):
    dimension = DTrue.shape[0]
    # color = iter(cm.rainbow(numpy.linspace(0, 1, dimension*dimension)))

    for i in range(dimension):
        for j in range(i + 1, dimension):
            if (i, j) not in pairs:
                pairs.append((i, j))
                # c = next(color)
                plt.hist(DTrue[i, :], bins=10, ec='black', density=True, alpha=0.3, label='DTrue')
                plt.hist(DFalse[j, :], bins=10, ec='black', density=True, alpha=0.3, label='DFalse')
                plt.title(title)
                plt.legend()
                plt.xlim([min(int(DTrue.min() - 1), int(DFalse.min() - 1)),
                          max(int(DTrue.max() + 1), int(DFalse.max() + 1))])
                plt.ylim([0, 0.5])
                plt.savefig(f"{filepath}{title} Dimension {i + 1}-{j + 1}")
                plt.clf()
    pairs.clear()


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
    C, CNormalized = compute_covariance(DC)
    # These are two different but equivalent ways to compute the subspace given...
    # ...the covariance matrix C. You call one or the other depending on the...
    # ...features of the matrix C (see the description inside the function implementation)
    P = compute_subspace(C, sub_dimension)
    P2 = compute_subspace_svd(C, sub_dimension)
    # numpy.dot is a matrix multiplication. I am projecting the dataset on the subspace
    DP = numpy.dot(P.T, D)
    return DP, P


def compute_covariance(D):
    """
    :param D: dataset
    :return: the covariance and the normalized Covariance of the dataset given as input
    """
    covariance = numpy.dot(D, D.T) / float(D.shape[1])
    normalizedCovariance = numpy.identity(covariance.shape[1])
    for i in range(normalizedCovariance.shape[0]):
        for j in range(normalizedCovariance.shape[1]):
            if i == j:
                normalizedCovariance[i, j] = 1
            else:
                normalizedCovariance[i, j] = covariance[i, j] / (
                        numpy.sqrt(covariance[i, i]) * numpy.sqrt(covariance[j, j]))

    return covariance, normalizedCovariance


def compute_empirical_covariance(D):
    """
     It performs the covariance of the matrix D after it has been centered on its mean
     :param D: dataset
     :return: the empirical covariance of the dataset given as input
     """
    D = center_data(D)
    covariance = numpy.dot(D, D.T) / float(D.shape[1])
    normalizedCovariance = numpy.identity(covariance.shape[1])
    for i in range(normalizedCovariance.shape[0]):
        for j in range(normalizedCovariance.shape[1]):
            if i == j:
                normalizedCovariance[i, j] = 1
            else:
                normalizedCovariance[i, j] = covariance[i, j] / (
                        numpy.sqrt(covariance[i, i]) * numpy.sqrt(covariance[j, j]))

    return covariance, normalizedCovariance


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


def compute_LDA_generalized_eigenvalue(D, L, directions, labels):
    """

    :param D: dataset
    :param class_mapping: It is a dictionary where each class is mapped to a number
    :param L: labels
    :param directions: the number of directions of reduction
    :return: the reduced database
    """

    Covariance_between = compute_between_covariance(D, L, labels)
    Covariance_within = compute_within_covariance(D, L, labels)
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
    return DP, W


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


def compute_between_covariance(D, L, labels):
    mu = to_column(D.mean(1))

    Covariance_between = 0
    tot_samples = 0
    for label in labels:
        D_class = D[:, L == label]
        mu_class = to_column(D_class.mean(1))
        num_samples = D_class.shape[1]
        tot_samples += num_samples
        Covariance_between = Covariance_between + num_samples * numpy.dot((mu_class - mu), (mu_class - mu).T)
    Covariance_between = Covariance_between / float(tot_samples)
    return Covariance_between


def compute_LDA_generalized_eigenvalue_by_joint_diagonalization(D, labels, L, directions):
    """
    It solves the generalized eigenvalue by joint diagonalization of the betweeen and within covariance matrices
    :param D: Dataset
    :param labels
    :param L: labels
    :param directions: the number of directions of reduction
    :return: the reduced database
    """
    Covariance_between = compute_between_covariance(D, labels, L)
    Covariance_within = compute_within_covariance(D, labels, L)
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
    arrayLogLikelihood = log_probab_distr_func_GAU_matrix(D, muML, CML)
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
    C, normalizedCaovariance = compute_covariance(DC)
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


def compute_MVG_llrs(DTR, LTR, DTE, labels):
    log_MVG_class_conditional_probabilities = compute_MVG_log_likelihood_as_score_matrix(DTR, LTR, DTE,
                                                                                         labels)
    llrs = numpy.zeros(log_MVG_class_conditional_probabilities.shape[1])
    for i in range(log_MVG_class_conditional_probabilities.shape[1]):
        llrs[i] = log_MVG_class_conditional_probabilities[1, i] - log_MVG_class_conditional_probabilities[0, i]

    return llrs


def compute_NB_llrs(DTR, LTR, DTE, labels):
    log_NB_class_conditional_probabilities = compute_NB_log_likelihood_as_score_matrix(DTR, LTR, DTE,
                                                                                       labels)
    llrs = numpy.zeros(log_NB_class_conditional_probabilities.shape[1])
    for i in range(log_NB_class_conditional_probabilities.shape[1]):
        llrs[i] = log_NB_class_conditional_probabilities[1, i] - log_NB_class_conditional_probabilities[0, i]

    return llrs


def compute_TC_llrs(DTR, LTR, DTE, labels):
    log_TC_class_conditional_probabilities = compute_TC_log_likelihood_as_score_matrix(DTR, LTR, DTE,
                                                                                       labels)
    llrs = numpy.zeros(log_TC_class_conditional_probabilities.shape[1])
    for i in range(log_TC_class_conditional_probabilities.shape[1]):
        llrs[i] = log_TC_class_conditional_probabilities[1, i] - log_TC_class_conditional_probabilities[0, i]

    return llrs


def compute_TNB_llrs(DTR, LTR, DTE, labels):
    log_TNB_class_conditional_probabilities = compute_TNB_log_likelihood_as_score_matrix(DTR, LTR, DTE,
                                                                                         labels)
    llrs = numpy.zeros(log_TNB_class_conditional_probabilities.shape[1])
    for i in range(log_TNB_class_conditional_probabilities.shape[1]):
        llrs[i] = log_TNB_class_conditional_probabilities[1, i] - log_TNB_class_conditional_probabilities[0, i]

    return llrs


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
    C, CNormalized = compute_covariance(DC)
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


def K_fold_cross_validation_accuracy(D, L, classifier, k, class_prior_probability, labels):
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
            log_MVG_posterior_probability = compute_log_posterior_probability(log_MVG_class_conditional_probabilities,
                                                                              class_prior_probability)
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


def K_fold_cross_validation_DCF(D, L, classifier, k, classPriorProbabilities, costs, labels, lambdaValues=[], K=1):
    num_samples = int(D.shape[1] / k)
    totDCF = 0
    DCFsNormalized1 = []
    thresholds = [i for i in numpy.arange(-30, 30, 0.1)]
    perm = numpy.random.permutation(D.shape[1])
    D = D[:, perm]
    L = L[perm]
    if classifier == "MVG":
        for i in range(k):
            (DTR, LTR), (DTE, LTE) = K_fold_generate_Training_and_Testing_samples(D, L, i, k, num_samples)
            llr_MVG = compute_MVG_llrs(DTR, LTR, DTE, labels)
            for threshold in thresholds:
                optimalBayesDecisionPredictions = compute_optimal_bayes_decision_given_threshold(llr_MVG, threshold)
                confusionMatrix = compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
                DCFsNormalized1.append(
                    compute_normalized_detection_cost_function(confusionMatrix, classPriorProbabilities, costs))
            totDCF += min(DCFsNormalized1)

    elif classifier == "NB":
        for i in range(k):
            (DTR, LTR), (DTE, LTE) = K_fold_generate_Training_and_Testing_samples(D, L, i, k, num_samples)
            llr_NB = compute_NB_llrs(DTR, LTR, DTE, labels)
            for threshold in thresholds:
                optimalBayesDecisionPredictions = compute_optimal_bayes_decision_given_threshold(llr_NB, threshold)
                confusionMatrix = compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
                DCFsNormalized1.append(
                    compute_normalized_detection_cost_function(confusionMatrix, classPriorProbabilities, costs))
            totDCF += min(DCFsNormalized1)

    elif classifier == "TC":
        for i in range(k):
            (DTR, LTR), (DTE, LTE) = K_fold_generate_Training_and_Testing_samples(D, L, i, k, num_samples)
            llr_TC = compute_TC_llrs(DTR, LTR, DTE, labels)
            for threshold in thresholds:
                optimalBayesDecisionPredictions = compute_optimal_bayes_decision_given_threshold(llr_TC, threshold)
                confusionMatrix = compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
                DCFsNormalized1.append(
                    compute_normalized_detection_cost_function(confusionMatrix, classPriorProbabilities, costs))
            totDCF += min(DCFsNormalized1)

    elif classifier == "TNB":
        for i in range(k):
            (DTR, LTR), (DTE, LTE) = K_fold_generate_Training_and_Testing_samples(D, L, i, k, num_samples)
            llr_TNB = compute_TNB_llrs(DTR, LTR, DTE, labels)
            for threshold in thresholds:
                optimalBayesDecisionPredictions = compute_optimal_bayes_decision_given_threshold(llr_TNB, threshold)
                confusionMatrix = compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
                DCFsNormalized1.append(
                    compute_normalized_detection_cost_function(confusionMatrix, classPriorProbabilities, costs))
            totDCF += min(DCFsNormalized1)

    elif classifier == "LR":
        if len(lambdaValues) == 0:
            print("Insert a list of lambda values!")
            exit(-1)
        x = dict()
        for i in range(k):
            (DTR, LTR), (DTE, LTE) = K_fold_generate_Training_and_Testing_samples(D, L, i, k, num_samples)
            for lambd in lambdaValues:
                llr_LR = compute_logistic_regression_binary_llr(DTR, LTR, DTE, lambd, classPriorProbabilities)
                for threshold in thresholds:
                    optimalBayesDecisionPredictions = compute_optimal_bayes_decision_given_threshold(llr_LR, threshold)
                    confusionMatrix = compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
                    DCFsNormalized1.append(
                        compute_normalized_detection_cost_function(confusionMatrix, classPriorProbabilities, costs))
                if lambd not in list(x.keys()):
                    x[lambd] = min(DCFsNormalized1)
                else:
                    x[lambd] = x[lambd] + min(DCFsNormalized1)
        for key in list(x.keys()):
            x[key] = x[key] / k
        return x

    elif classifier == "QLR":
        if len(lambdaValues) == 0:
            print("Insert a list of lambda values!")
            exit(-1)
        x = dict()
        for i in range(k):
            (DTR, LTR), (DTE, LTE) = K_fold_generate_Training_and_Testing_samples(D, L, i, k, num_samples)
            DTR = quadratic_expansion(DTR)
            DTE = quadratic_expansion(DTE)
            for lambd in lambdaValues:
                llr_QLR = compute_logistic_regression_binary_quadratic_llr(DTR, LTR, DTE, lambd,
                                                                           classPriorProbabilities)
                for threshold in thresholds:
                    optimalBayesDecisionPredictions = compute_optimal_bayes_decision_given_threshold(llr_QLR, threshold)
                    confusionMatrix = compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
                    DCFsNormalized1.append(
                        compute_normalized_detection_cost_function(confusionMatrix, classPriorProbabilities, costs))
                if lambd not in list(x.keys()):
                    x[lambd] = min(DCFsNormalized1)
                else:
                    x[lambd] = x[lambd] + min(DCFsNormalized1)
        for key in list(x.keys()):
            x[key] = x[key] / k
        return x
    elif classifier == "SVM":
        CList = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5]
        x = dict()
        for i in range(k):
            (DTR, LTR), (DTE, LTE) = K_fold_generate_Training_and_Testing_samples(D, L, i, k, num_samples)
            for C in CList:
                llr_SVM = compute_support_vector_machine_llr(DTR, LTR, DTE, LTE, K, C, classPriorProbabilities)
                for threshold in thresholds:
                    optimalBayesDecisionPredictions = compute_optimal_bayes_decision_given_threshold(llr_SVM, threshold)
                    confusionMatrix = compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
                    DCFsNormalized1.append(
                        compute_normalized_detection_cost_function(confusionMatrix, classPriorProbabilities, costs))
                if C not in list(x.keys()):
                    x[C] = min(DCFsNormalized1)
                else:
                    x[C] = x[C] + min(DCFsNormalized1)
        for key in list(x.keys()):
            x[key] = x[key] / k
        return x
    elif classifier == "KSVM":
        CList = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5]
        gammaValues = [10 ** -3, 10 ** -2, 10 ** -1]
        x = dict()
        for i in range(k):
            (DTR, LTR), (DTE, LTE) = K_fold_generate_Training_and_Testing_samples(D, L, i, k, num_samples)
            for gamma in gammaValues:
                for C in CList:
                    llr_KSVM = compute_support_vector_machine_kernel_llr(DTR, LTR, DTE, LTE, K, C,
                                                                         'r', classPriorProbabilities, c=None, d=None,
                                                                         gamma=gamma)
                    for threshold in thresholds:
                        optimalBayesDecisionPredictions = compute_optimal_bayes_decision_given_threshold(llr_KSVM,
                                                                                                         threshold)
                        confusionMatrix = compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
                        DCFsNormalized1.append(
                            compute_normalized_detection_cost_function(confusionMatrix, classPriorProbabilities, costs))
                    if (C, gamma) not in list(x.keys()):
                        x[(C, gamma)] = min(DCFsNormalized1)
                    else:
                        x[(C, gamma)] = x[(C, gamma)] + min(DCFsNormalized1)
        for key in list(x.keys()):
            x[key] = x[key] / k
        return x
    else:
        print("The given classifier %s is not recognized!" % classifier)
        exit(-1)

    return totDCF / k


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


# Project

def read_file(filename):
    file = open(filename, "r")
    D_list = list()
    D2_list = list()

    sample_mapping = list()

    for line in file:
        attr1 = line.rstrip().split(',')[0:5]
        attr2 = line.rstrip().split(',')[5:10]
        attr = line.rstrip().split(',')[0:10]
        D_list.append(to_column(numpy.array([float(i) for i in attr1])))
        D2_list.append(to_column(numpy.array([float(i) for i in attr])))

        sample_mapping.append(int(line.rstrip().split(',')[10]))
        D_list.append(to_column(numpy.array([float(i) for i in attr2])))
    file.close()
    sample_mapping = numpy.array(sample_mapping, dtype=numpy.int32)
    return numpy.hstack(D_list), sample_mapping, numpy.hstack(D2_list)


def merge_dataset(D):
    for index in range(0, D.shape[1], 2):
        matrix = D[:, index:index + 2]
        tmp = to_column(numpy.hstack(matrix))
        if index != 0:
            D_new = numpy.concatenate([D_new, tmp], axis=1)
        else:
            D_new = tmp

    return D_new


class logRegClass:

    def __init__(self, DTR, LTR, lambd, class_prior_probability, feature_space_dimension=None, num_classes=None):
        self.DTR = DTR
        self.LTR = LTR
        self.class_prior_probability = class_prior_probability
        # This is lambda (the hyper parameter)
        self.lambd = lambd
        self.feature_space_dimension = feature_space_dimension
        self.num_classes = num_classes

    def log_reg_obj_bin(self, v):
        w = v[0:-1]
        b = v[-1]
        regularization_term = (self.lambd / 2) * (numpy.power(numpy.linalg.norm(w), 2))
        objective_function_true = 0
        objective_function_false = 0
        DTR_false, DTR_true = filter_dataset_by_labels(self.DTR, self.LTR)
        for i in range(DTR_true.shape[1]):
            sample = DTR_true[:, i]
            objective_function_true = objective_function_true + \
                                      numpy.logaddexp(0, -1 * (numpy.dot(w.T, sample) + b))
        objective_function_true = (self.class_prior_probability[1] / DTR_true.shape[1]) * objective_function_true
        for i in range(DTR_false.shape[1]):
            sample = DTR_false[:, i]
            objective_function_false = objective_function_false + \
                                       numpy.logaddexp(0, -(-1) * (numpy.dot(w.T, sample) + b))
        objective_function_false = (self.class_prior_probability[0] / DTR_false.shape[1]) * objective_function_false
        objective_function = regularization_term + objective_function_false + objective_function_true
        return objective_function


def logistic_regression_binary(DTR, LTR, DTE, lambd, class_prior_probability, threshold):
    logReg = logRegClass(DTR, LTR, lambd, class_prior_probability)

    x, f, d = scipy.optimize.fmin_l_bfgs_b(logReg.log_reg_obj_bin, x0=numpy.zeros(DTR.shape[0] + 1), approx_grad=True,
                                           factr=100)
    # print("The objective value at the minimum is %f" % f)
    w = x[0:-1]  ##phi(x) da applicare su DTR +DTE
    b = x[-1]
    posterior_log_likelihood_ratio = (numpy.dot(w.T, DTE) + b)
    predictions = numpy.zeros(posterior_log_likelihood_ratio.size)
    for index in range(posterior_log_likelihood_ratio.size):
        if posterior_log_likelihood_ratio[index] >= threshold:
            predictions[index] = 1
    return predictions


def compute_logistic_regression_binary_llr(DTR, LTR, DTE, lambd, class_prior_probability):
    logReg = logRegClass(DTR, LTR, lambd, class_prior_probability)

    x, f, d = scipy.optimize.fmin_l_bfgs_b(logReg.log_reg_obj_bin, x0=numpy.zeros(DTR.shape[0] + 1), approx_grad=True,
                                           factr=100)
    # print("The objective value at the minimum is %f" % f)
    w = x[0:-1]  ##phi(x) da applicare su DTR +DTE
    b = x[-1]
    DTRFalse, DTRTrue = filter_dataset_by_labels(DTR, LTR)
    llr = (numpy.dot(w.T, DTE) + b) - numpy.log(DTRTrue.shape[1] / DTRFalse.shape[1])
    return llr


def compute_logistic_regression_binary_quadratic_llr(DTR, LTR, DTE, lambd, class_prior_probability):
    logReg = logRegClass(DTR, LTR, lambd, class_prior_probability)

    x, f, d = scipy.optimize.fmin_l_bfgs_b(logReg.log_reg_obj_bin, x0=numpy.zeros(DTR.shape[0] + 1), approx_grad=True,
                                           factr=100)
    # print("The objective value at the minimum is %f" % f)
    w = x[0:-1]  ##phi(x) da applicare su DTR +DTE
    b = x[-1]
    DTRFalse, DTRTrue = filter_dataset_by_labels(DTR, LTR)
    llr = (numpy.dot(w.T, DTE) + b) - numpy.log(DTRTrue.shape[1] / DTRFalse.shape[1])
    return llr


def logistic_regression_binary_quadratic_surface(DTR, LTR, DTE, lambd, class_prior_probability, threshold):
    DTR = quadratic_expansion(DTR)
    DTE = quadratic_expansion(DTE)

    logReg = logRegClass(DTR, LTR, lambd, class_prior_probability)

    x, f, d = scipy.optimize.fmin_l_bfgs_b(logReg.log_reg_obj_bin, x0=numpy.zeros(DTR.shape[0] + 1), approx_grad=True,
                                           factr=100)

    w = x[0:-1]
    b = x[-1]
    DTR_false, DTR_true = filter_dataset_by_labels(DTR, LTR)
    posterior_log_likelihood_ratio = (numpy.dot(w.T, DTE) + b) - numpy.log(DTR_true.shape[1] / DTR_false.shape[1])
    predictions = numpy.zeros(posterior_log_likelihood_ratio.size)
    for index in range(posterior_log_likelihood_ratio.size):
        if posterior_log_likelihood_ratio[index] > threshold:
            predictions[index] = 1
    return predictions


def quadratic_expansion(D):
    for i in range(0, D.shape[1]):
        x = D[:, i].reshape((D.shape[0], 1))
        x_2 = numpy.dot(x, x.T)
        x_3 = numpy.hstack(x_2).reshape(-1, 1)
        if i == 0:
            phi = numpy.concatenate((x_3, x), axis=0)
        else:
            tmp = numpy.concatenate((x_3, x), axis=0)
            phi = numpy.concatenate((phi, tmp), axis=1)

    return phi


class SVMClass:

    def __init__(self, DTR, LTR, DTE, K, C):
        self.DTR = numpy.vstack((DTR, K * numpy.ones((1, DTR.shape[1]))))
        self.LTR = LTR
        self.DTE = numpy.vstack((DTE, K * numpy.ones((1, DTE.shape[1]))))
        self.C = C
        self.K = K
        self.z = []
        self.bounds = []
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
        ones = numpy.ones(self.DTR.shape[1])
        dualObjectiveFunction = 1 / 2 * numpy.dot(numpy.dot(alpha.T, self.H), alpha) - numpy.dot(alpha.T, ones)
        gradient = numpy.dot(self.H, alpha) - ones
        gradient = numpy.reshape(gradient, (alpha.size,))
        return dualObjectiveFunction, gradient

    def compute_primal_obj(self, w_optimal):
        primalObjectiveFunction = 0
        regularitazionTerm = 0.5 * (numpy.power(numpy.linalg.norm(w_optimal), 2))
        for i in range(self.DTR.shape[1]):
            primalObjectiveFunction += self.bounds[i][1] * max(0,
                                                               1 - self.z[i] * numpy.dot(w_optimal.T, self.DTR[:, i]))
        primalObjectiveFunction = regularitazionTerm + primalObjectiveFunction
        return primalObjectiveFunction


def compute_support_vector_machine_llr(DTR, LTR, DTE, LTE, K, C, classProbabilities, threshold=0):
    # HERE CLASS BALANCING IS PERFORMED

    svm = SVMClass(DTR, LTR, DTE, K, C)
    svm.bounds = []
    empPriorTrue = filter_dataset_by_labels(DTR, LTR)[1].shape[1] / DTR.shape[1]
    empPriorFalse = filter_dataset_by_labels(DTR, LTR)[0].shape[1] / DTR.shape[1]
    Ct = C * (classProbabilities[1] / empPriorTrue)
    Cf = C * (classProbabilities[0] / empPriorFalse)
    for i in range(DTR.shape[1]):
        if LTR[i] == 0:
            svm.bounds.append((0, Cf))
        else:
            svm.bounds.append((0, Ct))
    x, f, d = scipy.optimize.fmin_l_bfgs_b(svm.svm_dual_obj, x0=numpy.zeros(svm.DTR.shape[1]), fprime=None,
                                           bounds=svm.bounds, factr=1.0)
    alpha = x
    z = []
    for index in range(svm.LTR.size):
        if LTR[index] == 1:
            z.append(1)
        else:
            z.append(-1)
    w_optimal = 0
    for i in range(svm.DTR.shape[1]):
        # HERE I RECOVER THE PRIMAL SOLUTION
        w_optimal += alpha[i] * z[i] * svm.DTR[:, i]
    DTRFalse, DTRTrue = filter_dataset_by_labels(DTR, LTR)
    llr = (numpy.dot(w_optimal.T, svm.DTE)) - numpy.log(DTRTrue.shape[1] / DTRFalse.shape[1])
    return llr


class SVMKernelClass:
    def __init__(self, DTR, LTR, DTE, K, C, kernelFunction, c=None, d=None, gamma=None):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.K = K
        self.C = C
        self.c = c
        self.d = d
        self.gamma = gamma
        self.z = []
        self.bounds = []
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
                    self.H[i, j] = self.z[i] * self.z[j] * self.kernelFunction(DTR[:, i], DTR[:, j], self.c, self.d,
                                                                               self.K)
                elif kernelFunction == 'r':
                    self.H[i, j] = self.z[i] * self.z[j] * self.kernelFunction(DTR[:, i], DTR[:, j], self.gamma, self.K)

    def svm_dual_obj(self, alpha):
        ones = numpy.ones(self.DTR.shape[1])
        dualObjectiveFunction = 1 / 2 * numpy.dot(numpy.dot(alpha.T, self.H), alpha) - numpy.dot(alpha.T, ones)
        gradient = numpy.dot(self.H, alpha) - ones
        gradient = numpy.reshape(gradient, (alpha.size,))
        return dualObjectiveFunction, gradient


def compute_support_vector_machine_kernel_llr(DTR, LTR, DTE, LTE, K, C, kernelFunction, classProbabilities, c=None,
                                              d=None, gamma=None):
    svm = SVMKernelClass(DTR, LTR, DTE, K, C, kernelFunction, c, d, gamma)
    svm.bounds = []
    empPriorTrue = filter_dataset_by_labels(DTR, LTR)[1].shape[1] / DTR.shape[1]
    empPriorFalse = filter_dataset_by_labels(DTR, LTR)[0].shape[1] / DTR.shape[1]
    Ct = C * (classProbabilities[1] / empPriorTrue)
    Cf = C * (classProbabilities[0] / empPriorFalse)
    for i in range(DTR.shape[1]):
        if LTR[i] == 0:
            svm.bounds.append((0, Cf))
        else:
            svm.bounds.append((0, Ct))
    x, f, d = scipy.optimize.fmin_l_bfgs_b(svm.svm_dual_obj, x0=numpy.zeros(svm.DTR.shape[1]), fprime=None,
                                           bounds=svm.bounds, factr=1.0)
    alpha = x
    llr = []
    DTRFalse, DTRTrue = filter_dataset_by_labels(DTR, LTR)
    for i in range(svm.DTE.shape[1]):
        tmp = 0
        for j in range(svm.DTR.shape[1]):
            if alpha[j] > 0:
                if svm.kernelFunction == polyKernelFunction:
                    tmp = tmp + alpha[j] * svm.z[j] * svm.kernelFunction(DTR[:, j], DTE[:, i], svm.c, svm.d, svm.K)
                elif svm.kernelFunction == radialBassKernelFunction:
                    tmp = tmp + alpha[j] * svm.z[j] * svm.kernelFunction(DTR[:, j], DTE[:, i], svm.gamma, svm.K)
        llr.append(tmp - numpy.log(DTRTrue.shape[1] / DTRFalse.shape[1]))
    return llr


def polyKernelFunction(x1, x2, c, d, K):
    return numpy.power((numpy.dot(x1.T, x2) + c), d) + K * K


def radialBassKernelFunction(x1, x2, gamma, K):
    return numpy.exp((-gamma) * numpy.power(numpy.linalg.norm(x1 - x2), 2)) + K * K


def withening_pre_processing(D):
    C, CNormalized = compute_empirical_covariance(D)
    CWhiten = C * numpy.identity(D.shape[0])
    DWhiten = numpy.dot(CWhiten, D)
    return DWhiten


def length_normalization(D):
    for j in range(D.shape[1]):
        D[:, j] = D[:, j] / numpy.linalg.norm(D[:, j])
    return D


def compute_binary_confusion_matrix(predictions, L):
    numLabels = 2
    matrix = numpy.zeros((numLabels, numLabels), dtype=numpy.int32)
    for index, prediction in enumerate(predictions.tolist()):
        matrix[prediction, L[index]] += 1
    return matrix


def compute_optimal_bayes_decision(logLikelihoodRatios, priorsProbability, costs):
    """

    :param logLikelihoodRatio:  ll of class 1 over ll of class 0
    :param priorsProbability: index 0 contains priors of class 0, index 1 contatins priors of class 1
    :param costs: index 0 contains cost of false negative, index 1 contains cost of false postive
    :return:
    """
    predictions = numpy.zeros((logLikelihoodRatios.size), dtype=numpy.int)
    threshold = - numpy.log((priorsProbability[1] * costs[0]) / (priorsProbability[0] * costs[1]))
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
    predictions = numpy.zeros(len(logLikelihoodRatios), dtype=numpy.int32)
    for index, llr in enumerate(logLikelihoodRatios):
        if llr >= threshold:
            # it is predicted as class 1
            predictions[index] = 1
        else:
            # it is predicted as class 0
            predictions[index] = 0
    return predictions


def compute_binary_prediction_rates(confusionMatrix):
    FNR = confusionMatrix[0, 1] / (confusionMatrix[0, 1] + confusionMatrix[1, 1])
    FPR = confusionMatrix[1, 0] / (confusionMatrix[0, 0] + confusionMatrix[1, 0])
    TNR = confusionMatrix[0, 0] / (confusionMatrix[0, 0] + confusionMatrix[1, 0])
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
    return classPriorsProbability[1] * costs[0] * FNR + (1 - classPriorsProbability[1]) * costs[1] * FPR


def compute_normalized_detection_cost_function(confusionMatrix, classPriorsProbability, costs):
    """

    :param confusionMatrix: confusionMatrix
    :param priorsProbability: index 0 contains priors of class 0, index 1 contatins priors of class 1
    :param costs: index 0 contains cost of false negative, index 1 contains cost of false postive
    :return:

    """
    DCF = compute_detection_cost_function(confusionMatrix, classPriorsProbability, costs)
    return DCF / min(classPriorsProbability[1] * costs[0], (1 - classPriorsProbability[1]) * costs[1])


def compute_missclassification_ratios(confusionMatrix):
    misClassification_ratios = numpy.zeros((confusionMatrix.shape[0], confusionMatrix.shape[1]))
    for i in range(confusionMatrix.shape[0]):
        for j in range(confusionMatrix.shape[1]):
            misClassification_ratios[i, j] = confusionMatrix[i, j] / (numpy.sum(confusionMatrix[:, j]))
    return misClassification_ratios


def compute_detection_cost_functio_by_misclassificationRatio(costs, misClassificationRatios, classPriorsProbability):
    DCF = 0
    for j in range(classPriorsProbability.size):
        sum = 0
        for i in range(classPriorsProbability.size):
            sum += costs[i, j] * misClassificationRatios[i, j]
        DCF += classPriorsProbability[j] * sum
    return DCF


def compute_treshold_qlr(DTR, LTR, DTE, LTE, lambd, class_prior_probability, costs):
    DTR = quadratic_expansion(DTR)
    DTE = quadratic_expansion(DTE)

    logReg = logRegClass(DTR, LTR, lambd, class_prior_probability)

    x, f, d = scipy.optimize.fmin_l_bfgs_b(logReg.log_reg_obj_bin, x0=numpy.zeros(DTR.shape[0] + 1), approx_grad=True,
                                           factr=100)
    # print("The objective value at the minimum is %f" % f)
    w = x[0:-1]  ##phi(x) da applicare su DTR +DTE
    b = x[-1]
    DTR_false, DTR_true = filter_dataset_by_labels(DTR, LTR)
    posterior_log_likelihood_ratio = (numpy.dot(w.T, DTE) + b) - numpy.log(DTR_true.shape[1] / DTR_false.shape[1])

    optimalBayesDecisionPredictions = compute_optimal_bayes_decision(posterior_log_likelihood_ratio,
                                                                     class_prior_probability,
                                                                     costs)
    confusionMatrix = compute_binary_confusion_matrix(optimalBayesDecisionPredictions, LTE)
    DCF = compute_detection_cost_function(confusionMatrix, class_prior_probability, costs)
    DCFNormalized = compute_normalized_detection_cost_function(confusionMatrix, class_prior_probability, costs)
    print("The priors probabilities are : ", class_prior_probability, "\n")
    print("The costs are :")
    print("Costs of false negative (label a class to 0 when the real is 1) : ", class_prior_probability[0], "\n")
    print("Costs of false positive (label a class to 1 when the real is 0) : ", class_prior_probability[1], "\n")
    print("Confusion Matrix : \n", confusionMatrix, "\n")
    print("DCF : %.3f" % DCF)
    print("Normalized DCF : %.3f\n" % DCFNormalized)

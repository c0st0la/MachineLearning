import numpy
import matplotlib.pyplot as plt
from itertools import combinations
import scipy

FILEPATH = 'Solution/iris.csv'


def to_column(array):
    return array.reshape((array.size, 1))


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
    :return: the covariance of the dataset given as inumpyut
    """
    covariance = numpy.dot(D, D.T) / float(D.shape[1])
    return covariance


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


def compute_within_covariance(D, class_mapping, samples_mapped_to_class):
    labels = class_mapping.values()
    Covariance_within = 0
    tot_samples = 0
    for label in labels:
        # D_class is the dataset filtered by class
        D_class = D[:, samples_mapped_to_class == label]
        DC_class = center_data(D_class)
        num_samples = D_class.shape[1]
        tot_samples += num_samples
        Covariance_within = Covariance_within + numpy.dot(DC_class, DC_class.T)
    Covariance_within = Covariance_within/float(tot_samples)
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
        Covariance_between = Covariance_between + num_samples * numpy.dot((mu_class-mu), (mu_class-mu).T)
    Covariance_between = Covariance_between/float(tot_samples)
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
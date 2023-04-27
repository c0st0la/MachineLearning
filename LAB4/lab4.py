from functions import *

FILEPATH = 'Solution/iris.csv'

if __name__=="__main__":
    D, L = load_iris_datasets_from_file(FILEPATH)

    XND = numpy.load('Solution/XND.npy')
    mu = numpy.load('Solution/muND.npy')
    C = numpy.load('Solution/CND.npy')
    pdfSol = numpy.load('Solution/llND.npy')
    pdfGau = log_probab_distr_func_GAU_matrix(XND, mu, C)
    print(numpy.abs(pdfSol - pdfGau).max())


    logLikelihood = log_likelihood(XND)
    print(logLikelihood)
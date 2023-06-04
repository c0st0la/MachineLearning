import numpy as np
from functions import *

def compute_class_conditional_log_likelihoods_as_score_matrix(dictTextTrain, text_Evaluation, class2Index, eps = 0.001):
    """
         IT IS USED IN THE DISCRETE DOMAIN
         This function return the score (e.g the class conditional probabilities/log_likehood f(x|C) )
         Each row of the score matrix corresponds to a class, and contains
         the conditional log-likelihoods for all the samples for that class

         -dictTextTrain is the text over which compute the model parameter;
         -text_Evaluation is the text over which compute the score matrix;
         -class2Index is a dictionary mapping each class to an int

    """
    log_class_conditional_probabilities = numpy.zeros((len(class2Index), len(text_Evaluation)))
    dictClassOccurrences, dictClassModelParameter = estimate_model_parameter(dictTextTrain, eps)

    for column_index, tercet in enumerate(text_Evaluation):
        sample_scores = compute_sample_log_likelihood(dictClassModelParameter, tercet, class2Index)
        log_class_conditional_probabilities[:, column_index] = sample_scores

    return log_class_conditional_probabilities


def compute_sample_log_likelihood(dictClassModelParameter, tercet, class2Index):
    """
    This function returns the log_likelihood  for the given text 'text' and for each class label given in 'class2Index'
    The returned object has dimension (len(class2Index), 1)
    """
    scores = numpy.zeros((len(class2Index), 1))
    for label in dictClassModelParameter.keys():
        row_index = class2Index[label]
        words = tercet.split()
        for word in words:
            if word in dictClassModelParameter[label]:
                scores[row_index, :] += dictClassModelParameter[label][word]
    return scores.ravel()


def estimate_model_parameter(dictTextTrain, eps = 0.001):
    """
    IT IS USED IN THE DISCRETE DOMAIN
    This function returns a pair containing first the occurrences and then the log
    of the frequency of a word in each sample. The frequencies are the model parameters
    """
    commonDict = set([])
    for cls in dictTextTrain.keys():
        dictCls = create_dictionary(dictTextTrain[cls])
        commonDict = commonDict.union(dictCls)

    dictClsOccurrences = dict.fromkeys(dictTextTrain, {})
    for cls in dictTextTrain.keys():
        dictClsOccurrences[cls] = compute_occurrence_given_dictionary(commonDict, dictTextTrain[cls], eps)

    dictClsFrequencies = dict.fromkeys(dictTextTrain, {})
    for cls in dictTextTrain.keys():
        dictClsFrequencies[cls] = compute_frequencies((sum(dictClsOccurrences[cls].values())), dictClsOccurrences[cls])

    return dictClsOccurrences, dictClsFrequencies


def compute_occurrence_given_dictionary(dictionary, text, eps):
    """
    This function returns the occurrences of dictionary's words in text
    -eps is a pseudo-count : since we are going to compute later the log, if a word is not present
        (occurrences = 0) we have to face with log(0) = -inf. We want to avoid this situation, that's why
        we add 'eps'
    """
    occurrences = dict.fromkeys(dictionary, eps)
    for tercet in text:
        words = tercet.split()
        for word in words:
            occurrences[word] += 1
    return occurrences


def compute_frequencies(totalWordsXClass, occurrencesXClass):
    """
    This function returns a dictionary containing for each discrete value the ratio of the occurrences of that
    value in a class over the total number of discrete values present in that class
    """
    frequencies = {}
    for key in occurrencesXClass.keys():
        frequencies[key] = numpy.log(occurrencesXClass[key] / totalWordsXClass)
    return frequencies


def create_dictionary(text):

    '''
    Create a set of all words contained in the list of tercets lTercets
    lTercets is a list of tercets (list of strings)
    '''

    sDict = set([])
    for tercet in text:
        words = tercet.split()
        for w in words:
            sDict.add(w)
    return sDict



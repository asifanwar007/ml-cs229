import collections

import numpy as np
import time
import datetime

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    w_list = list(map(str, message.split(' ')))
    word_list = []
    word_dic = {}
    for c in w_list:
        c = c.lower()
        if word_dic.get(c) is None:
            word_list.append(c)
            word_dic[c] = 1
    return word_list
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message. 

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    dic_word = {}
    for sms in messages:
        word_list = get_words(sms)
        for word in word_list:
            if dic_word.get(word) is None:
                dic_word[word] = 1
            else:
                dic_word[word] = dic_word[word]+1

    i = 0
    dic = {}
    for key in dic_word:
        if dic_word.get(key) >= 5:
            dic[key] = i
            i += 1
    return dic
    # *** END CODE HERE ***
def get_words1(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    w_list = list(map(str, message.split(' ')))
    word_list = []
    for c in w_list:
        c = c.lower()
        word_list.append(c)
    return word_list
    # *** END CODE HERE ***

def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each 
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that 
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    # *** START CODE HERE ***
    col = len(word_dictionary)
    row = len(messages)
    matrix = np.zeros((row, col), dtype=int)
    for i, sms in enumerate(messages):
        words = get_words1(sms)
        for word in words:
            # print(word)
            # print(type(word))
            pos = word_dictionary.get(word)
            matrix[i][pos] = matrix[i][pos] + 1
    print(matrix.shape)
    return matrix
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    print("----fit_naive_bayes_model---")
    row, col = matrix.shape
    y_i1 = labels[labels == 1].sum()
    phi_y = y_i1/row
    spam_matrix = matrix[labels==1]

    t_list = []
    for i in range(col):
        t_matrix = spam_matrix[:, i]
        diff = len(t_matrix)-len(t_matrix[t_matrix==0])
        # print(diff)
        t_list.append(diff)
    spam_matrix = np.array(t_list)
    phi_j_y1 = spam_matrix/y_i1

    non_spam_matrix = matrix[labels==0]
    t_list = []
    for i in range(col):
        t_matrix = non_spam_matrix[:, i]
        diff = len(t_matrix)-len(t_matrix[t_matrix==0])
        t_list.append(diff)
    non_spam_matrix = np.array(t_list)
    phi_j_y0 = non_spam_matrix/(row-y_i1)

    # print(spam_matrix[spam_matrix!=0].sum(axis=0).shape)
    # print(count_spam_matrix.shape)
    return phi_y, phi_j_y0, phi_j_y1
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    phi_y, phi_j_y0, phi_j_y1 = model
    ans = []
    for c in matrix:
        phi_j_y11 = phi_j_y1.copy()
        phi_j_y11[c==0] = 0
        a = np.prod(phi_j_y11[phi_j_y11 != 0])
        ans.append(a)
    p1 = np.array(ans)
    p1 = p1*phi_y
    ans = []
    for c in matrix:
        phi_j_y11 = phi_j_y0.copy()
        phi_j_y11[c==0] = 0
        a = np.prod(phi_j_y11[phi_j_y11 != 0])
        ans.append(a)
    p2 = np.array(ans)
    p2 = p2*(1-phi_y)

    pred = p1/(p1+p2)
    pred[pred>0.8] = 1
    pred[pred<=0.8] = 0
    pred.astype(int)
    # print(pred.shape)
    return pred
    # print(matrix.dot(phi_j_y0))
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: The top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider
    
    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    # *** END CODE HERE ***


def main():
    begin_time = datetime.datetime.now()
    train_messages, train_labels = util.load_spam_dataset('../data/ds6_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('../data/ds6_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('../data/ds6_test.tsv')
    
    dictionary = create_dictionary(train_messages)

    util.write_json('./output/p06_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('./output/p06_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('./output/p06_naive_bayes_predictions', naive_bayes_predictions)
    np.savetxt('./output/p06_test_labels', test_labels)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))
    print(datetime.datetime.now() - begin_time)

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('./output/p06_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('./output/p06_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()

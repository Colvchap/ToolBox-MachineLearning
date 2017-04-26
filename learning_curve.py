"""Explore learning curves for classification of handwritten digits
@author Colvin """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def display_digits():
    """Read in the 8x8 pictures of numbers and display 10 of them"""
    digits = load_digits()
    print(digits.DESCR)
    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(5, 2, i+1)
        subplot.matshow(numpy.reshape(digits.data[i], (8, 8)), cmap='gray')

    plt.show()


def train_model():
    """Train a model on pictures of digits.

    Read in 8x8 pictures of numbers and evaluate the accuracy of the model
    when different percentages of the data are used as training data. This function
    plots the average accuracy of the model as a function of the percent of data
    used to train it.
    """
    data = load_digits()
    num_trials = 75
    train_percentages = range(5, 95, 5)
    all_tests = numpy.zeros(len(train_percentages))
    test_accuracies = numpy.zeros(len(train_percentages))

    # train models with training percentages between 5 and 90 (see
    # train_percentages) and evaluate the resultant accuracy for each.
    # You should repeat each training percentage num_trials times to smooth out
    # variability.
    # For consistency with the previous example use
    # model = LogisticRegression(C=10**-10) for your learner
    for test in range(num_trials):
        i = 0
        for percentage in train_percentages:
            train_size = .01*percentage

            X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
                                                                train_size = train_size)
            model = LogisticRegression(C=100)

            model.fit(X_train, y_train)
            # print("Train accuracy %f" %model.score(X_train, y_train))
            # print("Test accuracy %f"%model.score(X_test, y_test))

            test_accuracies[i] = model.score(X_test, y_test)
            i += 1

        all_tests = numpy.vstack((all_tests, test_accuracies))    # extending output matrix

    final_test_accuracies = numpy.mean(all_tests, axis=0)
    print(final_test_accuracies)
    fig = plt.figure()
    print(train_percentages, final_test_accuracies)
    plt.plot(train_percentages, final_test_accuracies)
    plt.xlabel('Percentage of Data Used for Training')
    plt.ylabel('Accuracy on Test Set')
    plt.show()


if __name__ == "__main__":
    # Feel free to comment/uncomment as needed
    #display_digits()
    train_model()

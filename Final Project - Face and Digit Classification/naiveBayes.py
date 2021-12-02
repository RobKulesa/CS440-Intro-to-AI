# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod


class NaiveBayesClassifier(classificationMethod.ClassificationMethod):

    def __init__(self, legal_labels):
        super().__init__(legal_labels)
        self.type = "naivebayes"
        self.k = 1  # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False  # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):

        self.k = k

    def train(self, training_data, training_labels, validation_data, validation_labels):

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([f for datum in training_data for f in datum.keys()]))

        if self.automaticTuning:
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(training_data, training_labels, validation_data, validation_labels, kgrid)

    def trainAndTune(self, training_data, training_labels, validation_data, validation_labels, kgrid):

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def classify(self, test_data):

        guesses = []
        self.posteriors = []  # Log posteriors are stored for later data analysis (autograder).
        for datum in test_data:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):

        log_joint = util.Counter()

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        return log_joint

    def findHighOddsFeatures(self, label1, label2):

        features_odds = []

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        return features_odds

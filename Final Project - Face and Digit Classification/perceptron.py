# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Perceptron implementation
import util
import classificationMethod

PRINT = True


class PerceptronClassifier(classificationMethod.ClassificationMethod):

    def __init__(self, legal_labels, max_iterations):
        super().__init__(legal_labels)
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legal_labels:
            self.weights[label] = util.Counter()  # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, training_data, training_labels, validation_data, validation_labels):

        self.features = training_data[0].keys()  # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        for iteration in range(self.max_iterations):
            print("Starting iteration ", iteration, "...")
            for i in range(len(training_data)):
                "*** YOUR CODE HERE ***"
                util.raiseNotDefined()

    def classify(self, data):

        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses

    def findHighWeightFeatures(self, label):

        features_weights = []

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        return features_weights

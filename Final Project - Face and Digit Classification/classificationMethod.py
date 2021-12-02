# classificationMethod.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# This file contains the abstract class ClassificationMethod
import abc


class ClassificationMethod(abc.ABC):

    def __init__(self, legal_labels):
        self.legalLabels = legal_labels

    def train(self, training_data, training_labels, validation_data, validation_labels):
        pass

    def classify(self, data):
        pass

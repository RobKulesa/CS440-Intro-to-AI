# Parts of dataClassifier.py from  were used.
# For more information, see https://inst.eecs.berkeley.edu//~cs188/sp11/projects/classification/classification.html

import perceptron
import samples
import sys
import util
import time
import numpy as np
import random

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70
ITERATIONS = 3

def FeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def FeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def runClassifier():

  #Get Inputs
  dset = raw_input('Choose dataset (faces, digits): ')
  training_val = int(raw_input('Enter amount of training data to use: '))
  testing_amount = int(raw_input('Enter the amount of values to test: '))

  if(dset == "faces"):
    featureFunction = FeatureExtractorFace
  elif dset == "digits":
    featureFunction = FeatureExtractorDigit

  if dset == "digits" :
    legalLabels = range(10)
  else:
    legalLabels = range(2)

  #Classifier is Perceptron
  classifier = perceptron.PerceptronClassifier(legalLabels, ITERATIONS)
  
  #Load data
  numTraining = training_val
  numTest = testing_amount

  if dset == "faces":
    rawTrainingData, chosenList = samples.loadDataFile("facedata/facedatatrain", numTraining, FACE_DATUM_WIDTH,
                                                       FACE_DATUM_HEIGHT, True)
    trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", chosenList)
    rawValidationData, chosenList = samples.loadDataFile("facedata/facedatatrain", numTest, FACE_DATUM_WIDTH,
                                                         FACE_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", chosenList)
    rawTestData, chosenList = samples.loadDataFile("facedata/facedatatest", numTest, FACE_DATUM_WIDTH,
                                                   FACE_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", chosenList)
  elif dset == "digits":
    rawTrainingData, chosenList = samples.loadDataFile("digitdata/trainingimages", numTraining, DIGIT_DATUM_WIDTH,
                                                       DIGIT_DATUM_HEIGHT, True)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", chosenList)
    rawValidationData, chosenList = samples.loadDataFile("digitdata/validationimages", numTest, DIGIT_DATUM_WIDTH,
                                                         DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", chosenList)
    rawTestData, chosenList = samples.loadDataFile("digitdata/testimages", numTest, DIGIT_DATUM_WIDTH,
                                                   DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", chosenList)

  # Extract features
  print
  "Extracting features..."
  trainingData = list(map(featureFunction, rawTrainingData))
  validationData = list(map(featureFunction, rawValidationData))
  testData = list(map(featureFunction, rawTestData))

  #Conduct training
  print "Training..."
  test_begintime = time.time()
  classifier.train(trainingData, trainingLabels, validationData, validationLabels)
  test_endtime = time.time()
  test_duration = test_endtime - test_begintime
  
  #Conduct testing
  print "Testing..."
  guesses = classifier.classify(testData)
  correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
  print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))

  #Training Duration
  trainingDuration = "Training Duration : " + str(test_duration) + " seconds."
  print trainingDuration

def runClassifierStats(first, second, third):

  #Get Inputs
  dset = first
  training_val = second
  testing_amount = third

  if(dset == "faces"):
    featureFunction = FeatureExtractorFace
  elif dset == "digits":
    featureFunction = FeatureExtractorDigit

  if dset == "digits" :
    legalLabels = range(10)
  else:
    legalLabels = range(2)

  #Classifier is Perceptron
  classifier = perceptron.PerceptronClassifier(legalLabels, ITERATIONS)
  
  #Load data
  numTraining = training_val
  numTest = testing_amount

  if dset == "faces":
    rawTrainingData, chosenList = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT, True)
    trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", chosenList)
    rawValidationData, chosenList = samples.loadDataFile("facedata/facedatatrain", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", chosenList)
    rawTestData, chosenList = samples.loadDataFile("facedata/facedatatest", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", chosenList)
  elif dset == "digits":
    rawTrainingData, chosenList = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT, True)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", chosenList)
    rawValidationData, chosenList = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", chosenList)
    rawTestData, chosenList = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", chosenList)


  #Extract features
  print "Extracting features..."
  trainingData = list(map(featureFunction, rawTrainingData))
  validationData = list(map(featureFunction, rawValidationData))
  testData = list(map(featureFunction, rawTestData))

  #Conduct training
  print "Training..."
  test_begintime = time.time()
  classifier.train(trainingData, trainingLabels, validationData, validationLabels)
  test_endtime = time.time()
  test_duration = test_endtime - test_begintime
  
  #Conduct testing
  print "Testing..."
  guesses = classifier.classify(testData)
  correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
  print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))

  #Training Duration
  trainingDuration = "Training Duration : " + str(test_duration) + " seconds."
  print trainingDuration

  return float(100.0 * correct / len(testLabels))

if __name__ == '__main__':
  runClassifier()

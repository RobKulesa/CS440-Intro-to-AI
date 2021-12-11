# Parts of dataClassifier.py from  were used.
# For more information, see https://inst.eecs.berkeley.edu//~cs188/sp11/projects/classification/classification.html

import mostFrequent
import naiveBayes
import perceptron
import mira
import samples
import sys
import util
import time

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
    rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
    rawValidationData = samples.loadDataFile("facedata/facedatatrain", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTest)
    rawTestData = samples.loadDataFile("facedata/facedatatest", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", numTest)
  elif dset == "digits":
    rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
    rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
    rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)


  #Extract features
  print "Extracting features..."
  trainingData = map(featureFunction, rawTrainingData)
  validationData = map(featureFunction, rawValidationData)
  testData = map(featureFunction, rawTestData)

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

if __name__ == '__main__':
  runClassifier()
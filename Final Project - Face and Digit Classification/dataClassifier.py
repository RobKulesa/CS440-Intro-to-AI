# dataClassifier.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# This file contains feature extraction methods and harness 
# code for data classification

import perceptron
import naiveBayes
import samples
import sys
import util

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 70


def basicFeatureExtractorDigit(datum):
    """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features


def basicFeatureExtractorFace(datum):
    """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features


def enhancedFeatureExtractorDigit(datum):
    """
  Your feature extraction playground.
  
  You should return a util.Counter() of features
  for this datum (datum is of type samples.Datum).
  
  ## DESCRIBE YOUR ENHANCED FEATURES HERE...
  
  ##
  """
    features = basicFeatureExtractorDigit(datum)

    "*** YOUR CODE HERE ***"

    return features


def enhancedFeatureExtractorFace(datum):
    """
  Your feature extraction playground for faces.
  It is your choice to modify this.
  """
    features = basicFeatureExtractorFace(datum)
    return features


def analysis(classifier, guesses, test_labels, test_data, raw_test_data, print_image):
    """
  This function is called after learning.
  Include any code that you want here to help you analyze your results.
  
  Use the printImage(<list of pixels>) function to visualize features.
  
  An example of use has been given to you.
  
  - classifier is the trained classifier
  - guesses is the list of labels predicted by your classifier on the test set
  - testLabels is the list of true labels
  - testData is the list of training datapoints (as util.Counter of features)
  - rawTestData is the list of training datapoints (as samples.Datum)
  - printImage is a method to visualize the features 
  (see its use in the odds ratio part in runClassifier method)
  
  This code won't be evaluated. It is for your own optional use
  (and you can modify the signature if you want).
  """

    # Put any code here...
    # Example of use:
    for i in range(len(guesses)):
        prediction = guesses[i]
        truth = test_labels[i]
        if prediction != truth:
            print("===================================")
            print("Mistake on example %d" % i)
            print("Predicted %d; truth is %d" % (prediction, truth))
            print("Image: ")
            print(raw_test_data[i])
            break


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def printImage(self, pixels):
        """
      Prints a Datum object that contains all pixels in the 
      provided list of pixels.  This will serve as a helper function
      to the analysis function you write.
      
      Pixels should take the form 
      [(2,2), (2, 3), ...] 
      where each tuple represents a pixel.
      """
        image = samples.Datum(None, self.width, self.height)
        for pix in pixels:
            try:
                # This is so that new features that you could define which
                # which are not of the form of (x,y) will not break
                # this image printer...
                x, y = pix
                image.pixels[x][y] = 2
            except:
                print("new features:", pix)
                continue
        print(image)


def default(str):
    return str + ' [Default: %default]'


def readCommand(argv):
    # "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'),
                      choices=['nb', 'naiveBayes', 'perceptron', 'knn'],
                      default='nb')
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='digits')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
    parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False,
                      action="store_true")
    parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False,
                      action="store_true")
    parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
    parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
    parser.add_option('-w', '--weights', help=default('Whether to print(weights'), default=False, action="store_true")
    parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"),
                      type="float", default=2.0)
    parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False,
                      action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print("Doing classification")
    print("--------------------")
    print("data:\t\t" + options.data)
    print("classifier:\t\t" + options.classifier)
    print("using enhanced features?:\t" + str(options.features))
    print("training set size:\t" + str(options.training))
    if options.data == "digits":
        print_image = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
        if options.features:
            feature_function = enhancedFeatureExtractorDigit
        else:
            feature_function = basicFeatureExtractorDigit
    elif options.data == "faces":
        print_image = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
        if options.features:
            feature_function = enhancedFeatureExtractorFace
        else:
            feature_function = basicFeatureExtractorFace
    else:
        print("Unknown dataset", options.data)
        print(USAGE_STRING)
        sys.exit(2)

    if options.data == "digits":
        legal_labels = range(10)
    else:
        legal_labels = range(2)

    if options.training <= 0:
        print("Training set size should be a positive integer (you provided: %d)" % options.training)
        print(USAGE_STRING)
        sys.exit(2)

    if options.smoothing <= 0:
        print("Please provide a positive number for smoothing (you provided: %f)" % options.smoothing)
        print(USAGE_STRING)
        sys.exit(2)

    if options.odds:
        if options.label1 not in legal_labels or options.label2 not in legal_labels:
            print("Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2))
            print(USAGE_STRING)
            sys.exit(2)

    if options.classifier == "naiveBayes" or options.classifier == "nb":
        classifier = naiveBayes.NaiveBayesClassifier(legal_labels)
        classifier.setSmoothing(options.smoothing)
        if options.autotune:
            print("using automatic tuning for naivebayes")
            classifier.automaticTuning = True
        else:
            print("using smoothing parameter k=%f for naivebayes" % options.smoothing)
    elif options.classifier == "perceptron":
        classifier = perceptron.PerceptronClassifier(legal_labels, options.iterations)
    else:
        print("Unknown classifier:", options.classifier)
        print(USAGE_STRING)

        sys.exit(2)

    args['classifier'] = classifier
    args['feature_function'] = feature_function
    args['print_image'] = print_image

    return args, options


USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """


# Main harness code

def runClassifier(args, options):
    feature_function = args['feature_function']
    classifier = args['classifier']
    print_image = args['print_image']

    # Load data
    num_training = options.training
    num_test = options.test

    if options.data == "faces":
        raw_training_data = samples.loadDataFile("data/facedata/facedatatrain", num_training, FACE_DATUM_WIDTH,
                                                 FACE_DATUM_HEIGHT)
        training_labels = samples.loadLabelsFile("data/facedata/facedatatrainlabels", num_training)
        raw_validation_data = samples.loadDataFile("data/facedata/facedatatrain", num_test, FACE_DATUM_WIDTH,
                                                   FACE_DATUM_HEIGHT)
        validation_labels = samples.loadLabelsFile("data/facedata/facedatatrainlabels", num_test)
        raw_test_data = samples.loadDataFile("data/facedata/facedatatest", num_test, FACE_DATUM_WIDTH,
                                             FACE_DATUM_HEIGHT)
        test_labels = samples.loadLabelsFile("data/facedata/facedatatestlabels", num_test)
    else:
        raw_training_data = samples.loadDataFile("data/digitdata/trainingimages", num_training, DIGIT_DATUM_WIDTH,
                                                 DIGIT_DATUM_HEIGHT)
        training_labels = samples.loadLabelsFile("data/digitdata/traininglabels", num_training)
        raw_validation_data = samples.loadDataFile("data/digitdata/validationimages", num_test, DIGIT_DATUM_WIDTH,
                                                   DIGIT_DATUM_HEIGHT)
        validation_labels = samples.loadLabelsFile("data/digitdata/validationlabels", num_test)
        raw_test_data = samples.loadDataFile("data/digitdata/testimages", num_test, DIGIT_DATUM_WIDTH,
                                             DIGIT_DATUM_HEIGHT)
        test_labels = samples.loadLabelsFile("data/digitdata/testlabels", num_test)

    # Extract features
    print("Extracting features...")
    training_data = map(feature_function, raw_training_data)
    validation_data = map(feature_function, raw_validation_data)
    test_data = map(feature_function, raw_test_data)

    # Conduct training and testing
    print("Training...")
    classifier.train(training_data, training_labels, validation_data, validation_labels)
    # print("Validating...")
    # guesses = classifier.classify(validation_data)
    # correct = [guesses[i] == validation_labels[i] for i in range(len(validation_labels))].count(True)
    # print(str(correct),
    #       ("correct out of " + str(len(validation_labels)) + " (%.1f%%).") % (100.0 * correct / len(validation_labels)))
    print("Testing...")
    guesses = classifier.classify(test_data)
    correct = [guesses[i] == test_labels[i] for i in range(len(test_labels))].count(True)
    print(str(correct),
          ("correct out of " + str(len(test_labels)) + " (%.1f%%).") % (100.0 * correct / len(test_labels)))
    analysis(classifier, guesses, test_labels, test_data, raw_test_data, print_image)

    # do odds ratio computation if specified at command line
    if options.odds and (options.classifier == "naiveBayes" or options.classifier == "nb"):
        label1, label2 = options.label1, options.label2
        features_odds = classifier.findHighOddsFeatures(label1, label2)
        if options.classifier == "naiveBayes" or options.classifier == "nb":
            string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
        else:
            string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)

        print(string3)
        print_image(features_odds)

    if options.weights and options.classifier == "perceptron":
        for label in classifier.legalLabels:
            features_weights = classifier.findHighWeightFeatures(label)
            print(("=== Features with high weight for label %d ===" % label))
            print_image(features_weights)


if __name__ == '__main__':
    # Read input
    args, options = readCommand(sys.argv[1:])
    # Run classifier
    runClassifier(args, options)

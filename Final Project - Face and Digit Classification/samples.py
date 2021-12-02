# samples.py
# ----------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util

# Constants
DATUM_WIDTH = 0  # in pixels
DATUM_HEIGHT = 0  # in pixels


# Module Classes

class Datum:

    def __init__(self, data, width, height):

        DATUM_HEIGHT = height
        DATUM_WIDTH = width
        self.height = DATUM_HEIGHT
        self.width = DATUM_WIDTH
        if data is None:
            data = [[' ' for i in range(DATUM_WIDTH)] for j in range(DATUM_HEIGHT)]
        self.pixels = util.arrayInvert(convertToInteger(data))

    def getPixel(self, column, row):

        return self.pixels[column][row]

    def getPixels(self):

        return self.pixels

    def getAsciiString(self):

        rows = []
        data = util.arrayInvert(self.pixels)
        for row in data:
            ascii = map(asciiGrayscaleConversionFunction, row)
            rows.append("".join(ascii))
        return "\n".join(rows)

    def __str__(self):
        return self.getAsciiString()


# Data processing, cleanup and display functions

def loadDataFile(filename, n, width, height):
    DATUM_WIDTH = width
    DATUM_HEIGHT = height
    fin = readlines(filename)
    fin.reverse()
    items = list()
    for i in range(n):
        data = list()
        for j in range(height):
            data.append(list(fin.pop()))
        if len(data[0]) < DATUM_WIDTH - 1:
            # we encountered end of file...
            print("Truncating at %d examples (maximum)" % i)
            break
        items.append(Datum(data, DATUM_WIDTH, DATUM_HEIGHT))
    return items


import zipfile
import os


def readlines(filename):
    if os.path.exists(filename):
        return [l[:-1] for l in open(filename).readlines()]
    else:
        z = zipfile.ZipFile('../../../Desktop/classification/data.zip')
        return z.read(filename).split('\n')


def loadLabelsFile(filename, n):
    fin = readlines(filename)
    labels = []
    for line in fin[:min(n, len(fin))]:
        if line == '':
            break
        labels.append(int(line))
    return labels


def asciiGrayscaleConversionFunction(value):
    if value == 0:
        return ' '
    elif value == 1:
        return '+'
    elif value == 2:
        return '#'


def IntegerConversionFunction(character):
    if character == ' ':
        return 0
    elif character == '+':
        return 1
    elif character == '#':
        return 2


def convertToInteger(data):
    if not isinstance(data, type([])):
        return IntegerConversionFunction(data)
    else:
        return map(convertToInteger, data)


# Testing

def _test():
    import doctest
    doctest.testmod()  # Test the interactive sessions in function comments
    n = 1
    #  items = loadDataFile("data/facedata/facedatatrain", n,60,70)
    #  labels = loadLabelsFile("data/facedata/facedatatrainlabels", n)
    items = loadDataFile("data/digitdata/trainingimages", n, 28, 28)
    labels = loadLabelsFile("data/digitdata/traininglabels", n)
    for i in range(1):
        print(items[i])
        print(items[i])
        print(items[i].height)
        print(items[i].width)
        print(type((items[i])))
        print(items[i].getPixels())


if __name__ == "__main__":
    _test()

import perceptron_main
import numpy as np

TOTAL_FACE_DATA = 451
TOTAL_DIGIT_DATA = 5000

def main():
    calculate_data()

def calculate_data():
    dsets = ["faces", "digits"]
    training_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for dset in dsets:
        means = list()
        stds = list()
        for training_percent in training_percentages:
            accs = np.zeros(shape=5, dtype=float)
            for i in range(5):
                if dset == "faces":
                    training_amount = int(float(training_percent) / 100 * TOTAL_FACE_DATA)
                    testing_amount = 150
                else:
                    training_amount = int(float(training_percent) / 100 * TOTAL_DIGIT_DATA)
                    testing_amount = 1000
                print(dset, training_amount, testing_amount)
                accs[i] = perceptron_main.runClassifierStats(dset, training_amount, testing_amount)
            means.append(accs.mean())
            stds.append(accs.std())
        print('Mean and Standard Deviation Info:')
        for i in range(len(training_percentages)):
            print('Training Data Percentage: %.2f%%, mean(accuracy): %.2f%%, std(accuracy): %.2f' % (training_percentages[i], means[i], stds[i]))

if __name__ == '__main__':
    main()
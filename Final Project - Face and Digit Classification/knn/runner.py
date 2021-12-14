import knn_main
import numpy as np

dsets = ['faces', 'digits']
training_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

for dset in dsets:
    filename = 'knn_test_results_' + dset
    means = list()
    stds = list()
    for training_percentage in training_percentages:
        args = [dset, '-p', str(training_percentage), '-f']
        print('Running knn_main with args:', *args)
        accs = np.zeros(shape=5, dtype=float)
        for i in range(5):
            print('\titeration:', str(i+1))
            accs[i] = knn_main.main(args)
        means.append(accs.mean())
        stds.append(accs.std())
    with open(filename, 'a') as f:
        print('Mean and Standard Deviation Info:', file=f)
        for i in range(len(training_percentages)):
            print('Training Data Percentage: %.2f%%, mean(accuracy): %.2f%%, std(accuracy): %.2f' % (training_percentages[i], means[i], stds[i]), file=f)

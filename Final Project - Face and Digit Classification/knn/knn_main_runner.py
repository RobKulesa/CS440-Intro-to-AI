import knn_main
import re

dsets = ['faces', 'digits']
training_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
iterations = 5


def main():
    # calculate_results()
    avg_runtimes()


def calculate_results():
    for dset in dsets:
        filename = 'knn_test_results_' + dset
        means = list()
        stds = list()
        for training_percentage in training_percentages:
            args = [dset, '-p', str(training_percentage), '-f']
            print('Running knn_main with args:', *args)
            accs = np.zeros(shape=5, dtype=float)
            for i in range(iterations):
                print('\titeration:', str(i+1))
                accs[i] = knn_main.main(args)
            means.append(accs.mean())
            stds.append(accs.std())
        with open(filename, 'a') as f:
            print('Mean and Standard Deviation Info:', file=f)
            for i in range(len(training_percentages)):
                print('Training Data Percentage: %.2f%%, mean(accuracy): %.2f%%, std(accuracy): %.2f' % (training_percentages[i], means[i], stds[i]), file=f)


def sum_runtimes():
    for dset in dsets:
        filename = 'knn_test_results_' + dset
        with open(filename, 'r') as file:
            lines = file.read()
            matches = re.findall(r'(?<=Testing duration : ).*(?= seconds)', lines)
            total = sum(list(map(float, matches)))
            print('Total Runtime for %s: %.2f minutes, or %.2f hours' % (dset, total/60, total/3600))


def avg_runtimes():
    for dset in dsets:
        filename = 'knn_test_results_' + dset
        with open(filename, 'r') as file:
            lines = file.read()
            counter = 0
            percent = 10
            sum = 0
            for match in re.finditer(r'(?<=Testing duration : ).*(?= seconds)', lines):
                sum += float(match[0])
                if counter < 4:
                    counter += 1
                else:
                    print('Avg Runtime for %s at %.1f%% training data: %.2f seconds' % (dset, percent, sum / iterations))
                    sum = 0
                    counter = 0
                    percent += 10



if __name__ == '__main__':
    main()

import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from experiment import get_data

def main(argname):
    with open('archive/heart.csv', mode='r') as data_file:
        colnames = np.array(next(data_file).strip().split(',')[:-1])
    with open(f'best_model_{argname}.pkl', mode='rb') as model_file:
        model = pickle.load(model_file)

    inputs_train, inputs_test, targets_train, targets_test = get_data()

    per = permutation_importance(model, inputs_test, targets_test, n_repeats=30, random_state=0)
    box_array = np.zeros((2, 13))
    idxes = per.importances_mean.argsort()[::-1]
    for i in idxes:
        print(
            f"{colnames[i]:<8}"
            f"{per.importances_mean[i]:.3f}"
            f" +/- {per.importances_std[i]:.3f}"
        )
        box_array[0, i] = per.importances_mean[i]
        box_array[1, i] = per.importances_std[i]
    plt.boxplot(box_array)
    plt.show()

if __name__ == '__main__':
    # main('learning_rate_init')
    main('hidden_layer_sizes')

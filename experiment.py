from pathlib import Path
import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

PLOT_FOLDER = Path('plots')
random_forest = False

def get_data():
    dataset = np.genfromtxt(
        'archive/heart.csv', delimiter=',', skip_header=1, dtype=np.float32
    )
    return preprocess(dataset[:, :13], dataset[:, 13])

def learn_risk(argname, argvals):
    with open('archive/heart.csv', mode='r') as data_file:
        colnames = np.array(next(data_file).strip().split(',')[:-1])

    inputs_train, inputs_test, targets_train, targets_test = get_data()

    # puts the dataset into the classifier to train
    models = []
    for i, value in enumerate(argvals):
        if(random_forest):
            classifier = RandomForestClassifier(random_state=0)
        else:
            classifier = MLPClassifier(random_state=0, max_iter=1200, **{argname: value})
        classifier.fit(inputs_train, targets_train)
        predictions_train = classifier.predict(inputs_train)
        predictions_test = classifier.predict(inputs_test)

        display_confusion_matrix(targets_train, predictions_train, plot_title=f'{argname} Train Performance')
        display_confusion_matrix(targets_test, predictions_test, plot_title=f'{argname} Test Performance')
        # one line accuracy of the machine learning
        print(f'\nTrain Accuracy for {argname} with {value = }: {np.mean(np.equal(predictions_train, targets_train)) * 100:.3f}%')
        print(f'Test Accuracy for {argname} with {value = }: {np.mean(np.equal(predictions_test, targets_test)) * 100:.3f}%')
        if (not random_forest):
            print(f'Best Loss for {argname} with {value = }: {classifier.best_loss_}%')
        models.append(classifier)
        if (random_forest):
            for i, value in enumerate (classifier.feature_importances_):
                print(f'{colnames[i]}: {value}')

    if (not random_forest):
        plot(argvals, models, argname)

        # Compute argmax to attain best model

        best_model = min((model.best_loss_, model) for model in models)[1]
        model_file_name = f'best_model_{argname}.pkl'
        with open(model_file_name, mode='wb') as model_file:
            pickle.dump(best_model, model_file)
        print(f'Successfully saved `{model_file_name}`')

def plot(argvals, models, argname):
    for value, classifier in zip(argvals, models):
        plt.title('Hyperparameter experimentation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(range(len(classifier.loss_curve_)), classifier.loss_curve_, label=f'{value=}')
    plt.legend()
    PLOT_FOLDER.mkdir(exist_ok=True)

    if argname == "learning_rate_init":
        plt.xscale("log")

    plt.savefig(PLOT_FOLDER / f'{argname}_epoch_vs_loss.png')
    plt.close()
    plt.title('Final Loss')
    plt.xlabel('Experiment number')
    plt.ylabel('Loss')
    ext = np.zeros(len(argvals))
    for x in range(len(argvals)):
        ext[x] = x + 1
    plt.plot(ext, [m.best_loss_ for m in models])
    plt.savefig(PLOT_FOLDER / f'{argname}_vs_best_loss.png')
    plt.close()

# preprocessing and making a random test group to pull from
def preprocess(inputs, targets):
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.10, random_state=0,
    )
    inputs_train = inputs_train / np.float32(255.)
    inputs_test = inputs_test / np.float32(255.)
    return inputs_train, inputs_test, targets_train, targets_test

def display_confusion_matrix(target, predictions, labels=['Low Risk', 'High Risk'], plot_title='Performance'):
    cm = confusion_matrix(target, predictions)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots()
    cm_display.plot(ax=ax)
    ax.set_title(plot_title)
    plt.savefig(PLOT_FOLDER / f'{plot_title.lower().replace(" ", "_")}.png')
    plt.close()

if __name__ == '__main__':
    learn_risk('learning_rate_init', np.array([380, 400, 470]))
    learn_risk('hidden_layer_sizes', [(100,) * i for i in range(1, 5)])

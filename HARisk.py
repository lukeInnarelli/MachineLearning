from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

PLOT_FOLDER = Path('plots')

def get_data():
    dataset = np.genfromtxt(
        'archive/heart.csv', delimiter=',', skip_header=1, dtype=np.float32
    )
    return preprocess(dataset[:, :13], dataset[:, 13])

def learn_risk():
    inputs_train, inputs_test, targets_train, targets_test = get_data()

    # puts the dataset into the classifier to train
    tests = np.array([380,400,470])
    classes = np.zeros(len(tests))
    param = "learning_rate_init"
    for i, value in enumerate(tests):
        classifier = MLPClassifier(random_state=0, max_iter=1200, **{param:value},)
        classifier.fit(inputs_train, targets_train)
        predictions_train = classifier.predict(inputs_train)
        predictions_test = classifier.predict(inputs_test)

        # one line accuracy of the machine learning
        print(f'\nTrain Accuracy for {value = }: {np.mean(np.equal(predictions_train, targets_train)) * 100:.3f}%')
        print(f'Test Accuracy for {value = }: {np.mean(np.equal(predictions_test, targets_test)) * 100:.3f}%')
        plt.title('Hyperparameter experimentation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(range(len(classifier.loss_curve_)), classifier.loss_curve_, label=f'{value=}')
        classes[i] = classifier.best_loss_
        #display_confusion_matrix(targets_train, predictions_train, plot_title=f'Train Performance {hidden_layer_sizes}')
        #display_confusion_matrix(targets_test, predictions_test, plot_title=f'Test Performance {hidden_layer_sizes}')
    plot(tests, classes, param)

def plot(tests, classes, param):
    plt.legend()
    PLOT_FOLDER.mkdir(exist_ok=True)

    if param == "learning_rate_init":
        plt.xscale("log")

    plt.savefig(PLOT_FOLDER / 'epoch_vs_loss.png')
    plt.close()
    plt.title('Final Loss')
    plt.xlabel('Experiment number')
    plt.ylabel('Loss')
    ext = np.zeros(len(tests))
    for x in range(len(tests)):
        ext[x] = x + 1
    plt.plot(ext, classes)
    plt.savefig(PLOT_FOLDER / 'hidden_layer_sizes_vs_best_loss.png')
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
    learn_risk()

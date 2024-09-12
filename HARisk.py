from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def get_data():
    dataset = np.genfromtxt(
        'archive/heart.csv', delimiter=',', skip_header=1, dtype=np.float32
    )
    return preprocess(dataset[:, :13], dataset[:, 13])

def learnRisk():

    inputs_train, inputs_test, targets_train, targets_test = get_data()

    # puts the dataset into the classifier to train
    i = 0
    tests = np.array([470, 475, 480, 485, 490, 495, 1000])
    classes = np.zeros(len(tests))
    for hidden_layer_sizes in tests:
        classifier = MLPClassifier(random_state=0, max_iter=1200, hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=0.06)
        classifier.fit(inputs_train, targets_train)
        results = classifier.predict(inputs_test)

        # one line accuracy of the machine learning
        print(np.mean(np.equal(results, targets_test)))
        plt.title('Hyperparameter experimentation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(range(len(classifier.loss_curve_)), classifier.loss_curve_, label=f'{hidden_layer_sizes=}')
        classes[i] = classifier.best_loss_
        i = i + 1
        # display_confusion_matrix(targets_test, results, plot_title='Test Performance')
    plt.legend()
    plt.show()
    plt.title('Final Loss')
    plt.xlabel('Experiment number')
    plt.ylabel('Loss')
    ext = np.zeros(len(tests))
    for x in range(len(tests)):
        ext[x] = x + 1
    plt.plot(ext, classes)
    plt.legend()
    plt.show()

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
    plt.show()

if __name__ == '__main__':
    learnRisk()

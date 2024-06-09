from matplotlib import pyplot
import numpy as np
import matplotlib.pyplot as plt

def summarise_diagnostics(history):
    # plot accuracy
    pyplot.subplot(211)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')

    # plot loss
    pyplot.subplot(212)
    pyplot.title('Mean Squared Error Loss')
    pyplot.plot(history.history['loss'], color='orange', label='train')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')

    pyplot.tight_layout()
    pyplot.show()

def save_losses(losses):
    t = np.arange(len(losses))
    plt.plot(t, losses)
    plt.savefig('loss.png')
    # plt.show()
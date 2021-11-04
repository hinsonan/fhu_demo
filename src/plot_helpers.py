import matplotlib.pyplot as plt

def plot_net_history(history):
    _, axes = plt.subplots(2)
    plt.subplots_adjust(hspace=0.9)
    axes[0].plot(history['accuracy'])
    axes[0].plot(history['val_accuracy'])
    axes[0].set_title('model accuracy')
    axes[0].set_ylabel('accuracy')
    axes[0].set_xlabel('epoch')
    axes[0].legend(['train', 'validation'], loc='upper left')


    axes[1].plot(history['loss'])
    axes[1].plot(history['val_loss'])
    axes[1].set_title('model loss')
    axes[1].set_ylabel('loss')
    axes[1].set_xlabel('epoch')
    axes[1].legend(['train', 'validation'], loc='upper right')
    plt.savefig(f'src/neural_net_accuracy_and_loss')
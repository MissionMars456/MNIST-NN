import matplotlib.pyplot as plt
import tensorflow as tf

(train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255

def showImage(i):
    plt.imshow(train_images[i], cmap='Grays')
    plt.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.colorbar()
    plt.title(f"Target: {train_labels[i]}")
    plt.show()

showImage(0)
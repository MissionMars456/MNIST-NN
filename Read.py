import tensorflow as tf
import matplotlib.pyplot as plt

(train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255

def showImage(i):
    plt.imshow(train_images[i], cmap='Grays')
    plt.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.colorbar()
    plt.title(f"Target: {train_labels[i]}")
    plt.figtext(0.5, 0.07, f"Prediction {i}", 
                ha="center")

    plt.show()

showImage(0)
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Get Data
(train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
train_images = (train_images / 255)[:100]
flattened_images = tf.reshape(train_images, [100, 784])

# Init Weights & Biases
std_l1 = np.sqrt(2.0 / 784)
weights_l1 = np.random.normal(loc=0.0, scale=std_l1, size=(784, 16)).astype(np.float32)
bias_l1 = np.zeros(16)

std_l2 = np.sqrt(2.0 / 16)
weights_l2 = np.random.normal(loc=0.0, scale=std_l2, size=(16, 16)).astype(np.float32)
bias_l2 = np.zeros(16)

std_l3 = np.sqrt(2.0 / 16)
weights_l3 = np.random.normal(loc=0.0, scale=std_l3, size=(16, 10)).astype(np.float32)
bias_l3 = np.zeros(10)

# Model
def pass_layer(input, weights, biases):
    calculated_layer = tf.matmul(input, weights) + biases
    
    return calculated_layer

# Layer 1 with ReLU
raw_l1_outputs = pass_layer(flattened_images, weights_l1, bias_l1)
activated_l1_outputs = tf.maximum(0.0, raw_l1_outputs)

# Layer 2 with ReLU
raw_l2_outputs = pass_layer(activated_l1_outputs, weights_l2, bias_l2)
activated_l2_outputs = tf.maximum(0.0, raw_l2_outputs)

# Layer 3 without ReLU
l3_outputs = pass_layer(activated_l2_outputs, weights_l3, bias_l3)

predictions = tf.argmax(l3_outputs, axis=1)

def showImage(i):
    plt.imshow(train_images[i], cmap='Grays')
    plt.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.colorbar()
    plt.title(f"Target: {train_labels[i]}")
    plt.figtext(0.5, 0.07, f"Prediction {predictions[i]}", 
                ha="center")

    plt.show()

i = 0
while True:
    i += 1
    showImage(i)
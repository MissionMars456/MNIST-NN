import numpy as np
import tensorflow as tf

# Get Data
(train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
train_images = (train_images / 255)[:100]
flattened_images = tf.reshape(train_images, [100, 784])

# Init Weights & Biases
std_l1 = np.sqrt(2.0 / 784)
weights_l1 = np.random.normal(loc=0.0, scale=std_l1, size=(784, 16)).astype(np.float32)
bias_l1 = np.zeros(784)

std_l2 = np.sqrt(2.0 / 16)
weights_l2 = np.random.normal(loc=0.0, scale=std_l2, size=(16, 16)).astype(np.float32)
bias_l2 = np.zeros(16)

std_l3 = np.sqrt(2.0 / 16)
weights_l3 = np.random.normal(loc=0.0, scale=std_l3, size=(16, 10)).astype(np.float32)
bias_l3 = np.zeros(10)

# Model
def pass_layer(input, weights, biases):
    calculated_layer = (input * weights) + biases
    
    return calculated_layer

for i in range(len(train_images)):
    current_input = flattened_images[i]
    print(tf.shape(current_input))
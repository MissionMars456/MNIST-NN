import numpy as np

inputVals = hidden1Vals = hidden2Vals = outputVals = None
hidden1Weights = hidden2Weights = outputWeights = None
hidden1Biases = hidden2Biases = outputBiases = None

def softmax(vals):
    max_z = np.max(vals, axis=1, keepdims=True)
    exp_z = np.exp(vals - max_z)

    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def startupRand():
    global hidden1Weights, hidden2Weights, outputWeights
    global hidden1Biases, hidden2Biases, outputBiases

    hidden1Weights = np.random.rand(784, 128) * np.sqrt(2.0 / 784)
    hidden2Weights = np.random.rand(128, 64) * np.sqrt(2.0 / 128)
    outputWeights = np.random.rand(64, 10) * np.sqrt(2.0 / 64)

    hidden1Biases = np.zeros((1, 128))
    hidden2Biases = np.zeros((1, 64))
    outputBiases = np.zeros((1, 10))

    write()

def write():
    np.savez(
        "MatrixValues.npz",
        h1W=hidden1Weights,
        h2W=hidden2Weights,
        outW=outputWeights,
        h1B=hidden1Biases,
        h2B=hidden2Biases,
        outB=outputBiases
    )

def read():
    global hidden1Weights, hidden2Weights, outputWeights
    global hidden1Biases, hidden2Biases, outputBiases
    global inputVals, hidden1Vals, hidden2Vals, outputVals

    loadedMatrixValues = np.load("MatrixValues.npz")

    hidden1Weights = loadedMatrixValues["h1W"]
    hidden2Weights = loadedMatrixValues["h2W"]
    outputWeights = loadedMatrixValues["outW"]

    hidden1Biases = loadedMatrixValues["h1B"]
    hidden2Biases = loadedMatrixValues["h2B"]
    outputBiases = loadedMatrixValues["outB"]

def backpropagate(targetOneHot):
    global hidden1Weights, hidden2Weights, outputWeights
    global hidden1Biases, hidden2Biases, outputBiases

    outputError = outputVals - targetOneHot

    outputWeights_gradient = np.dot(hidden2Vals.T, outputError)
    outputBiases_gradient = np.sum(outputError, axis=0, keepdims=True)

    hidden2Error = np.dot(outputError, outputWeights.T) * (hidden2Vals > 0)

    hidden2Weights_gradient = np.dot(hidden1Vals.T, hidden2Error)
    hidden2Biases_gradient = np.sum(hidden2Error, axis=0, keepdims=True)

    hidden1Error = np.dot(hidden2Error, hidden2Weights.T) * (hidden1Vals > 0)

    hidden1Weights_gradient = np.dot(inputVals.T, hidden1Error)
    hidden1Biases_gradient = np.sum(hidden1Error, axis=0, keepdims=True)

    # Update
    learning_rate = 0.01

    outputWeights -= learning_rate * outputWeights_gradient
    outputBiases -= learning_rate * outputBiases_gradient

    hidden2Weights -= learning_rate * hidden2Weights_gradient
    hidden2Biases -= learning_rate * hidden2Biases_gradient

    hidden1Weights -= learning_rate * hidden1Weights_gradient
    hidden1Biases -= learning_rate * hidden1Biases_gradient

    write()

def runNet(image, goal):
    global inputVals, hidden1Vals, hidden2Vals, outputVals

    inputVals=image
    read()

    hidden1_pre = np.dot(inputVals, hidden1Weights) + hidden1Biases
    hidden1Vals = np.maximum(0, hidden1_pre)
    hidden2_pre = np.dot(hidden1Vals, hidden2Weights) + hidden2Biases
    hidden2Vals = np.maximum(0, hidden2_pre)
    output_pre = np.dot(hidden2Vals, outputWeights) + outputBiases
    outputVals = softmax(output_pre)

    if goal:
        backpropagate(goal)

    return outputVals
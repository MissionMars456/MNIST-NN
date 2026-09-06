import os

import numpy as np
from tensorflow.keras.datasets import mnist

import Neural_Net as nn

np.set_printoptions(suppress=True)

(xTrainRaw, yTrainRaw), (_, _) = mnist.load_data()
xTrain = xTrainRaw.reshape(-1,784).astype('float32') / 255.0
yTrain = np.eye(10)[yTrainRaw]

nn.startupRand()

epochs = 10
batchSize = 60000

history = []

for epoch in range(epochs):
    correct_predictions = 0
    
    for i in range(batchSize):
        img = xTrain[i:i+1]
        label = yTrain[i:i+1]
        
        prediction = nn.runNet(img, label)
        
        if np.argmax(prediction) == yTrainRaw[i]:
            correct_predictions += 1

        if (i + 1) % 5000 == 0:
            os.system("cls")
            
            print(f"Training network on {batchSize} images for {epochs} epochs...")
            for past_log in history:
                print(past_log)
            print(f"\n--- Current Epoch {epoch + 1}/{epochs} ---")
            
            # Draw the progress bar
            blocks = int(((i + 1) / batchSize) * 20)
            print(f"Processed {i + 1}/{batchSize} |{'█'*blocks:<20}|")

    accuracy = (correct_predictions / batchSize) * 100
    history.append(f"Epoch {epoch + 1}/{epochs} | Accuracy: {accuracy:.2f}%")

os.system("clear")
print(f"Training network on {batchSize} images for {epochs} epochs...")
for past_log in history:
    print(past_log)

print("\nTraining complete! Weights saved to MatrixValues.npz.")
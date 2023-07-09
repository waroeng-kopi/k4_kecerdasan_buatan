import numpy as np
import pandas as pd

class NeuralNetwork():

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize the weights and biases of the neural network
        self.weights_1 = np.random.randn(input_size, hidden_size)
        self.biases_1 = np.random.randn(hidden_size)
        self.weights_2 = np.random.randn(hidden_size, output_size)
        self.biases_2 = np.random.randn(output_size)

    def forward_propagate(self, inputs):
        # Calculate the output of the hidden layer
        hidden_layer_output = np.dot(inputs, self.weights_1) + self.biases_1
        hidden_layer_activation = np.sigmoid(hidden_layer_output)

        # Calculate the output of the output layer
        output_layer_output = np.dot(hidden_layer_activation, self.weights_2) + self.biases_2
        return output_layer_output

    def backpropagate(self, inputs, targets):
        # Calculate the error at the output layer
        output_layer_error = targets - self.forward_propagate(inputs)

        # Calculate the error at the hidden layer
        hidden_layer_error = np.dot(output_layer_error, self.weights_2.T) * hidden_layer_activation * (1 - hidden_layer_activation)

        # Update the weights and biases of the neural network
        self.weights_1 += np.dot(inputs.T, hidden_layer_error) * self.learning_rate
        self.biases_1 += np.mean(hidden_layer_error, axis=0) * self.learning_rate
        self.weights_2 += np.dot(hidden_layer_activation.T, output_layer_error) * self.learning_rate
        self.biases_2 += np.mean(output_layer_error, axis=0) * self.learning_rate

    def train(self, inputs, targets, epochs, learning_rate):
        # Iterate over the epochs
        for epoch in range(epochs):
            # Iterate over the inputs and targets
            for input, target in zip(inputs, targets):
                # Backpropagate the error
                self.backpropagate(input, target)

    def predict(self, inputs):
        # Forward propagate the input data through the neural network
        return self.forward_propagate(inputs)

if __name__ == "__main__":

    # Load the dataset
    df = pd.read_csv("kc_house_prices.csv")

    # Extract the features and the target
    features = df.drop("price", axis=1)
    target = df["price"]

    # Create a neural network
    neural_network = NeuralNetwork(input_size=len(features.columns), hidden_size=100, output_size=1)

    # Train the neural network
    neural_network.train(features, target, epochs=1000, learning_rate=0.001)

    # Make a prediction
    prediction = neural_network.predict(features.iloc[0])

    # Print the prediction
    print(prediction)

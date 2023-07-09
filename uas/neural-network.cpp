#include <iostream>
#include <vector>
#include <cmath>

// Define the activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Define the derivative of the activation function
double sigmoidDerivative(double x) {
    double sigmoidX = sigmoid(x);
    return sigmoidX * (1.0 - sigmoidX);
}

class NeuralNetwork {
private:
    int numInputNeurons;
    int numHiddenNeurons;
    int numOutputNeurons;
    std::vector<std::vector<double>> hiddenWeights;
    std::vector<double> hiddenBiases;
    std::vector<std::vector<double>> outputWeights;
    std::vector<double> outputBiases;
    double learningRate;
    int numIterations;

public:
    NeuralNetwork(int numInputs, int numHidden, int numOutputs, double learningRate, int numIterations)
        : numInputNeurons(numInputs), numHiddenNeurons(numHidden), numOutputNeurons(numOutputs),
          learningRate(learningRate), numIterations(numIterations) {
        
        // Initialize weights and biases randomly
        hiddenWeights.resize(numInputNeurons, std::vector<double>(numHiddenNeurons));
        hiddenBiases.resize(numHiddenNeurons);
        outputWeights.resize(numHiddenNeurons, std::vector<double>(numOutputNeurons));
        outputBiases.resize(numOutputNeurons);

        // Randomly initialize the weights and biases
        initializeWeightsAndBiases();
    }

    void initializeWeightsAndBiases() {
        // Initialize the weights and biases randomly between -1 and 1
        for (int i = 0; i < numInputNeurons; ++i) {
            for (int j = 0; j < numHiddenNeurons; ++j) {
                hiddenWeights[i][j] = (2.0 * rand() / RAND_MAX) - 1.0;
            }
        }

        for (int i = 0; i < numHiddenNeurons; ++i) {
            hiddenBiases[i] = (2.0 * rand() / RAND_MAX) - 1.0;
        }

        for (int i = 0; i < numHiddenNeurons; ++i) {
            for (int j = 0; j < numOutputNeurons; ++j) {
                outputWeights[i][j] = (2.0 * rand() / RAND_MAX) - 1.0;
            }
        }

        for (int i = 0; i < numOutputNeurons; ++i) {
            outputBiases[i] = (2.0 * rand() / RAND_MAX) - 1.0;
        }
    }

    std::vector<double> forwardPropagation(const std::vector<double>& inputs) {
        std::vector<double> hiddenActivations(numHiddenNeurons);
        std::vector<double> outputActivations(numOutputNeurons);

        // Compute the activations of the hidden layer
        for (int j = 0; j < numHiddenNeurons; ++j) {
            double weightedSum = 0.0;
            for (int i = 0; i < numInputNeurons; ++i) {
                weightedSum += inputs[i] * hiddenWeights[i][j];
            }
            weightedSum += hiddenBiases[j];
            hiddenActivations[j] = sigmoid(weightedSum);
        }

        // Compute the activations of the output layer
        for (int k = 0; k < numOutputNeurons; ++k) {
            double weightedSum = 0.0;
            for (int j = 0; j < numHiddenNeurons; ++j) {
                weightedSum += hiddenActivations[j] * outputWeights[j][k];
            }
            weightedSum += outputBiases[k];
            outputActivations[k] = sigmoid(weightedSum);
        }

        return outputActivations;
    }

    void backpropagation(const std::vector<double>& inputs, const std::vector<double>& targets) {
        std::vector<double> hiddenActivations(numHiddenNeurons);
        std::vector<double> outputActivations(numOutputNeurons);
        std::vector<double> outputDeltas(numOutputNeurons);
        std::vector<double> hiddenDeltas(numHiddenNeurons);

        // Compute the activations of the hidden layer
        for (int j = 0; j < numHiddenNeurons; ++j) {
            double weightedSum = 0.0;
            for (int i = 0; i < numInputNeurons; ++i) {
                weightedSum += inputs[i] * hiddenWeights[i][j];
            }
            weightedSum += hiddenBiases[j];
            hiddenActivations[j] = sigmoid(weightedSum);
        }

        // Compute the activations of the output layer
        for (int k = 0; k < numOutputNeurons; ++k) {
            double weightedSum = 0.0;
            for (int j = 0; j < numHiddenNeurons; ++j) {
                weightedSum += hiddenActivations[j] * outputWeights[j][k];
            }
            weightedSum += outputBiases[k];
            outputActivations[k] = sigmoid(weightedSum);
        }

        // Compute the deltas of the output layer
        for (int k = 0; k < numOutputNeurons; ++k) {
            double outputError = targets[k] - outputActivations[k];
            outputDeltas[k] = outputError * sigmoidDerivative(outputActivations[k]);
        }

        // Compute the deltas of the hidden layer
        for (int j = 0; j < numHiddenNeurons; ++j) {
            double hiddenError = 0.0;
            for (int k = 0; k < numOutputNeurons; ++k) {
                hiddenError += outputDeltas[k] * outputWeights[j][k];
            }
            hiddenDeltas[j] = hiddenError * sigmoidDerivative(hiddenActivations[j]);
        }

        // Update the weights and biases of the output layer
        for (int j = 0; j < numHiddenNeurons; ++j) {
            for (int k = 0; k < numOutputNeurons; ++k) {
                outputWeights[j][k] += learningRate * hiddenActivations[j] * outputDeltas[k];
            }
        }

        for (int k = 0; k < numOutputNeurons; ++k) {
            outputBiases[k] += learningRate * outputDeltas[k];
        }

        // Update the weights and biases of the hidden layer
        for (int i = 0; i < numInputNeurons; ++i) {
            for (int j = 0; j < numHiddenNeurons; ++j) {
                hiddenWeights[i][j] += learningRate * inputs[i] * hiddenDeltas[j];
            }
        }

        for (int j = 0; j < numHiddenNeurons; ++j) {
            hiddenBiases[j] += learningRate * hiddenDeltas[j];
        }
    }

    void train(const std::vector<std::vector<double>>& trainingData) {
        for (int iteration = 0; iteration < numIterations; ++iteration) {
            for (const auto& data : trainingData) {
                std::vector<double> inputs(data.begin(), data.end() - numOutputNeurons);
                std::vector<double> targets(data.end() - numOutputNeurons, data.end());
                backpropagation(inputs, targets);
            }
        }
    }
};

int main() {
    // Example usage
    int numInputs = 2;
    int numHidden = 4;
    int numOutputs = 1;
    double learningRate = 0.1;
    int numIterations = 1000;

    NeuralNetwork network(numInputs, numHidden, numOutputs, learningRate, numIterations);

    // Training data: XOR problem
    std::vector<std::vector<double>> trainingData = {
        {0, 0, 0},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 0}
    };

    network.train(trainingData);

    // Test the trained network
    std::vector<double> input = {1, 1};
    std::vector<double> output = network.forwardPropagation(input);

    std::cout << "Input: " << input[0] << ", " << input[1] << std::endl;
    std::cout << "Output: " << output[0] << std::endl;

    return 0;
}

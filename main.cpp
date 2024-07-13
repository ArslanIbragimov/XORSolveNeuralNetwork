#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>


double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}
double sigmoid_derivative(double x) {
    return x * (1 - x);
}
double getRandom() {
    return static_cast<double>(rand()) / RAND_MAX;
}

int main() {
    srand(static_cast<unsigned>(time(0)));
    std::vector<std::vector<double>> inputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    std::vector<double> outputs = { 0, 1, 1, 0 };
    int inputLayerNeurons = 2;
    int hiddenLayerNeurons = 2;
    int outputLayerNeurons = 1;
    std::vector<std::vector<double>> hidden_weights(inputLayerNeurons, std::vector<double>(hiddenLayerNeurons));
    std::vector<double> hidden_bias(hiddenLayerNeurons);
    std::vector<std::vector<double>> output_weights(hiddenLayerNeurons, std::vector<double>(outputLayerNeurons));
    std::vector<double> output_bias(outputLayerNeurons);
    for (int i = 0; i < inputLayerNeurons; ++i)
        for (int j = 0; j < hiddenLayerNeurons; ++j)
            hidden_weights[i][j] = getRandom();

    for (int i = 0; i < hiddenLayerNeurons; ++i)
        hidden_bias[i] = getRandom();

    for (int i = 0; i < hiddenLayerNeurons; ++i)
        for (int j = 0; j < outputLayerNeurons; ++j)
            output_weights[i][j] = getRandom();

    for (int i = 0; i < outputLayerNeurons; ++i)
        output_bias[i] = getRandom();
    double learning_rate = 0.1;
    int epochs = 10000;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<double> hidden_layer_input(hiddenLayerNeurons);
            std::vector<double> hidden_layer_activation(hiddenLayerNeurons);

            for (int j = 0; j < hiddenLayerNeurons; ++j) {
                hidden_layer_input[j] = hidden_bias[j];
                for (int k = 0; k < inputLayerNeurons; ++k) {
                    hidden_layer_input[j] += inputs[i][k] * hidden_weights[k][j];
                }
                hidden_layer_activation[j] = sigmoid(hidden_layer_input[j]);
            }

            std::vector<double> output_layer_input(outputLayerNeurons);
            std::vector<double> predicted_output(outputLayerNeurons);

            for (int j = 0; j < outputLayerNeurons; ++j) {
                output_layer_input[j] = output_bias[j];
                for (int k = 0; k < hiddenLayerNeurons; ++k) {
                    output_layer_input[j] += hidden_layer_activation[k] * output_weights[k][j];
                }
                predicted_output[j] = sigmoid(output_layer_input[j]);
            }
            std::vector<double> error(outputLayerNeurons);
            std::vector<double> d_predicted_output(outputLayerNeurons);

            for (int j = 0; j < outputLayerNeurons; ++j) {
                error[j] = outputs[i] - predicted_output[j];
                d_predicted_output[j] = error[j] * sigmoid_derivative(predicted_output[j]);
            }

            std::vector<double> error_hidden_layer(hiddenLayerNeurons);
            std::vector<double> d_hidden_layer(hiddenLayerNeurons);

            for (int j = 0; j < hiddenLayerNeurons; ++j) {
                error_hidden_layer[j] = 0.0;
                for (int k = 0; k < outputLayerNeurons; ++k) {
                    error_hidden_layer[j] += d_predicted_output[k] * output_weights[j][k];
                }
                d_hidden_layer[j] = error_hidden_layer[j] * sigmoid_derivative(hidden_layer_activation[j]);
            }
            for (int j = 0; j < outputLayerNeurons; ++j) {
                output_bias[j] += d_predicted_output[j] * learning_rate;
                for (int k = 0; k < hiddenLayerNeurons; ++k) {
                    output_weights[k][j] += hidden_layer_activation[k] * d_predicted_output[j] * learning_rate;
                }
            }

            for (int j = 0; j < hiddenLayerNeurons; ++j) {
                hidden_bias[j] += d_hidden_layer[j] * learning_rate;
                for (int k = 0; k < inputLayerNeurons; ++k) {
                    hidden_weights[k][j] += inputs[i][k] * d_hidden_layer[j] * learning_rate;
                }
            }
        }
    }
    std::cout << "Выходные данные после обучения:   \n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::vector<double> hidden_layer_input(hiddenLayerNeurons);
        std::vector<double> hidden_layer_activation(hiddenLayerNeurons);

        for (int j = 0; j < hiddenLayerNeurons; ++j) {
            hidden_layer_input[j] = hidden_bias[j];
            for (int k = 0; k < inputLayerNeurons; ++k) {
                hidden_layer_input[j] += inputs[i][k] * hidden_weights[k][j];
            }
            hidden_layer_activation[j] = sigmoid(hidden_layer_input[j]);
        }
        std::vector<double> output_layer_input(outputLayerNeurons);
        std::vector<double> predicted_output(outputLayerNeurons);

        for (int j = 0; j < outputLayerNeurons; ++j) {
            output_layer_input[j] = output_bias[j];
            for (int k = 0; k < hiddenLayerNeurons; ++k) {
                output_layer_input[j] += hidden_layer_activation[k] * output_weights[k][j];
            }
            predicted_output[j] = sigmoid(output_layer_input[j]);
        }

        std::cout << predicted_output[0] << std::endl;
    }

    return 0;
}

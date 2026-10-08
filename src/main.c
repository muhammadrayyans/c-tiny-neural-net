#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUTS 2
#define HIDDEN 2
#define OUTPUTS 1

#define LEARNING_RATE 0.5
#define MAX_EPOCHS 100000

// -----------------------------
// Activation functions
// -----------------------------

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double output)
{
    return output * (1.0 - output);
}

// -----------------------------
// Random weight initialization
// -----------------------------

double random_weight()
{
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

// -----------------------------
// Neural Network
// -----------------------------

typedef struct
{
    // Input -> Hidden weights
    double w_input_hidden[INPUTS][HIDDEN];

    // Hidden biases
    double bias_hidden[HIDDEN];

    // Hidden -> Output weights
    double w_hidden_output[HIDDEN];

    // Output bias
    double bias_output;

} NeuralNetwork;

// -----------------------------
// Initialize network
// -----------------------------

void initialize_network(NeuralNetwork *network)
{
    for (int i = 0; i < INPUTS; i++)
    {
        for (int j = 0; j < HIDDEN; j++)
        {
            network->w_input_hidden[i][j] = random_weight();
        }
    }

    for (int j = 0; j < HIDDEN; j++)
    {
        network->bias_hidden[j] = random_weight();
        network->w_hidden_output[j] = random_weight();
    }

    network->bias_output = random_weight();
}

// -----------------------------
// Forward Propagation
// -----------------------------

double forward(
    NeuralNetwork *network,
    double input[INPUTS],
    double hidden[HIDDEN])
{
    // Input -> Hidden Layer
    for (int j = 0; j < HIDDEN; j++)
    {
        double sum = network->bias_hidden[j];

        for (int i = 0; i < INPUTS; i++)
        {
            sum += input[i] * network->w_input_hidden[i][j];
        }

        hidden[j] = sigmoid(sum);
    }

    // Hidden -> Output Layer

    double output_sum = network->bias_output;

    for (int j = 0; j < HIDDEN; j++)
    {
        output_sum += hidden[j] * network->w_hidden_output[j];
    }

    return sigmoid(output_sum);
}

// -----------------------------
// Train one example
// -----------------------------

double train(
    NeuralNetwork *network,
    double input[INPUTS],
    double expected)
{
    double hidden[HIDDEN];

    // Forward propagation
    double output = forward(network, input, hidden);

    // -----------------------------
    // Calculate error
    // -----------------------------

    double error = expected - output;

    // Mean Squared Error contribution
    double loss = error * error;

    // -----------------------------
    // Output layer gradient
    // -----------------------------

    double output_delta =
        error * sigmoid_derivative(output);

    // Save old output weights because
    // hidden gradients need them
    double old_output_weights[HIDDEN];

    for (int j = 0; j < HIDDEN; j++)
    {
        old_output_weights[j] =
            network->w_hidden_output[j];
    }

    // -----------------------------
    // Update Hidden -> Output
    // -----------------------------

    for (int j = 0; j < HIDDEN; j++)
    {
        network->w_hidden_output[j] +=
            LEARNING_RATE *
            output_delta *
            hidden[j];
    }

    network->bias_output +=
        LEARNING_RATE * output_delta;

    // -----------------------------
    // Hidden layer gradients
    // -----------------------------

    double hidden_delta[HIDDEN];

    for (int j = 0; j < HIDDEN; j++)
    {
        hidden_delta[j] =
            output_delta *
            old_output_weights[j] *
            sigmoid_derivative(hidden[j]);
    }

    // -----------------------------
    // Update Input -> Hidden
    // -----------------------------

    for (int i = 0; i < INPUTS; i++)
    {
        for (int j = 0; j < HIDDEN; j++)
        {
            network->w_input_hidden[i][j] +=
                LEARNING_RATE *
                hidden_delta[j] *
                input[i];
        }
    }

    // Hidden biases

    for (int j = 0; j < HIDDEN; j++)
    {
        network->bias_hidden[j] +=
            LEARNING_RATE *
            hidden_delta[j];
    }

    return loss;
}

// -----------------------------
// Save trained model
// -----------------------------

void save_model(
    NeuralNetwork *network,
    const char *filename)
{
    FILE *file = fopen(filename, "w");

    if (file == NULL)
    {
        printf("Unable to save model.\n");
        return;
    }

    for (int i = 0; i < INPUTS; i++)
    {
        for (int j = 0; j < HIDDEN; j++)
        {
            fprintf(
                file,
                "%.15f\n",
                network->w_input_hidden[i][j]);
        }
    }

    for (int j = 0; j < HIDDEN; j++)
    {
        fprintf(
            file,
            "%.15f\n",
            network->bias_hidden[j]);
    }

    for (int j = 0; j < HIDDEN; j++)
    {
        fprintf(
            file,
            "%.15f\n",
            network->w_hidden_output[j]);
    }

    fprintf(
        file,
        "%.15f\n",
        network->bias_output);

    fclose(file);
}

// -----------------------------
// Print network parameters
// -----------------------------

void print_network(NeuralNetwork *network)
{
    printf("\n===== Learned Parameters =====\n");

    printf("\nInput -> Hidden Weights\n");

    for (int i = 0; i < INPUTS; i++)
    {
        for (int j = 0; j < HIDDEN; j++)
        {
            printf(
                "w[%d][%d] = %.4f\n",
                i,
                j,
                network->w_input_hidden[i][j]);
        }
    }

    printf("\nHidden Biases\n");

    for (int j = 0; j < HIDDEN; j++)
    {
        printf(
            "b_hidden[%d] = %.4f\n",
            j,
            network->bias_hidden[j]);
    }

    printf("\nHidden -> Output Weights\n");

    for (int j = 0; j < HIDDEN; j++)
    {
        printf(
            "w_output[%d] = %.4f\n",
            j,
            network->w_hidden_output[j]);
    }

    printf(
        "\nOutput Bias = %.4f\n",
        network->bias_output);
}

// -----------------------------
// Main
// -----------------------------

int main()
{
    srand((unsigned int)time(NULL));

    // XOR dataset

    double dataset[4][INPUTS] =
        {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}};

    double expected[4] =
        {
            0,
            1,
            1,
            0};

    NeuralNetwork network;

    initialize_network(&network);

    printf("Training XOR Neural Network...\n");

    // -----------------------------
    // Training loop
    // -----------------------------

    int epoch;

    for (epoch = 0; epoch < MAX_EPOCHS; epoch++)
    {
        double total_loss = 0.0;

        for (int i = 0; i < 4; i++)
        {
            total_loss +=
                train(
                    &network,
                    dataset[i],
                    expected[i]);
        }

        total_loss /= 4.0;

        if (epoch % 5000 == 0)
        {
            printf(
                "Epoch %-6d Loss: %.6f\n",
                epoch,
                total_loss);
        }

        if (total_loss < 0.001)
        {
            break;
        }
    }

    printf(
        "\nTraining completed in %d epochs.\n",
        epoch);

    // -----------------------------
    // Test network
    // -----------------------------

    printf("\n===== XOR Predictions =====\n\n");

    for (int i = 0; i < 4; i++)
    {
        double hidden[HIDDEN];

        double output =
            forward(
                &network,
                dataset[i],
                hidden);

        int prediction =
            output >= 0.5 ? 1 : 0;

        printf(
            "%.0f XOR %.0f -> "
            "raw: %.4f  prediction: %d  "
            "expected: %.0f\n",
            dataset[i][0],
            dataset[i][1],
            output,
            prediction,
            expected[i]);
    }

    print_network(&network);

    save_model(
        &network,
        "model.txt");

    printf(
        "\nModel parameters saved to model.txt\n");

    return 0;
}
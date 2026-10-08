#include <stdio.h>

int prediction(double a, double b, double wa, double wb, double bias);
void train(double a, double b, double out, double *wa, double *wb, double *bias);
void train_algo(int col, int (*arr)[col], int row);

int main()
{
    int data_set[4][3] = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
    train_algo(3, data_set, 4);
    return 0;
}

void train_algo(int col, int (*arr)[col], int row)
{
    double wa = 0, wb = 0, bias = 0, learning_rate = 0.1;
    int epochs = 0;
    while (epochs < 1000)
    {
        int correct = 0;

        for (int i = 0; i < row; i++)
        {
            train(arr[i][0], arr[i][1], arr[i][2], &wa, &wb, &bias);
        }
        for (int i = 0; i < row; i++)
        {
            if (prediction(arr[i][0], arr[i][1], wa, wb, bias) == arr[i][2])
            {
                correct++;
            }
        }
        if (correct == row)
        {
            for(int i = 0; i < row; i++)
            {
                printf("%d %d %d\n", arr[i][0], arr[i][1], arr[i][2]);
            }
            break;
        }
        epochs++;
    }
    printf("Solved in %d epochs\n", epochs);
    printf("%.2f is wa, %.2f is wb, %.2f is bias\n", wa, wb, bias);
}

void train(double a, double b, double out, double *wa, double *wb, double *bias)
{
    double learning_rate = 0.1;

    double error = out - prediction(a, b, *wa, *wb, *bias);
    *wa = *wa + learning_rate * error * a;
    *wb = *wb + learning_rate * error * b;
    *bias = *bias + learning_rate * error;
}

int prediction(double a, double b, double wa, double wb, double bias)
{
    double result = a * wa + b * wb + bias;

    if (result > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

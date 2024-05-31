/**utils.c
 *
 * A collection of tools for matrix and vector manipulation
 * as well as computational tasks. All matrix are assumed
 * in row major order
 */

#include <complex.h>
#include <stdbool.h>
#include <stdio.h>
#include "utils.h"

/**printVector(double *vector, int N) -> void
 *
 * Print an N-th dimensional vector.
 */
void printVector(int N, double vector[N])
{
    printf("[");
    for (int i = 0; i < N - 1; i++)
    {
        printf("%.2f, ", vector[i]);
    }
    printf("%.2f]\n", vector[N - 1]);
}

/**printMatrix(int rows, int cols, double matrix[rows][cols]) -> void
 *
 * Print a rows x cols double matrix in terminal.
 * Assumes row major ordering.
 */
void printMatrix(int rows, int cols, double matrix[rows][cols])
{
    // printf("[");
    for (int i = 0; i < rows; i++)
    {
        printf("[");
        for (int j = 0; j < cols; j++)
        {
            printf("%.3f", matrix[i][j]);
            if (j < cols - 1)
                printf(", ");
        }

        if (i != rows - 1)
            printf("]\n");
    }
    printf("]\n");
}

/**printCMatrix(int rows, int cols, double matrix[rows][cols]) -> void
 *
 * Print a rows x cols complex double matrix in terminal.
 * Assumes row major ordering.
 */
void printCMatrix(int rows, int cols, double complex matrix[rows][cols])
{
    // printf("[");
    for (int i = 0; i < rows; i++)
    {
        printf("(%d): [", i);
        for (int j = 0; j < cols; j++)
        {
            printf("%.3f + %.3fi", creal(matrix[i][j]), cimag(matrix[i][j]));
            if (j < cols - 1)
                printf(", ");
        }

        if (i != rows - 1)
            printf("]\n");
    }
    printf("]\n");
}

/**tranSquare(int N, double A[N][N], double R[N][N]) -> void
 *
 * Transpose the square matrix A. The input matrix must be stored in contiguous
 * memory. Result is passed by pointer R. This operation can be done in place.
 */
void tranSquare(int N, double A[N][N], double R[N][N])
{
    double temp = -1.;

    for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++)
        {
            temp = A[i][j];
            R[i][j] = A[j][i];
            R[j][i] = temp;
        }
}

/**utils.h
 *
 * DEPRECATED
 * A collection of tools for matrix and vector manipulation
 * as well as computational tasks. All matrix are assumed
 * in row major order
 */

#ifndef UTILS_H
#define UTILS_H

/**printVector(int N, double vector[N]) -> void
 *
 * Print an N-th dimensional vector.
 */
void printVector(int N, double vector[N]);

/**printMatrix(int rows, int cols, double matrix[rows][cols]) -> void
 *
 * Print a rows x cols double matrix in terminal.
 * Assumes row major ordering.
 */
void printMatrix(int rows, int cols, double matrix[rows][cols]);

/**printCMatrix(int rows, int cols, double matrix[rows][cols]) -> void
 *
 * Print a rows x cols complex double matrix in terminal.
 * Assumes row major ordering.
 */
void printCMatrix(int rows, int cols, double complex matrix[rows][cols]);

/**tranSquare(int N, double A[N][N], double R[N][N]) -> void
 *
 * Transpose the square matrix A. The input matrix must be stored in contiguous
 * memory. Result is passed by pointer R. This operation can be done in place.
 */
void tranSquare(int N, double A[N][N], double R[N][N]);

#endif /*UTILS_H*/
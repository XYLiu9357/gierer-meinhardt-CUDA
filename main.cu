/**main.cu
 *
 * Solve the Gierer-Meinhardt model with serial implementation.
 * This serves a reference for creating and debugging a parallel
 * version of the same program implemented with CUDA.
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cufft.h>

#include "solver.h"

#ifndef L
#define L 200
#endif

/**
 * Variables in constant memory
 */

#ifndef CONSTANTS
#define CONSTANTS
__device__ static int dev_N;
__device__ static double dev_Du;
__device__ static double dev_Dv;
__device__ static double dev_a;
__device__ static double dev_b;
__device__ static double dev_c;
__device__ static double dev_epsilon;
// __device__ static int dev_steps;
__device__ static double dev_dt;
__device__ static double dev_pi_multiple;
#endif /*CONSTANTS*/

/**
 * Error handler
 */

#ifndef CUDA_ERRHANDLER_H
#define CUDA_ERRHANDLER_H

static void HANDLEERROR(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(1);
    }
}
#define cudaErrChk(err) (HANDLEERROR(err, __FILE__, __LINE__))

#endif /*CUDA_ERRHANDLER_H*/

/**
 * Main
 */

int main(int argc, char *argv[])
{
    /**
     * Choose GPU
     */

    cudaDeviceProp prop;
    int dev;
    memset(&prop, 0, sizeof(cudaDeviceProp));

    // Quadro RTX 8000
    prop.multiProcessorCount = 72;
    prop.major = 7;
    prop.minor = 5;

    cudaChooseDevice(&dev, &prop);
    cudaSetDevice(dev);

    /**
     * Read arguments
     */

    int argi = 0;
    if (argc != 9 && argc != 10)
    {
        printf("argc should be 9 or 10, received %d\n", argc);
        return 1;
    }

    int N = atoi(argv[++argi]);
    double Du = atof(argv[++argi]);
    double Dv = atof(argv[++argi]);
    double a = atof(argv[++argi]);
    double b = atof(argv[++argi]);
    double c = atof(argv[++argi]);
    double epsilon = atof(argv[++argi]);
    int steps = atoi(argv[++argi]);

    double T = 100.;
    double dt = T / steps;
    double pi_multiple = 4. * M_PI * M_PI / L / L;

    // default seed is 0
    unsigned int seed = -1U;

    if (argc == 9)
        seed = atoi(argv[++argi]);

    // print parameters
    printf("-------------------------\n**GIERER PARAMS**\nN = %d; T = %d\nDu = %.2e; Dv = %.2e\na = %.2e; b = %.2e\nc = %.2e; epsilon = %.2e\nseed = %d\n-------------------------\n", N, steps, Du, Dv, a, b, c, epsilon, seed);

    /**
     * Parameter compatibility checks
     */

    // check if N is even
    if (N % 2 != 0)
    {
        printf("Incompatible input N = %d. N must be an even integer.\n", N);
        return 1;
    }

    // due to the way shared memory is used, N cannot exceed 2048
    if (N > 2048)
    {
        printf("N exceeds maximum limit 2048.\n");
        return 1;
    }

    /**
     * Initialize data on host
     * Assumes row major ordering
     */

    // host data arrays
    double(*u)[N] = (double(*)[N])malloc(sizeof(*u) * N);
    double(*v)[N] = (double(*)[N])malloc(sizeof(*v) * N);

    // generate random initial conditions on host
    srand(seed);
    for (int j = 0; j < N; j++)
        for (int k = 0; k < N; k++)
        {
            // generate random data
            u[j][k] = (a + c) / b - 4.5 * drand48();
            v[j][k] = (a + c) * (a + c) / (c * b * b);
        }

    // create file for writing output
    FILE *fileU = fopen("GiererU.out", "w");
    FILE *fileV = fopen("GiererV.out", "w");

    // write initial conditions
    printf("t = %.2f: data stored\n", 0.);
    fwrite(u, sizeof(double), N * N, fileU);
    fwrite(v, sizeof(double), N * N, fileV);
    printf("**Iteration starts**\n");

    /**
     * Initialize data on device
     */

    // store constants constant memory
    cudaMemcpyToSymbol(dev_N, &N, sizeof(int));
    cudaMemcpyToSymbol(dev_Du, &Du, sizeof(double));
    cudaMemcpyToSymbol(dev_Dv, &Dv, sizeof(double));
    cudaMemcpyToSymbol(dev_a, &a, sizeof(double));
    cudaMemcpyToSymbol(dev_b, &b, sizeof(double));
    cudaMemcpyToSymbol(dev_c, &c, sizeof(double));
    cudaMemcpyToSymbol(dev_epsilon, &epsilon, sizeof(double));
    // cudaMemcpyToSymbol(dev_steps, &steps, sizeof(int));
    cudaMemcpyToSymbol(dev_dt, &dt, sizeof(double));
    cudaMemcpyToSymbol(dev_pi_multiple, &pi_multiple, sizeof(double));

    // device data arrays
    cufftDoubleReal *dev_u;
    cufftDoubleReal *dev_v;
    cudaErrChk(cudaMalloc((void **)&dev_u, sizeof(cufftDoubleReal) * N * N));
    cudaErrChk(cudaMalloc((void **)&dev_v, sizeof(cufftDoubleReal) * N * N));

    // device buffers for FFT used to store spectral coefficients
    cufftDoubleComplex *dev_fftBufferU;
    cufftDoubleComplex *dev_fftBufferV;
    cudaErrChk(cudaMalloc((void **)&dev_fftBufferU, sizeof(cufftDoubleComplex) * (N / 2 + 1) * N));
    cudaErrChk(cudaMalloc((void **)&dev_fftBufferV, sizeof(cufftDoubleComplex) * (N / 2 + 1) * N));

    // device laplacian arrays
    cufftDoubleReal *dev_laplacianU;
    cufftDoubleReal *dev_laplacianV;
    cudaErrChk(cudaMalloc((void **)&dev_laplacianU, sizeof(cufftDoubleReal) * N * N));
    cudaErrChk(cudaMalloc((void **)&dev_laplacianV, sizeof(cufftDoubleReal) * N * N));

    // device buffers for Runge-Kutta update
    cufftDoubleReal *dev_rkBufferU;
    cufftDoubleReal *dev_rkBufferV;
    cudaErrChk(cudaMalloc((void **)&dev_rkBufferU, sizeof(cufftDoubleReal) * N * N));
    cudaErrChk(cudaMalloc((void **)&dev_rkBufferV, sizeof(cufftDoubleReal) * N * N));

    // send data
    cudaErrChk(cudaMemcpy(dev_u, u, sizeof(double) * N * N, cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(dev_v, v, sizeof(double) * N * N, cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(dev_rkBufferU, u, sizeof(double) * N * N, cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(dev_rkBufferV, v, sizeof(double) * N * N, cudaMemcpyHostToDevice));

    /**
     * FFT plans
     */

    // configure plans
    cufftHandle forward;
    cufftHandle backward;

    cufftPlan2d(&forward, N, N, CUFFT_D2Z);
    cufftPlan2d(&backward, N, N, CUFFT_Z2D);

    /**
     * Pseudospectral iterations
     */

    double startTime = clock(); // start timer

    for (int iter = 0; iter < steps; iter++)
    {
        for (int rkIter = 0; rkIter < 4; rkIter++)
        {
            // forward transform
            cufftExecD2Z(forward, dev_rkBufferU, dev_fftBufferU);
            cufftExecD2Z(forward, dev_rkBufferV, dev_fftBufferV);

            // calculate laplacians
            // block n process row n
            getLaplacian<<<N, N / 2 + 1, N * sizeof(cufftDoubleComplex)>>>(dev_fftBufferU);
            getLaplacian<<<N, N / 2 + 1, N * sizeof(cufftDoubleComplex)>>>(dev_fftBufferV);

            // backward transform
            cufftExecZ2D(backward, dev_fftBufferU, dev_laplacianU);
            cufftExecZ2D(backward, dev_fftBufferV, dev_laplacianV);

            // time update with RK-4
            // block 2n and 2n + 1 process row n
            // requires a 6 * (N / 2) shared memory
            rkUpdate<<<2 * N, N / 2, 6 * (N / 2) * sizeof(cufftDoubleReal)>>>(dev_rkBufferU, dev_u, dev_laplacianU, dev_rkBufferV, dev_v, dev_laplacianV, rkIter);
        }

        // write intermediate results
        if (iter % (steps / 10) == 0 && iter > 0)
        {
            // cufftDoubleComplex(*testu)[N] = (cufftDoubleComplex(*)[N])malloc(sizeof(*testu) * N);
            // cufftDoubleComplex(*testv)[N] = (cufftDoubleComplex(*)[N])malloc(sizeof(*testv) * N);

            // copy memory from device to host
            cudaMemcpy(u, dev_u, sizeof(double) * N * N, cudaMemcpyDeviceToHost);
            cudaMemcpy(v, dev_v, sizeof(double) * N * N, cudaMemcpyDeviceToHost);

            fwrite(u, sizeof(double), N * N, fileU);
            fwrite(v, sizeof(double), N * N, fileV);
            printf("t = %.2f: data stored\n", iter * dt);
        }
    }

    // write final results
    printf("t = %.2f: data stored\n", T);
    cudaMemcpy(u, dev_u, sizeof(double) * N * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(v, dev_v, sizeof(double) * N * N, cudaMemcpyDeviceToHost);
    fwrite(u, sizeof(double), N * N, fileU);
    fwrite(v, sizeof(double), N * N, fileV);
    fclose(fileU);
    fclose(fileV);

    /**
     * END of pseudospectral iterations
     */

    // end timer
    printf("Elapsed time: %.3fs\n", (double)(clock() - startTime) / CLOCKS_PER_SEC);

    // free resources
    cufftDestroy(forward);
    cufftDestroy(backward);

    cudaFree(dev_fftBufferU);
    cudaFree(dev_fftBufferV);
    cudaFree(dev_laplacianU);
    cudaFree(dev_laplacianV);
    cudaFree(dev_rkBufferU);
    cudaFree(dev_rkBufferV);

    cudaFree(dev_u);
    cudaFree(dev_v);

    free(u);
    free(v);

    return 0;
}

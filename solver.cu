/**solver.cu
 *
 * Kernel functions used for solving the
 * Gierer-Meinhardt model.
 */

#include <stdio.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include "solver.h"

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

/**laplacian(cufftDoubleComplex *dev_fftBuffer)
 * Compute the spectral coefficients for finding the laplacian.
 * Assumes memory is allocated on the device in advance.
 */
__global__ void getLaplacian(cufftDoubleComplex *dev_fftBuffer)
{
    // guard to prevent unspecified core calls
    if (blockIdx.x >= dev_N)
        return;

    // dynamically allocated buffer array
    extern __shared__ cufftDoubleComplex dev_localLapBuffer[];

    // compute global & local indices
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int localIdx = threadIdx.x;

    // load data to shared memory
    dev_localLapBuffer[localIdx] = dev_fftBuffer[globalIdx];

    // compute spectral coefficients and rescale
    // real parts and imaginary parts are processed separately
    if (blockIdx.x < dev_N / 2 + 1)
    {
        dev_localLapBuffer[localIdx].x = -dev_localLapBuffer[localIdx].x * (blockIdx.x * blockIdx.x + threadIdx.x * threadIdx.x) / dev_N / dev_N;
        dev_localLapBuffer[localIdx].y = -dev_localLapBuffer[localIdx].y * (blockIdx.x * blockIdx.x + threadIdx.x * threadIdx.x) / dev_N / dev_N;
    }
    else
    {
        dev_localLapBuffer[localIdx].x = -dev_localLapBuffer[localIdx].x * ((blockIdx.x - dev_N) * (blockIdx.x - dev_N) + threadIdx.x * threadIdx.x) / dev_N / dev_N;
        dev_localLapBuffer[localIdx].y = -dev_localLapBuffer[localIdx].y * ((blockIdx.x - dev_N) * (blockIdx.x - dev_N) + threadIdx.x * threadIdx.x) / dev_N / dev_N;
    }

    // load data to global memory
    dev_fftBuffer[globalIdx] = dev_localLapBuffer[localIdx];
}

/**rkUpdate(cufftDoubleReal *dev_rkBufferU, cufftDoubleReal *dev_curStepBufferU, cufftDoubleReal *dev_laplacianU,
 *                       cufftDoubleReal *dev_rkBufferV, cufftDoubleReal *dev_curStepBufferV, cufftDoubleReal *dev_laplacianV, int rkIter)
 * Compute the time update results for u and v with the fourth-order Runge-Kutta method.
 * The step is determined with rkIter.
 */
__global__ void rkUpdate(cufftDoubleReal *dev_rkBufferU, cufftDoubleReal *dev_curStepBufferU, cufftDoubleReal *dev_laplacianU,
                         cufftDoubleReal *dev_rkBufferV, cufftDoubleReal *dev_curStepBufferV, cufftDoubleReal *dev_laplacianV, int rkIter)
{
    // guard to prevent unspecified core calls
    if (blockIdx.x >= dev_N)
        return;

    // dynamically allocated buffer array
    // row 0: quarter-step runge-kutta buffer for u (output)
    // row 1: current full-step buffer for u
    // row 2: laplacian buffer for u
    // row 3, 4, 5: same arrangement but for v
    extern __shared__ cufftDoubleReal dev_localBuffer[];

    // compute global & local indices
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int localIdx = threadIdx.x;

    // load data to shared memory
    dev_localBuffer[localIdx] = dev_rkBufferU[globalIdx];
    dev_localBuffer[blockDim.x + localIdx] = dev_curStepBufferU[globalIdx];
    dev_localBuffer[2 * blockDim.x + localIdx] = dev_laplacianU[globalIdx];

    dev_localBuffer[3 * blockDim.x + localIdx] = dev_rkBufferV[globalIdx];
    dev_localBuffer[4 * blockDim.x + localIdx] = dev_curStepBufferV[globalIdx];
    dev_localBuffer[5 * blockDim.x + localIdx] = dev_laplacianV[globalIdx];

    // compute forcing terms
    cufftDoubleReal forcingU = dev_a + (dev_localBuffer[localIdx] * dev_localBuffer[localIdx]) / (dev_localBuffer[3 * blockDim.x + localIdx] * (1 + dev_epsilon * dev_localBuffer[localIdx] * dev_localBuffer[localIdx])) - dev_b * dev_localBuffer[localIdx];
    cufftDoubleReal forcingV = dev_localBuffer[localIdx] * dev_localBuffer[localIdx] - dev_c * dev_localBuffer[3 * blockDim.x + localIdx];

    // rk4 update
    dev_localBuffer[localIdx] = dev_localBuffer[blockDim.x + localIdx] + dev_dt / (4 - rkIter) * (dev_pi_multiple * dev_Du * dev_localBuffer[2 * blockDim.x + localIdx] + forcingU);
    dev_localBuffer[3 * blockDim.x + localIdx] = dev_localBuffer[4 * blockDim.x + localIdx] + dev_dt / (4 - rkIter) * (dev_pi_multiple * dev_Dv * dev_localBuffer[5 * blockDim.x + localIdx] + forcingV);

    __syncthreads();

    // load data to global memory
    dev_rkBufferU[globalIdx] = dev_localBuffer[localIdx];
    dev_rkBufferV[globalIdx] = dev_localBuffer[3 * blockDim.x + localIdx];

    // If rkIter = 3 (last stage of each complete step), the results will be
    // stored in both dev_rkBuffer and dev_curStepBuffer.
    dev_curStepBufferU[globalIdx] = rkIter == 3 ? dev_localBuffer[localIdx] : dev_curStepBufferU[globalIdx];
    dev_curStepBufferV[globalIdx] = rkIter == 3 ? dev_localBuffer[3 * blockDim.x + localIdx] : dev_curStepBufferV[globalIdx];
}

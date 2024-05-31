/**solver.h
 *
 * Header for solver functions used for solving
 * the Gierer-Meinhardt model.
 */

#ifndef SOLVER_H
#define SOLVER_H

#include <cufft.h>

/**laplacian(cufftDoubleComplex *dev_fftBuffer)
 * Compute the spectral coefficients for finding the laplacian.
 * Assumes memory is allocated on the device in advance.
 */
__global__ void getLaplacian(cufftDoubleComplex *dev_fftBuffer);

/**rkUpdate(cufftDoubleReal *dev_rkBufferU, cufftDoubleReal *dev_curStepBufferU, cufftDoubleReal *dev_laplacianU,
 *                       cufftDoubleReal *dev_rkBufferV, cufftDoubleReal *dev_curStepBufferV, cufftDoubleReal *dev_laplacianV, int rkIter)
 * Compute the time update results for u and v with the fourth-order Runge-Kutta method.
 * The step is determined with rkIter.
 */
__global__ void rkUpdate(cufftDoubleReal *dev_rkBufferU, cufftDoubleReal *dev_curStepBufferU, cufftDoubleReal *dev_laplacianU,
                         cufftDoubleReal *dev_rkBufferV, cufftDoubleReal *dev_curStepBufferV, cufftDoubleReal *dev_laplacianV, int rkIter);

#endif /*SOLVER_H*/
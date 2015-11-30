/*
 * GPU_functions.h
 *
 *  Created on: 8 avr. 2013
 *      Author: legal
 */

#ifndef GPU_TRANSPOSE_FUNCTIONS_H_
#define GPU_TRANSPOSE_FUNCTIONS_H_

// Includes
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8
extern __global__ void transposeDiagonal(float *odata, float *idata, int width, int height);

extern __global__ void transposeDiagonal_and_hard_decision(float *odata, float *idata, int width, int height);

extern __global__ void transposeDiagonal_and_hard_decision(unsigned int* odata, unsigned int* idata, int width, int height);

#endif /* GPU_FUNCTIONS_H_ */

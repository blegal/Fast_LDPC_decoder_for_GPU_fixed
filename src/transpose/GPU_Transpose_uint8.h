/*
 * GPU_functions.h
 *
 *  Created on: 8 avr. 2013
 *      Author: legal
 */

#ifndef GPU_TRANSPOSE_UINT8_FUNCTIONS_H_
#define GPU_TRANSPOSE_UINT8_FUNCTIONS_H_

// Includes
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>

extern __global__ void Interleaver_uint8(int *in, int* out, int taille_frame, int nb_frames);

extern __global__ void InvInterleaver_uint8(int *in, int* out, int taille_frame, int nb_frames);


#endif /* GPU_FUNCTIONS_H_ */

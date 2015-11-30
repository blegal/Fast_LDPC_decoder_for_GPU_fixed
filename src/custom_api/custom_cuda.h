/*
 * custom_cuda.h
 *
 *  Created on: 9 avr. 2013
 *      Author: legal
 */

#ifndef CUSTOM_CUDA_H_
#define CUSTOM_CUDA_H_

// Includes
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cstring>


#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>

using namespace std;

extern bool ERROR_CHECK(cudaError_t Status, string file, int line);

extern void CUDA_MALLOC_HOST(float** ptr, size_t nbElements, const char * file, int line);

extern void CUDA_MALLOC_HOST(int** ptr, size_t nbElements, const char * file, int line);

extern void CUDA_MALLOC_HOST(unsigned int** ptr, size_t nbElements, const char * file, int line);

extern void CUDA_MALLOC_HOST(char** ptr, size_t nbElements, const char * file, int line);

extern void CUDA_MALLOC_DEVICE(float** ptr, size_t nbElements, const char * file, int line);

extern void CUDA_MALLOC_DEVICE(int** ptr, size_t nbElements, const char * file, int line);

extern void CUDA_MALLOC_DEVICE(unsigned int** ptr, size_t nbElements, const char * file, int line);

extern void CUDA_MALLOC_DEVICE(char** ptr, size_t nbElements, const char * file, int line);

#endif /* CUSTOM_CUDA_H_ */

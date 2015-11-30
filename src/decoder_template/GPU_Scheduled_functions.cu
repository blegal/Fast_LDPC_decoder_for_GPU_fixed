/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 3
 * of the programming guide with some additions like error checking.
 *
 */

#include "../matrix/constantes_gpu.h"

#include "stdio.h"

// Device code
__global__ void LDPC_Init_Message_Array(float *MSG_C_2_V, int nb_total_msg) { // N : nombre reel de donnees

	int i = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	if (i < nb_total_msg)
		MSG_C_2_V[i] = (float) 0.0f;

}

__global__ void LDPC_TakeHardDecision_LLR_Array(int* hard_decision, const float *soft_decision, int N) {

#if _1D
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int o = _N * i;
	
	for (int j = 0; j<_N; j++) {
		hard_decision[j+o] = (soft_decision[j+o] > 0.0) ? 1 : 0;

	}
#endif

#if _2D
	int i = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	if (i < N) {
		hard_decision[i] = (soft_decision[i] > 0.0) ? 1 : 0;
	}
#endif
}


__global__ void LDPC_Convert_Float_LLR_to_8b_Fixed_Point(int *dst, const float *src, int N)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	if (i < N) {
		float value = 8.0f * src[i];
		value = fmaxf(value, -31.0f);
		value = fminf(value, +31.0f);
		int fp = (int) value;
		dst[i] = (fp << 24) | (fp << 16) | (fp << 8) | fp;
	}
}


__global__ void LDPC_TakeHardDecision_LLR_Array_SIMD(int *dst, const int *src, int N)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	if (i < N) {
	    union {char c[4]; unsigned int i;} value;
		value.i = src[i];
		value.c[0] = value.c[0] > 0 ? 1 : 0;
		value.c[1] = value.c[1] > 0 ? 1 : 0;
		value.c[2] = value.c[2] > 0 ? 1 : 0;
		value.c[3] = value.c[3] > 0 ? 1 : 0;
		dst[i]     = value.i;
	}
}
/*
__global__ void LDPC_Convert_Float_LLR_to_8b_Fixed_Point(int *dst, const float *src, int N)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	if (i < N) {
		int value = src[i];
		int v1 = ((value & 0x000000FF) << 24) >> 24;
		int v2 = ((value & 0x0000FF00) << 16) >> 24;
		int v3 = ((value & 0x00FF0000) <<  8) >> 24;
		int v4 = ((value & 0xFF000000)      ) >> 24;
		v1 = (v1 > 0);
		v2 = (v2 > 0);
		v3 = (v3 > 0);
		v4 = (v4 > 0);
		dst[i] = (v4 << 24) | (v3 << 16) | (v2 << 8) | v1;
	}
}
*/

//__global__ void LDPC_Interleave_LLR_Array(float* dst, float* src, int nb_total_var) {
//    const int i = blockDim.x * blockIdx.x + threadIdx.x;
//	if( i < nb_total_var ) {
//		dst[ i ] = src[ i ];
//	}
//}

__global__ void LDPC_Interleave_LLR_Array(float* dst, float* src, int nb_total_var, int nb_frames) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if( i < nb_total_var ) {
		dst[ nb_frames * (i%_N) + i/_N ] = src[ i ];
	}
}

//__global__ void LDPC_Deinterleave_LLR_Array(float* dst, float* src, int nb_nodes) {
//	#if _2D
//		const int i  = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
//		const int ii = blockDim.x * blockDim.y * gridDim.x; // A VERIFIER !!!!
//		if (i < nb_nodes) {
//			dst[ i % ii * _N + i/ii] = src[ i ];
//		}
//	#endif
//}

__global__ void LDPC_Deinterleave_LLR_Array(float* dst, float* src, int nb_total_var, int nb_frames) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if( i < nb_total_var ) {
		dst[ _N * (i%nb_frames) + i/nb_frames ] = src[ i ];
	}
}

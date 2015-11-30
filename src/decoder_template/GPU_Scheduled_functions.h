/*
 * GPU_functions.h
 *
 *  Created on: 8 avr. 2013
 *      Author: legal
 */

#ifndef GPU_FUNCTIONS_H_
#define GPU_FUNCTIONS_H_

// Includes
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cstring>

// includes, project
// includes, CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>

using namespace std;

extern __global__ void LDPC_Init_Message_Array(float *MSG_C_2_V, int nn);

extern __global__ void LDPC_Interleave_LLR_Array(float* dst, float* src, int nb_total_var);
extern __global__ void LDPC_Interleave_LLR_Array(float* dst, float* src, int nb_total_var, int nb_frames);
//extern __global__ void LDPC_Sched_Stage_0_interleaves(float var_nodes[]);

extern __global__ void LDPC_Sched_Stage_1_MS(
		float var_nodes[],
		float var_mesgs[],
		unsigned int PosNoeudsVariable[]);

extern __global__ void LDPC_Sched_Stage_1_MS_SIMD_old(
		unsigned int var_nodes[],
		unsigned int var_mesgs[],
		unsigned int PosNoeudsVariable[]);


extern __global__ void LDPC_Sched_Stage_1_MS_SIMD(
		unsigned int var_nodes[],
		unsigned int var_mesgs[],
		unsigned int PosNoeudsVariable[],
		unsigned int loops);

extern __global__ void LDPC_Sched_Stage_1_MS_SIMD_deg6_only(
		unsigned int var_nodes[],
		unsigned int var_mesgs[],
		unsigned int PosNoeudsVariable[],
		unsigned int loops);

extern __global__ void LDPC_Sched_Stage_1_OMS(
		float var_nodes[],
		float var_mesgs[],
		unsigned int PosNoeudsVariable[], float offset);
extern __global__ void LDPC_Sched_Stage_1_NMS(
		float var_nodes[],
		float var_mesgs[],
		unsigned int PosNoeudsVariable[], float norm);

extern __global__ void LDPC_TakeHardDecision_LLR_Array(int* hard_decision, const float *soft_decision, int nn);

//extern __global__ void LDPC_Sched_Stage_2_desinterleaves(float var_nodes[], int N);
extern __global__ void LDPC_Deinterleave_LLR_Array(float* dst, float* src, int nb_nodes);
extern __global__ void LDPC_Deinterleave_LLR_Array(float* dst, float* src, int nb_nodes, int nb_frames);

extern __global__ void LDPC_Convert_Float_LLR_to_8b_Fixed_Point(int *dst, const float *src, int N);
extern __global__ void LDPC_TakeHardDecision_LLR_Array_SIMD(int *dst, const int *src, int N);

#endif /* GPU_FUNCTIONS_H_ */

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

#include "../../matrix/constantes_gpu.h"
#include "simd_functions.h"

#include "stdio.h"

#define EARLY_TERM 0

//////////////////////////////////////////////////////////////////////////////////////////////////////
//
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////


#define THREAD_PER_BLOCK	128

__global__ void LDPC_Sched_Stage_1_MS_SIMD_Deg_6_Only(unsigned int var_nodes[_N],
		unsigned int var_mesgs[_M], unsigned int PosNoeudsVariable[_M],
   		unsigned int loops
		) {

	const int i  = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	const int ii = blockDim.x * blockDim.y * gridDim.x; // A VERIFIER !!!!

	__shared__ unsigned int tab_vContr [DEG_1][THREAD_PER_BLOCK];
	__shared__ unsigned int iTable     [DEG_1];
	///////////////////////////////////////////////////////////////////////////
	//
	//
	//
	{
		unsigned int *p_msg1w       = var_mesgs + i;
		unsigned int *p_indice_nod1 = PosNoeudsVariable;

#if EARLY_TERM == 1
		unsigned int ov_sign = 0x00000000;
#endif		
		//
		// ON UTILISE UNE PETITE ASTUCE AFIN D'ACCELERER LA SIMULATION DU DECODEUR
		//
		for (int z = 0; z < DEG_1_COMPUTATIONS; z++) {

			unsigned int sign_du_check = 0x00000000;
			unsigned int min1 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 1 A 127
			unsigned int min2 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 2 A 127

			//
			// ON PRECHARGE LES INDICES D'ENTRELACEMENT DES VN
			//
			__syncthreads();
			if( threadIdx.x < DEG_1){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			}
			__syncthreads();
			p_indice_nod1 += DEG_1;


			#pragma unroll 6
			for (int j = 0; j < 6; j++) {
				const unsigned int vAddr  = iTable[j]  * ii + i;
				const unsigned int vContr = var_nodes[vAddr];
				tab_vContr[j][threadIdx.x] = vContr;
				const unsigned int vAbs        = vabs4(vContr);
				min2 = vminu4(min2, vmaxu4(vAbs, min1));
				min1 = vminu4(min1, vAbs);
				sign_du_check = sign_du_check ^ vcmpgts4(vContr, 0x00000000);
			}

			const unsigned int cste_1 = vminu4(vsubus4(min2, 0x01010101), 0x1F1F1F1F);
			const unsigned int cste_2 = vminu4(vsubus4(min1, 0x01010101), 0x1F1F1F1F);

			#pragma unroll 6
			for (int j = 0; j < 6; j++) {
				const unsigned int vContr      = tab_vContr[j][threadIdx.x];
				const unsigned int ab          = vabs4  (vContr);
				const unsigned int m1          = vcmpeq4(ab, min1);
				const unsigned int m2          = vcmpne4(ab, min1);
				const unsigned int re          = (m1 & cste_1) | (m2 & cste_2);
				const unsigned int sign_msg    = sign_du_check ^ vcmpgts4(vContr, 0x00000000);
				const unsigned int msg_sortant = (re & sign_msg) | (vneg4(re) & (sign_msg ^ 0xFFFFFFFF));
				*p_msg1w = msg_sortant;
				p_msg1w += ii;
				var_nodes[iTable[j] * ii + i] = vaddss4(vContr, msg_sortant);
			}
#if EARLY_TERM == 1
			ov_sign = ov_sign | sign_du_check;
#endif
		}
	}
	loops -= 1;

	////////////////////////////////////////////////////////////////////////////
	//
	//
	//
	while (loops--) {
		unsigned int *p_msg1r = var_mesgs + i;
		unsigned int *p_msg1w = var_mesgs + i;
		unsigned int *p_indice_nod1 = PosNoeudsVariable;
#if EARLY_TERM == 1
		unsigned int  ov_sign = 0x00000000;
#endif			
		//
		// ON UTILISE UNE PETITE ASTUCE AFIN D'ACCELERER LA SIMULATION DU DECODEUR
		//
		for (int z = 0; z < DEG_1_COMPUTATIONS; z++) {
			unsigned int sign_du_check = 0x00000000;
			unsigned int min1 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 1 A 127
			unsigned int min2 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 2 A 127

			__syncthreads();
			if( threadIdx.x < DEG_1){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			}
			__syncthreads();
			p_indice_nod1 += DEG_1;

			#pragma unroll 6
			for (int j = 0; j < 6; j++) {
				const unsigned int vAddr    = iTable[j]  * ii + i;
				const unsigned int vContr   = vsubss4(var_nodes[vAddr], (*p_msg1r));
				tab_vContr[j][threadIdx.x] = vContr;
				const unsigned int vAbs  = vabs4(vContr);
				const unsigned int vMax  = vmaxu4(vAbs, min1);
				const unsigned int vSign = vcmpgts4(vContr, 0x00000000);
				p_msg1r += ii;
				min2 = vminu4(min2, vMax);
				min1 = vminu4(min1, vAbs);
				sign_du_check = sign_du_check ^ vSign;
			}

			const unsigned int cste_1 = vminu4(vsubus4(min2, 0x01010101), 0x1F1F1F1F);
			const unsigned int cste_2 = vminu4(vsubus4(min1, 0x01010101), 0x1F1F1F1F);

			#pragma unroll 6
			for (int j = 0; j < 6; j++) {
				const unsigned int vContr      = tab_vContr[j][threadIdx.x];
				const unsigned int ab          = vabs4(vContr);
				const unsigned int m1          = vcmpeq4(ab, min1);
				const unsigned int m2          = vcmpne4(ab, min1);
				const unsigned int re          = (m1 & cste_1) | (m2 & cste_2);
				const unsigned int sign_msg    = sign_du_check ^ vcmpgts4(vContr, 0x00000000);
				const unsigned int msg_sortant = (re & sign_msg) | (vneg4(re) & (sign_msg ^ 0xFFFFFFFF));
				*p_msg1w = msg_sortant;
				p_msg1w += ii;
				const unsigned int vAddr       = iTable[j]  * ii + i;
				var_nodes[vAddr]               = vaddss4(vContr, msg_sortant);
			}
#if EARLY_TERM == 1
			ov_sign = ov_sign | sign_du_check;
#endif			
		}

#if EARLY_TERM == 1
		if (ov_sign == (0x00000000))
			break;
#endif		
	}
	__syncthreads();
}



//////////////////////////////////////////////////////////////////////////////////////////////////////
//
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////



__global__ void LDPC_Sched_Stage_1_MS_SIMD_Deg_7_Only(unsigned int var_nodes[_N],
		unsigned int var_mesgs[_M], unsigned int PosNoeudsVariable[_M],
   		unsigned int loops
		) {

	const int i  = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	const int ii = blockDim.x * blockDim.y * gridDim.x; // A VERIFIER !!!!

	__shared__ unsigned int tab_vContr [THREAD_PER_BLOCK][DEG_1]; //[THREAD_PER_BLOCK * DEG_1];
	__shared__ unsigned int iTable     [DEG_1];
	///////////////////////////////////////////////////////////////////////////
	//
	//
	//
	{
		unsigned int *p_msg1w       = var_mesgs + i;
		unsigned int *p_indice_nod1 = PosNoeudsVariable;

#if EARLY_TERM == 1
		unsigned int ov_sign = 0x00000000;
#endif		
		//
		// ON UTILISE UNE PETITE ASTUCE AFIN D'ACCELERER LA SIMULATION DU DECODEUR
		//
		for (int z = 0; z < DEG_1_COMPUTATIONS; z++) {

			unsigned int sign_du_check = 0x00000000;
			unsigned int min1 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 1 A 127
			unsigned int min2 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 2 A 127

			//
			// ON PRECHARGE LES INDICES D'ENTRELACEMENT DES VN
			//
			__syncthreads();
			if( threadIdx.x < DEG_1){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			}
			__syncthreads();
			p_indice_nod1 += DEG_1;


			#pragma unroll 7
			for (int j = 0; j < 7; j++) {
				const unsigned int vAddr  = iTable[j]  * ii + i;
				const unsigned int vContr = var_nodes[vAddr];
				tab_vContr[threadIdx.x][j] = vContr;
				const unsigned int vAbs        = vabs4(vContr);
				min2 = vminu4(min2, vmaxu4(vAbs, min1));
				min1 = vminu4(min1, vAbs);
				sign_du_check = sign_du_check ^ vcmpgts4(vContr, 0x00000000);
			}

			const unsigned int cste_1 = vminu4(vsubus4(min2, 0x01010101), 0x1F1F1F1F);
			const unsigned int cste_2 = vminu4(vsubus4(min1, 0x01010101), 0x1F1F1F1F);

			#pragma unroll 7
			for (int j = 0; j < 7; j++) {
				const unsigned int vContr      = tab_vContr[threadIdx.x][j];
				const unsigned int ab          = vabs4  (vContr);
				const unsigned int m1          = vcmpeq4(ab, min1);
				const unsigned int m2          = vcmpne4(ab, min1);
				const unsigned int re          = (m1 & cste_1) | (m2 & cste_2);
				const unsigned int sign_msg    = sign_du_check ^ vcmpgts4(vContr, 0x00000000);
				const unsigned int msg_sortant = (re & sign_msg) | (vneg4(re) & (sign_msg ^ 0xFFFFFFFF));
				*p_msg1w = msg_sortant;
				p_msg1w += ii;
				var_nodes[iTable[j] * ii + i] = vaddss4(vContr, msg_sortant);
			}
#if EARLY_TERM == 1
			ov_sign = ov_sign | sign_du_check;
#endif			
		}
	}
	loops -= 1;

	////////////////////////////////////////////////////////////////////////////
	//
	//
	//
	while (loops--) {
		unsigned int *p_msg1r = var_mesgs + i;
		unsigned int *p_msg1w = var_mesgs + i;
		unsigned int *p_indice_nod1 = PosNoeudsVariable;
#if EARLY_TERM == 1
		unsigned int  ov_sign = 0x00000000;
#endif			
		//
		// ON UTILISE UNE PETITE ASTUCE AFIN D'ACCELERER LA SIMULATION DU DECODEUR
		//
		for (int z = 0; z < DEG_1_COMPUTATIONS; z++) {
			unsigned int sign_du_check = 0x00000000;
			unsigned int min1 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 1 A 127
			unsigned int min2 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 2 A 127

			__syncthreads();
			if( threadIdx.x < DEG_1){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			}
			__syncthreads();
			p_indice_nod1 += DEG_1;

			#pragma unroll 7
			for (int j = 0; j < 7; j++) {
				const unsigned int vAddr    = iTable[j]  * ii + i;
				const unsigned int vContr   = vsubss4(var_nodes[vAddr], (*p_msg1r));
				tab_vContr[threadIdx.x][j] = vContr;
				const unsigned int vAbs  = vabs4(vContr);
				const unsigned int vMax  = vmaxu4(vAbs, min1);
				const unsigned int vSign = vcmpgts4(vContr, 0x00000000);
				p_msg1r += ii;
				min2 = vminu4(min2, vMax);
				min1 = vminu4(min1, vAbs);
				sign_du_check = sign_du_check ^ vSign;
			}

			const unsigned int cste_1 = vminu4(vsubus4(min2, 0x01010101), 0x1F1F1F1F);
			const unsigned int cste_2 = vminu4(vsubus4(min1, 0x01010101), 0x1F1F1F1F);

			#pragma unroll 7
			for (int j = 0; j < 7; j++) {
				const unsigned int vContr      = tab_vContr[threadIdx.x][j];
				const unsigned int ab          = vabs4(vContr);
				const unsigned int m1          = vcmpeq4(ab, min1);
				const unsigned int m2          = vcmpne4(ab, min1);
				const unsigned int re          = (m1 & cste_1) | (m2 & cste_2);
				const unsigned int sign_msg    = sign_du_check ^ vcmpgts4(vContr, 0x00000000);
				const unsigned int msg_sortant = (re & sign_msg) | (vneg4(re) & (sign_msg ^ 0xFFFFFFFF));
				*p_msg1w = msg_sortant;
				p_msg1w += ii;
				const unsigned int vAddr       = iTable[j]  * ii + i;
				var_nodes[vAddr]               = vaddss4(vContr, msg_sortant);
			}
#if EARLY_TERM == 1
			ov_sign = ov_sign | sign_du_check;
#endif			
		}

#if EARLY_TERM == 1
		if (ov_sign == (0x00000000))
			break;
#endif
	}
	__syncthreads();
}



//////////////////////////////////////////////////////////////////////////////////////////////////////
//
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void LDPC_Sched_Stage_1_MS_SIMD_Two_CN_Degs(
		unsigned int var_nodes[_N],
		unsigned int var_mesgs[_M],
		unsigned int PosNoeudsVariable[_M],
   		unsigned int loops
	) {

	const int i  = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	const int ii = blockDim.x * blockDim.y * gridDim.x; // A VERIFIER !!!!

	__shared__ unsigned int tab_vContr [THREAD_PER_BLOCK * DEG_1];
	__shared__ unsigned int iTable     [DEG_1];

	///////////////////////////////////////////////////////////////////////////
	//
	//
	//
	{
		unsigned int *p_msg1w       = var_mesgs + i;
		unsigned int *p_indice_nod1 = PosNoeudsVariable;

		//
		// CALCULS RELATIFS AU CN DE DEGREE 1
		//
		for (int z = 0; z < DEG_1_COMPUTATIONS; z++) {
			unsigned int sign_du_check = 0x00000000;
			unsigned int min1 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 1 A 127
			unsigned int min2 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 2 A 127

			__syncthreads();
			if( threadIdx.x < DEG_1){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			} __syncthreads();
			p_indice_nod1 += DEG_1;

			#pragma unroll
			for (int j = 0; j < DEG_1; j++) {
				const unsigned int vAddr  = iTable[j]  * ii + i;
				const unsigned int vContr = var_nodes[vAddr];
				tab_vContr[threadIdx.x + THREAD_PER_BLOCK * j] = vContr;
				const unsigned int vAbs        = vabs4(vContr);
				min2 = vminu4(min2, vmaxu4(vAbs, min1));
				min1 = vminu4(min1, vAbs);
				sign_du_check = sign_du_check ^ vcmpgts4(vContr, 0x00000000);
			}

			const unsigned int cste_1 = vminu4(vsubus4(min2, 0x01010101), 0x1F1F1F1F);
			const unsigned int cste_2 = vminu4(vsubus4(min1, 0x01010101), 0x1F1F1F1F);

			#pragma unroll
			for (int j = 0; j < DEG_1; j++) {
				const unsigned int vContr      = tab_vContr[threadIdx.x + THREAD_PER_BLOCK * j];
				const unsigned int ab          = vabs4  (vContr);
				const unsigned int m1          = vcmpeq4(ab, min1);
				const unsigned int m2          = vcmpne4(ab, min1);
				const unsigned int re          = (m1 & cste_1) | (m2 & cste_2);
				const unsigned int sign_msg    = sign_du_check ^ vcmpgts4(vContr, 0x00000000);
				const unsigned int msg_sortant = (re & sign_msg) | (vneg4(re) & (sign_msg ^ 0xFFFFFFFF));
				*p_msg1w = msg_sortant;
				p_msg1w += ii;
				var_nodes[iTable[j] * ii + i] = vaddss4(vContr, msg_sortant);
			}
		}

#if NB_DEGRES > 1
		//
		// CALCULS RELATIFS AU CN DE DEGREE 2
		//
		for (int z = 0; z < DEG_2_COMPUTATIONS; z++) {
			unsigned int sign_du_check = 0x00000000;
			unsigned int min1 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 1 A 127
			unsigned int min2 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 2 A 127

			__syncthreads();
			if( threadIdx.x < DEG_2){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			} __syncthreads();
			p_indice_nod1 += DEG_2;

			#pragma unroll
			for (int j = 0; j < DEG_2; j++) {
				const unsigned int vAddr  = iTable[j]  * ii + i;
				const unsigned int vContr = var_nodes[vAddr];
				tab_vContr[threadIdx.x + THREAD_PER_BLOCK * j] = vContr;
				const unsigned int vAbs        = vabs4(vContr);
				min2 = vminu4(min2, vmaxu4(vAbs, min1));
				min1 = vminu4(min1, vAbs);
				sign_du_check = sign_du_check ^ vcmpgts4(vContr, 0x00000000);
			}

			const unsigned int cste_1 = vminu4(vsubus4(min2, 0x01010101), 0x1F1F1F1F);
			const unsigned int cste_2 = vminu4(vsubus4(min1, 0x01010101), 0x1F1F1F1F);

			#pragma unroll
			for (int j = 0; j < DEG_2; j++) {
				const unsigned int vContr      = tab_vContr[threadIdx.x + THREAD_PER_BLOCK * j];
				const unsigned int ab          = vabs4  (vContr);
				const unsigned int m1          = vcmpeq4(ab, min1);
				const unsigned int m2          = vcmpne4(ab, min1);
				const unsigned int re          = (m1 & cste_1) | (m2 & cste_2);
				const unsigned int sign_msg    = sign_du_check ^ vcmpgts4(vContr, 0x00000000);
				const unsigned int msg_sortant = (re & sign_msg) | (vneg4(re) & (sign_msg ^ 0xFFFFFFFF));
				*p_msg1w = msg_sortant;
				p_msg1w += ii;
				var_nodes[iTable[j] * ii + i] = vaddss4(vContr, msg_sortant);
			}
		}
#endif		
	}
	loops -= 1;


	////////////////////////////////////////////////////////////////////////////
	//
	//
	//
	while (loops--) {
		unsigned int *p_msg1r = var_mesgs + i;
		unsigned int *p_msg1w = var_mesgs + i;
		unsigned int *p_indice_nod1 = PosNoeudsVariable;
#if EARLY_TERM == 1
		unsigned int  ov_sign = 0x00000000;
#endif			

		//
		// CALCULS RELATIFS AU CN DE DEGREE 1
		//
		for (int z = 0; z < DEG_1_COMPUTATIONS; z++) {
			unsigned int sign_du_check = 0x00000000;
			unsigned int min1 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 1 A 127
			unsigned int min2 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 2 A 127

			__syncthreads();
			if( threadIdx.x < DEG_1){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			} __syncthreads();
			p_indice_nod1 += DEG_1;

			#pragma unroll
			for (int j = 0; j < DEG_1; j++) {
				const unsigned int vAddr    = iTable[j]  * ii + i;
				const unsigned int vContr   = vsubss4(var_nodes[vAddr], (*p_msg1r));
				tab_vContr[threadIdx.x + THREAD_PER_BLOCK * j] = vContr;
				const unsigned int vAbs  = vabs4(vContr);
				const unsigned int vMax  = vmaxu4(vAbs, min1);
				const unsigned int vSign = vcmpgts4(vContr, 0x00000000);
				p_msg1r += ii;
				min2 = vminu4(min2, vMax);
				min1 = vminu4(min1, vAbs);
				sign_du_check = sign_du_check ^ vSign;
			}

			const unsigned int cste_1 = vminu4(vsubus4(min2, 0x01010101), 0x1F1F1F1F);
			const unsigned int cste_2 = vminu4(vsubus4(min1, 0x01010101), 0x1F1F1F1F);

			#pragma unroll
			for (int j = 0; j < DEG_1; j++) {
				const unsigned int vContr      = tab_vContr[threadIdx.x + THREAD_PER_BLOCK * j];
				const unsigned int ab          = vabs4(vContr);
				const unsigned int m1          = vcmpeq4(ab, min1);
				const unsigned int m2          = vcmpne4(ab, min1);
				const unsigned int re          = (m1 & cste_1) | (m2 & cste_2);
				const unsigned int sign_msg    = sign_du_check ^ vcmpgts4(vContr, 0x00000000);
				const unsigned int msg_sortant = (re & sign_msg) | (vneg4(re) & (sign_msg ^ 0xFFFFFFFF));
				*p_msg1w = msg_sortant;
				p_msg1w += ii;
				const unsigned int vAddr       = iTable[j]  * ii + i;
				var_nodes[vAddr]               = vaddss4(vContr, msg_sortant);
			}
#if EARLY_TERM == 1
			ov_sign = ov_sign | sign_du_check;
#endif			
		}

#if NB_DEGRES > 1
		//
		// CALCULS RELATIFS AU CN DE DEGREE 2
		//
		for (int z = 0; z < DEG_2_COMPUTATIONS; z++) {
			unsigned int sign_du_check = 0x00000000;
			unsigned int min1 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 1 A 127
			unsigned int min2 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 2 A 127

			__syncthreads();
			if( threadIdx.x < DEG_2){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			} __syncthreads();
			p_indice_nod1 += DEG_2;

			#pragma unroll
			for (int j = 0; j < DEG_2; j++) {
				const unsigned int vAddr    = iTable[j]  * ii + i;
				const unsigned int vContr   = vsubss4(var_nodes[vAddr], (*p_msg1r));
				tab_vContr[threadIdx.x + THREAD_PER_BLOCK * j] = vContr;
				const unsigned int vAbs  = vabs4(vContr);
				const unsigned int vMax  = vmaxu4(vAbs, min1);
				const unsigned int vSign = vcmpgts4(vContr, 0x00000000);
				p_msg1r += ii;
				min2 = vminu4(min2, vMax);
				min1 = vminu4(min1, vAbs);
				sign_du_check = sign_du_check ^ vSign;
			}

			const unsigned int cste_1 = vminu4(vsubus4(min2, 0x01010101), 0x1F1F1F1F);
			const unsigned int cste_2 = vminu4(vsubus4(min1, 0x01010101), 0x1F1F1F1F);

			#pragma unroll
			for (int j = 0; j < DEG_2; j++) {
				const unsigned int vContr      = tab_vContr[threadIdx.x + THREAD_PER_BLOCK * j];
				const unsigned int ab          = vabs4(vContr);
				const unsigned int m1          = vcmpeq4(ab, min1);
				const unsigned int m2          = vcmpne4(ab, min1);
				const unsigned int re          = (m1 & cste_1) | (m2 & cste_2);
				const unsigned int sign_msg    = sign_du_check ^ vcmpgts4(vContr, 0x00000000);
				const unsigned int msg_sortant = (re & sign_msg) | (vneg4(re) & (sign_msg ^ 0xFFFFFFFF));
				*p_msg1w = msg_sortant;
				p_msg1w += ii;
				const unsigned int vAddr       = iTable[j]  * ii + i;
				var_nodes[vAddr]               = vaddss4(vContr, msg_sortant);
			}
#if EARLY_TERM == 1
			ov_sign = ov_sign | sign_du_check;
#endif			
		}
#endif

#if EARLY_TERM == 1
		if (ov_sign == (0x00000000))
			break;
#endif		
	}
	__syncthreads();
}






__global__ void LDPC_Sched_Stage_1_MS_SIMD_deg6_only(
		unsigned int var_nodes[_N],
		unsigned int var_mesgs[_M],
		unsigned int PosNoeudsVariable[_M],
   		unsigned int loops
) {

	const int i  = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	const int ii =                            blockDim.x              * blockDim.y * gridDim.x; // A VERIFIER !!!!


	//////////////////////////////////////////////////////////////////////////////////////////////////
	//
	// VERSION ALLEGEE DU DECODAGE POUR LA PREMIERE ITERATION
	//
	{
		unsigned int *p_msg1w = var_mesgs + i; // POINTEUR MESG_C_2_V (pour l'�criture)
		unsigned int *p_indice_nod1 = PosNoeudsVariable;

		for (int z = 0; z < DEG_1_COMPUTATIONS; z++) {

			unsigned int sign_du_check = 0x00000000;
			unsigned int min1 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 1 A 127
			unsigned int min2 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 2 A 127

			unsigned int tab_vContr[DEG_1];
			unsigned int iTable[DEG_1];

		    uint2* p = reinterpret_cast<uint2*>(p_indice_nod1);
		    uint2  dA = p[0];
		    uint2  dB = p[1];
		    uint2  dC = p[2];
			p_indice_nod1 += 6;

			iTable[0]     = dA.x * ii + i;
			iTable[1]     = dA.y * ii + i;
			iTable[2]     = dB.x * ii + i;
			iTable[3]     = dB.y * ii + i;
			iTable[4]     = dC.x * ii + i;
			iTable[5]     = dC.y * ii + i;

			#pragma unroll 6
			for (int j = 0; j < DEG_1; j++) {
//				iTable[j] = (*p_indice_nod1++) * ii + i; // Ieme INDEX (NODE INDICE)
				tab_vContr[j] = var_nodes[ iTable[j] ]; // CALCUL DE LA Ieme CONTRIBUTION
				min2          = vminu4(min2, vmaxu4(vabs4(tab_vContr[j]), min1));
				min1          = vminu4(min1, vabs4(tab_vContr[j]));
				sign_du_check = sign_du_check ^ vcmpgts4(tab_vContr[j], 0x00000000);
			}

			const unsigned int cste_1 = vminu4(vsubus4(min2, 0x01010101),
					0x1F1F1F1F);
			const unsigned int cste_2 = vminu4(vsubus4(min1, 0x01010101),
					0x1F1F1F1F);

#pragma unroll 6
			for (int j = 0; j < DEG_1; j++) {

				unsigned int ab = vabs4(tab_vContr[j]);
				unsigned int m1 = vcmpeq4(ab, min1);
				unsigned int m2 = vcmpne4(ab, min1);
				unsigned int re = (m1 & cste_1) | (m2 & cste_2);

				unsigned int sign_msg = sign_du_check
						^ vcmpgts4(tab_vContr[j], 0x00000000);

				unsigned int msg_sortant = (re & sign_msg)
						| (vneg4(re) & (sign_msg ^ 0xFFFFFFFF));
				*p_msg1w = msg_sortant;
				p_msg1w += ii;
				var_nodes[iTable[j]] = vaddss4(tab_vContr[j], msg_sortant);
			}
		}
	}loops--;


	//////////////////////////////////////////////////////////////////////////////////////////////////
	//
	// ON START LE DECODAGE "COMPLET"
	//
	while (loops--) {
		unsigned int *p_msg1r = var_mesgs + i; // POINTEUR MESG_C_2_V (pour la lecture, z-1)
		unsigned int *p_msg1w = var_mesgs + i; // POINTEUR MESG_C_2_V (pour l'�criture)
		unsigned int *p_indice_nod1 = PosNoeudsVariable;

		unsigned int ov_sign = 0x00000000;
		//
		// ON UTILISE UNE PETITE ASTUCE AFIN D'ACCELERER LA SIMULATION DU DECODEUR
		//
		for (int z = 0; z < DEG_1_COMPUTATIONS; z++) {

			unsigned int sign_du_check = 0x00000000;
			unsigned int min1 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 1 A 127
			unsigned int min2 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 2 A 127

			unsigned int tab_vContr[DEG_1];
			unsigned int iTable[DEG_1];

//		    uint2* p = reinterpret_cast<uint2*>(p_indice_nod1);
//		    uint2  dA = p[0]; iTable[0] = dA.x * ii + i; iTable[1] = dA.y * ii + i;
//		    uint2  dB = p[1]; iTable[2] = dB.x * ii + i; iTable[3] = dB.y * ii + i;
//		    uint2  dC = p[2]; iTable[4] = dC.x * ii + i; iTable[5] = dC.y * ii + i;
//			p_indice_nod1 += 6;

#pragma unroll 6
			for (int j = 0; j < DEG_1; j++) {
				iTable[j] = (*p_indice_nod1++) * ii + i; // Ieme INDEX (NODE INDICE)
				tab_vContr[j] = vsubss4(var_nodes[iTable[j]], (*p_msg1r)); // CALCUL DE LA Ieme CONTRIBUTION

				p_msg1r += ii;

				min2 = vminu4(min2, vmaxu4(vabs4(tab_vContr[j]), min1));
				min1 = vminu4(min1, vabs4(tab_vContr[j]));

				sign_du_check = sign_du_check
						^ vcmpgts4(tab_vContr[j], 0x00000000);
			}

			const unsigned int cste_1 = vminu4(vsubus4(min2, 0x01010101),
					0x1F1F1F1F);
			const unsigned int cste_2 = vminu4(vsubus4(min1, 0x01010101),
					0x1F1F1F1F);

#pragma unroll 6
			for (int j = 0; j < DEG_1; j++) {

				unsigned int ab = vabs4(tab_vContr[j]);
				unsigned int m1 = vcmpeq4(ab, min1);
				unsigned int m2 = vcmpne4(ab, min1);
				unsigned int re = (m1 & cste_1) | (m2 & cste_2);

				unsigned int sign_msg = sign_du_check
						^ vcmpgts4(tab_vContr[j], 0x00000000);
				unsigned int msg_sortant = (re & sign_msg)
						| (vneg4(re) & (sign_msg ^ 0xFFFFFFFF));

				*p_msg1w = msg_sortant;

				p_msg1w += ii;
				var_nodes[iTable[j]] = vaddss4(tab_vContr[j], msg_sortant);
			}

			ov_sign = ov_sign | sign_du_check;
		}

		if (ov_sign == (0x00000000)) {
			break;
		}
	}

	__syncthreads();
}



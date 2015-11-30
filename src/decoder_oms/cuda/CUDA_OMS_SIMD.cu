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

union t_1x4
{
	unsigned int v;
    char c[4];
};

__global__ void LDPC_Sched_Stage_1_OMS_SIMD(unsigned int var_nodes[_N],
		unsigned int var_mesgs[_M], unsigned int PosNoeudsVariable[_M],
   		unsigned int loops
		) {

	const int i  = threadIdx.x + blockIdx.x * blockDim.x
			                   + blockIdx.y * blockDim.x * gridDim.x;
	const int ii = blockDim.x  * blockDim.y * gridDim.x; // A VERIFIER !!!!

	__shared__ unsigned int iTable[DEG_1];

	///////////////////////////////////////////////////////////////////////////
	//
	//
	//
	{
		unsigned int *p_msg1w       = var_mesgs + i;
		unsigned int *p_indice_nod1 = PosNoeudsVariable;

		//
		// ON UTILISE UNE PETITE ASTUCE AFIN D'ACCELERER LA SIMULATION DU DECODEUR
		//
		for (int z = 0; z < DEG_1_COMPUTATIONS; z++) {

			unsigned int sign_du_check = 0x00000000;
			unsigned int min1 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 1 A 127
			unsigned int min2 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 2 A 127

			register unsigned int tab_vContr[DEG_1];

			//
			// ON PRECHARGE LES INDICES D'ENTRELACEMENT DES VN
			//
			__syncthreads();
			if( threadIdx.x < DEG_1){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			}
			__syncthreads();
			p_indice_nod1 += DEG_1;

			#pragma unroll
			for (int j = 0; j < DEG_1; j++) {
				tab_vContr[j] = var_nodes[ iTable[j] * ii + i ];
				min2 = vminu4(min2, vmaxu4(vabs4(tab_vContr[j]), min1));
				min1 = vminu4(min1, vabs4(tab_vContr[j]));
				sign_du_check = sign_du_check ^ vcmpgts4(tab_vContr[j], 0x00000000);
			}

			const unsigned int cste_1 = vminu4(vsubus4(min2, 0x01010101), 0x1F1F1F1F);
			const unsigned int cste_2 = vminu4(vsubus4(min1, 0x01010101), 0x1F1F1F1F);

			#pragma unroll
			for (int j = 0; j < DEG_1; j++) {
				unsigned int ab = vabs4(tab_vContr[j]);
				unsigned int m1 = vcmpeq4( ab, min1 );
				unsigned int m2 = vcmpne4( ab, min1 );
				unsigned int re = (m1 & cste_1) | (m2 & cste_2);
				unsigned int sign_msg = sign_du_check ^ vcmpgts4(tab_vContr[j], 0x00000000);
				unsigned int msg_sortant = (re & sign_msg) | (vneg4(re) & (sign_msg ^ 0xFFFFFFFF));
				*p_msg1w = msg_sortant;
				p_msg1w += ii;
				var_nodes[ iTable[j] * ii + i ] = vaddss4(tab_vContr[j], msg_sortant);
			}
		}

#if NB_DEGRES > 1
		for (int z = 0; z <DEG_2_COMPUTATIONS; z++) {
			unsigned int sign_du_check = 0x00000000;
			unsigned int min1 = 0x7F7F7F7F;
			unsigned int min2 = 0x7F7F7F7F;
			unsigned int tab_vContr [DEG_2];

			//
			// ON PRECHARGE LES INDICES D'ENTRELACEMENT DES VN
			//
			__syncthreads();
			if( threadIdx.x < DEG_2){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			}
			__syncthreads();
			p_indice_nod1 += DEG_2;

			#pragma unroll
			for (int j = 0; j<DEG_2; j++) {
				tab_vContr[j] = var_nodes[ iTable[j] * ii + i ];
				min2 = vminu4(min2, vmaxu4(vabs4(tab_vContr[j]), min1));
				min1 = vminu4(min1, vabs4(tab_vContr[j]));
				sign_du_check = sign_du_check ^ vcmpgts4(tab_vContr[j], 0x00000000);
			}

			const unsigned int cste_1 = vsubus4(min2, 0x01010101);
			const unsigned int cste_2 = vsubus4(min1, 0x01010101);

			#pragma unroll
			for (int j = 0; j < DEG_2; j++) {
				unsigned int ab = vabs4(tab_vContr[j]);
				unsigned int m1 = vcmpeq4( ab, min1 );
				unsigned int m2 = vcmpne4( ab, min1 );
				unsigned int re = (m1 & cste_1) | (m2 & cste_2);
				unsigned int sign_msg = sign_du_check ^ vcmpgts4(tab_vContr[j], 0x00000000);
				unsigned int msg_sortant = (re & sign_msg) | (vneg4(re) & (sign_msg ^ 0xFFFFFFFF));
				*p_msg1w = msg_sortant;
				p_msg1w += ii;
				var_nodes[ iTable[j] * ii + i ] = vaddss4(tab_vContr[j], msg_sortant);
			}
		}
#endif
	}loops -= 1;

	////////////////////////////////////////////////////////////////////////////
	//
	//
	//
	while (loops--) {
		unsigned int *p_msg1r = var_mesgs + i;
		unsigned int *p_msg1w = var_mesgs + i;
		unsigned int *p_indice_nod1 = PosNoeudsVariable;
#if EARLY_TERM == 1
		register unsigned int  ov_sign = 0x00000000;
#endif			
		//
		// ON UTILISE UNE PETITE ASTUCE AFIN D'ACCELERER LA SIMULATION DU DECODEUR
		//
		for (int z = 0; z < DEG_1_COMPUTATIONS; z++) {
			unsigned int sign_du_check = 0x00000000;
			unsigned int min1 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 1 A 127
			unsigned int min2 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 2 A 127
			unsigned int tab_vContr[DEG_1];

			//
			// ON PRECHARGE LES INDICES D'ENTRELACEMENT DES VN
			//
			__syncthreads();
			if( threadIdx.x < DEG_1){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			}
			__syncthreads();
			p_indice_nod1 += DEG_1;

			#pragma unroll
			for (int j = 0; j < DEG_1; j++) {
				tab_vContr[j] = vsubss4(var_nodes[iTable[j] * ii + i], (*p_msg1r)); // CALCUL DE LA Ieme CONTRIBUTION
				p_msg1r += ii;
				min2 = vminu4(min2, vmaxu4(vabs4(tab_vContr[j]), min1));
				min1 = vminu4(min1, vabs4(tab_vContr[j]));
				sign_du_check = sign_du_check ^ vcmpgts4(tab_vContr[j], 0x00000000);
			}

			const unsigned int cste_1 = vminu4(vsubus4(min2, 0x01010101), 0x1F1F1F1F);
			const unsigned int cste_2 = vminu4(vsubus4(min1, 0x01010101), 0x1F1F1F1F);

			#pragma unroll
			for (int j = 0; j < DEG_1; j++) {
				unsigned int ab          = vabs4(tab_vContr[j]);
				unsigned int m1          = vcmpeq4(ab, min1);
				unsigned int m2          = vcmpne4(ab, min1);
				unsigned int re          = (m1 & cste_1) | (m2 & cste_2);
				unsigned int sign_msg    = sign_du_check ^ vcmpgts4(tab_vContr[j], 0x00000000);
				unsigned int msg_sortant = (re & sign_msg) | (vneg4(re) & (sign_msg ^ 0xFFFFFFFF));
				*p_msg1w = msg_sortant;
				p_msg1w += ii;
				var_nodes[iTable[j] * ii + i] = vaddss4(tab_vContr[j], msg_sortant);
			}
#if EARLY_TERM == 1
			ov_sign = ov_sign | sign_du_check;
#endif			
		}

#if NB_DEGRES > 1
		for (int z = 0; z <DEG_2_COMPUTATIONS; z++) {

			unsigned int sign_du_check = 0x00000000;
			unsigned int min1 = 0x7F7F7F7F;
			unsigned int min2 = 0x7F7F7F7F;
			unsigned int tab_vContr [DEG_2];

			//
			// ON PRECHARGE LES INDICES D'ENTRELACEMENT DES VN
			//
			__syncthreads();
			if( threadIdx.x < DEG_2){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			}
			__syncthreads();
			p_indice_nod1 += DEG_2;

			#pragma unroll
			for (int j = 0; j<DEG_2; j++) {
				tab_vContr[j] = vsubss4(var_nodes[iTable[j] * ii + i], (*p_msg1r));// CALCUL DE LA Ieme CONTRIBUTION
				p_msg1r += ii;
				min2 = vminu4(min2, vmaxu4(vabs4(tab_vContr[j]), min1));
				min1 = vminu4(min1, vabs4(tab_vContr[j]));
				sign_du_check = sign_du_check ^ vcmpgts4(tab_vContr[j], 0x00000000);
			}

			const unsigned int cste_1 = vminu4(vsubus4(min2, 0x01010101), 0x1F1F1F1F);
			const unsigned int cste_2 = vminu4(vsubus4(min1, 0x01010101), 0x1F1F1F1F);

			#pragma unroll
			for (int j = 0; j < DEG_2; j++) {

				unsigned int ab = vabs4(tab_vContr[j]);
				unsigned int m1 = vcmpeq4( ab, min1 );
				unsigned int m2 = vcmpne4( ab, min1 );
				unsigned int re = (m1 & cste_1) | (m2 & cste_2);
				unsigned int sign_msg = sign_du_check ^ vcmpgts4(tab_vContr[j], 0x00000000);
				unsigned int msg_sortant = (re & sign_msg) | (vneg4(re) & (sign_msg ^ 0xFFFFFFFF));
				*p_msg1w = msg_sortant;
				p_msg1w += ii;
				var_nodes[ iTable[j] * ii + i ] = vaddss4(tab_vContr[j], msg_sortant);
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


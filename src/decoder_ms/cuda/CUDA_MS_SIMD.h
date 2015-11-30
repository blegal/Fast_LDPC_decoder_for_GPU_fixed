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

extern __global__ void LDPC_Sched_Stage_1_MS_SIMD(
		unsigned int var_nodes[],
		unsigned int var_mesgs[],
		unsigned int PosNoeudsVariable[],
		unsigned int loops
	);

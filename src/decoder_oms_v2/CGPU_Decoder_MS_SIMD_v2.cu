/*
 *  ldcp_decoder.h
 *  ldpc3
 *
 *  Created by legal on 02/04/11.
 *  Copyright 2011 ENSEIRB. All rights reserved.
 *
 */

/*----------------------------------------------------------------------------*/

#include "CGPU_Decoder_MS_SIMD_v2.h"
#include "../transpose/GPU_Transpose.h"
#include "../transpose/GPU_Transpose_uint8.h"
#include "../tools/debug_fx.h"
#include "./cuda/CUDA_OMS_SIMD_v2.h"

static const size_t BLOCK_SIZE = 128; // 96 for exp.

CGPU_Decoder_MS_SIMD_v2::CGPU_Decoder_MS_SIMD_v2(size_t _nb_frames, size_t n, size_t k, size_t m):
CGPUDecoder(_nb_frames, n, k, m)
{
	size_t nb_blocks = nb_frames / BLOCK_SIZE;
	printf("(II) Decoder configuration: BLOCK_SIZE = %ld, nb_frames = %ld, nb_blocks = %ld\n", BLOCK_SIZE, nb_frames, nb_blocks);


	struct cudaDeviceProp devProp;
  	cudaGetDeviceProperties(&devProp, 0);
  	printf("(II) Identifiant du GPU (CUDA)   : %s\n", devProp.name);
  	printf("(II) Nombre de Multi-Processor   : %d\n", devProp.multiProcessorCount);
  	printf("(II) Taille de memoire globale   : %ld\n", devProp.totalGlobalMem);
  	printf("(II) Taille de sharedMemPerBlock : %ld\n", devProp.sharedMemPerBlock);

/*  	
        int regsPerBlock;
        int warpSize;
        size_t memPitch;
        int maxThreadsPerBlock;
        int clockRate;
        size_t totalConstMem;
        int major;
        int minor;
        int memoryClockRate;
        int memoryBusWidth;
*/
  	struct cudaFuncAttributes attr;    
	cudaFuncGetAttributes(&attr, LDPC_Sched_Stage_1_MS_SIMD_Two_CN_Degs); 

  	int nMP      = devProp.multiProcessorCount; // NOMBRE DE STREAM PROCESSOR
  	int nWarp    = attr.maxThreadsPerBlock/32;  // PACKET DE THREADs EXECUTABLES EN PARALLELE
  	int nThreads = nWarp * 32;					// NOMBRE DE THREAD MAXI PAR SP
  	int nDOF     = nb_frames;
  	int nBperMP  = 65536 / (attr.numRegs); 	// Nr of blocks per MP
  	int minB     = min(nBperMP*nThreads,1024);
  	int nBlocks  = max(minB/nThreads * nMP, nDOF/nThreads);  //Total number of blocks
  	printf("(II) Nombre de Warp    : %d\n", nWarp);
  	printf("(II) Nombre de Threads           : %d\n", nThreads);

#if (NB_DEGRES != 1)
  	printf("(II) LDPC_Sched_Stage_1_MS_SIMD_Two_CN_Degs :\n");
  	struct cudaFuncAttributes attr_fx2;    
	cudaFuncGetAttributes(&attr_fx2, LDPC_Sched_Stage_1_MS_SIMD_Two_CN_Degs); 
  	printf("(II) LDPC_Sched_Stage_1_MS_SIMD :\n");
  	printf("(II) - Nombre de regist/thr : %d\n", attr.numRegs);
  	printf("(II) - Nombre de local/thr  : %ld\n", attr.localSizeBytes);
    printf("(II) - Nombre de shared/thr : %ld\n", attr.sharedSizeBytes);
    printf("(II) - Nombre de pBLOCKs    : %f\n", (float)nb_frames / (float)BLOCK_SIZE);
    printf("(II) - Nombre de pBLOCKs/uP : %f\n", (float)nb_frames / (float)BLOCK_SIZE / (float)devProp.multiProcessorCount);

#elif (DEG_1 == 6)
  	printf("(II) LDPC_Sched_Stage_1_MS_SIMD_Deg_6_Only :\n");
  	struct cudaFuncAttributes attr_fx2;    
	cudaFuncGetAttributes(&attr_fx2, LDPC_Sched_Stage_1_MS_SIMD_Deg_6_Only); 
  	printf("(II) - Nombre de regist/thr : %d\n", attr_fx2.numRegs);
  	printf("(II) - Nombre de local/thr  : %d\n", attr_fx2.localSizeBytes);
  	printf("(II) - Nombre de shared/thr : %d\n", attr_fx2.sharedSizeBytes);
    printf("(II) - Nombre de pBLOCKs    : %f\n", (float)nb_frames / (float)BLOCK_SIZE);
    printf("(II) - Nombre de pBLOCKs/uP : %f\n", (float)nb_frames / (float)BLOCK_SIZE / (float)devProp.multiProcessorCount);

#elif (DEG_1 == 7)
  	printf("(II) LDPC_Sched_Stage_1_MS_SIMD_Deg_7_Only :\n");
  	struct cudaFuncAttributes attr_fx2;    
    cudaFuncGetAttributes(&attr_fx2, LDPC_Sched_Stage_1_MS_SIMD_Deg_7_Only); 
  	printf("(II) - Nombre de regist/thr : %d\n", attr_fx2.numRegs);
  	printf("(II) - Nombre de local/thr  : %d\n", attr_fx2.localSizeBytes);
  	printf("(II) - Nombre de shared/thr : %d\n", attr_fx2.sharedSizeBytes);
    printf("(II) - Nombre de pBLOCKs    : %f\n", (float)nb_frames / (float)BLOCK_SIZE);
    printf("(II) - Nombre de pBLOCKs/uP : %f\n", (float)nb_frames / (float)BLOCK_SIZE / (float)devProp.multiProcessorCount);
#endif

  	if( attr.numRegs <= 32 ){
	  	printf("(II) Best BLOCK_SIZE   : %d\n", 128);
	  	printf("(II) Best BLOCK_SIZE   : %d\n", nBperMP/256);
  	}else if( attr.numRegs <= 40 ){
	  	printf("(II) Best BLOCK_SIZE   : %d\n", 96);
	  	printf("(II) Best BLOCK_SIZE   : %d\n", nBperMP/256);
  	}else if( attr.numRegs <= 48 ){
	  	printf("(II) Best BLOCK_SIZE   : %d\n", 128);
	  	printf("(II) Best BLOCK_SIZE   : %d\n", nBperMP/256);
  	}else if( attr.numRegs < 64 ){
	  	printf("(II) Best BLOCK_SIZE   : %d\n", 96);
	  	printf("(II) Best BLOCK_SIZE   : %d\n", nBperMP/256);
  	}else{
	  	printf("(II) Best BLOCK_SIZE   : ???\n");
	  	exit( 0 );
  	}

}


CGPU_Decoder_MS_SIMD_v2::~CGPU_Decoder_MS_SIMD_v2()
{
}

void CGPU_Decoder_MS_SIMD_v2::initialize()
{
}


void CGPU_Decoder_MS_SIMD_v2::decode(float Intrinsic_fix[_N], int Rprime_fix[_N], int nombre_iterations)
{
    cudaError_t Status;

    size_t nb_blocks = nb_frames / BLOCK_SIZE;
	if( nb_frames % BLOCK_SIZE != 0 ){
		printf("(%ld - %ld)  (%ld - %ld)\n", nb_frames, BLOCK_SIZE, nb_frames/BLOCK_SIZE, nb_frames%BLOCK_SIZE);
		exit( 0 );
	}


//#define MULTISTREAM
//#ifdef MULTISTREAM
//	cudaStream_t stream[16];
//	for(int y=0; y<nb_blocks; y++){
//		Status = cudaStreamCreate(&stream[y]);
//	}
//
//	for(int y=0; y<nb_blocks; y++){
//		unsigned int offSet = y * (sz_nodes/nb_blocks);
//
//		Status = cudaMemcpyAsync(
//				device_V + offSet,						Intrinsic_fix + offSet,
//				(sz_nodes/nb_blocks) * sizeof(float),	cudaMemcpyHostToDevice,
//				stream[y]);
//
//
//	    LDPC_Sched_Stage_1_MS_SIMD_deg6_only<<<1, BLOCK_SIZE, 0, stream[y]>>>(
//	    		(unsigned int*)(device_V    + offSet),
//	    		(unsigned int*)(d_MSG_C_2_V + offSet),
//	    		d_transpose, nombre_iterations);
//
//		{
//			unsigned int NB_TRAMES    = nb_blocks; // DIFFERENCE ICI
//			unsigned int FRAME_LENGTH = _N;
//			dim3 grid(NB_TRAMES/TILE_DIM, FRAME_LENGTH/TILE_DIM), threads(TILE_DIM,BLOCK_ROWS);
//			transposeDiagonal_and_hard_decision<<<grid, threads, 0, stream[y]>>>(
//					(float*)(d_MSG_C_2_V + offSet),
//					(float*)(device_V    + offSet),
//					NB_TRAMES, FRAME_LENGTH);
//		}
//
//	    Status = cudaMemcpyAsync(
//	    			Rprime_fix  + offSet,
//	    			d_MSG_C_2_V + offSet,
//	    			(sz_nodes/nb_blocks) * sizeof(float),
//	    			cudaMemcpyDeviceToHost, stream[y]);
//
//	}
//
//	for(int y=0; y<nb_blocks; y++){
//		Status = cudaStreamSynchronize(stream[y]);
//		ERROR_CHECK(Status, __FILE__, __LINE__);
//	}
//
//	for(int y=0; y<nb_blocks; y++){
//		Status = cudaStreamDestroy(stream[y]);
//		ERROR_CHECK(Status, __FILE__, __LINE__);
//	}
//
//	return;
//
//#endif

	//
	// ON COPIE LES DONNEES DANS => device_V
	//
    Status = cudaMemcpy/*Async*/(device_V, Intrinsic_fix, sz_nodes * sizeof(float), cudaMemcpyHostToDevice);
    ERROR_CHECK(Status, __FILE__, __LINE__);

//	printf("(MS_SIMD.cu) adr = %p\n", Intrinsic_fix);
//    for (int z = 0; z < 8; z++){
//		union {char c[4]; unsigned int i; float f;} value;
//		value.f = Intrinsic_fix[z];
//		printf("(MS_SIMD.cu : inp) z=%d, v1 = %d, v2 = %d, v3 = %d, v4 = %d\n", z, value.c[0], value.c[1], value.c[2], value.c[3]);
//	}

	//
    // INITIALISATION DE LA MEMOIRE MESSAGES => d_MSG_C_2_V
    //

//	DumpFloatMemoryDataSet("MSGS", d_MSG_C_2_V, 1024);
//	DumpFloatMemoryDataSet("MSGS", d_MSG_C_2_V + sz_msgs - 1024, 1024);

	//
	// PERUMATTATION DES DONNEES D'ENTREE DU DECODEUR
	//
//    LDPC_Interleave_LLR_Array<<<blocksPerGrid, threadsPerBlock>>>((float*)device_V, (float*)device_R, sz_nodes, nb_frames);

	const int debug = 0;

	if( debug ){
		DumpFloatMemoryDataSet("VARS", device_V,                   1024);
		DumpFloatMemoryDataSet("VARS", device_V + sz_nodes - 1024, 1024);
		DumpFloatMemoryDataSet("VARS", device_V + 8294400 - 16,    256);
	}


    //
    // LANCEMENT DU PROCESSUS DE DECODAGE SUR n ITERATIONS
    //

#if NB_DEGRES != 1

    LDPC_Sched_Stage_1_MS_SIMD_Two_CN_Degs<<<nb_blocks, BLOCK_SIZE>>>((unsigned int*)device_V, (unsigned int*)d_MSG_C_2_V, d_transpose, nombre_iterations);

#elif DEG_1 == 6

    LDPC_Sched_Stage_1_MS_SIMD_Deg_6_Only<<<nb_blocks, BLOCK_SIZE>>>((unsigned int*)device_V, (unsigned int*)d_MSG_C_2_V, d_transpose, nombre_iterations);

#elif DEG_1 == 7

    LDPC_Sched_Stage_1_MS_SIMD_Deg_7_Only<<<nb_blocks, BLOCK_SIZE>>>((unsigned int*)device_V, (unsigned int*)d_MSG_C_2_V, d_transpose, nombre_iterations);

#elif DEG_1 == 7

    #error "CN degree is not supported YET !"

#endif

#define NORMAL 1
#if NORMAL == 1
	{
//		printf("(II) NB_TRAMES       = %d;\n", nb_frames);
//		printf("(II) FRAME_LENGTH    = %d;\n", _N);
		dim3 grid(1, nb_frames/32);
		dim3 threads(32, 32);
//		printf("(II) Processing grid = %d, %d, %d;\n", grid.x, grid.y, grid.z);
//		printf("(II) Thread grid     = %d, %d, %d;\n", threads.x, threads.y, threads.z);
		InvInterleaver_uint8<<<grid, threads>>>((int*)device_V, (int*)d_MSG_C_2_V, _N, nb_frames);
	}
#else
	{
		unsigned int NB_TRAMES    = nb_frames;
		unsigned int FRAME_LENGTH = _N;
		dim3 grid(NB_TRAMES/TILE_DIM, FRAME_LENGTH/TILE_DIM), threads(TILE_DIM,BLOCK_ROWS);
		transposeDiagonal_and_hard_decision<<<grid, threads>>>((unsigned int*)d_MSG_C_2_V, (unsigned int*)device_V, NB_TRAMES, FRAME_LENGTH);
	}
#endif

    //
    //
    //
    Status = cudaMemcpy(Rprime_fix, d_MSG_C_2_V, sz_nodes * sizeof(float), cudaMemcpyDeviceToHost);
	ERROR_CHECK(Status, __FILE__, __LINE__);

}

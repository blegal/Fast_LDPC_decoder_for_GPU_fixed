/*
 *  ldcp_decoder.h
 *  ldpc3
 *
 *  Created by legal on 02/04/11.
 *  Copyright 2011 ENSEIRB. All rights reserved.
 *
 */

/*----------------------------------------------------------------------------*/

#include "CGPU_Decoder_NMS_SIMD.h"
#include "../transpose/GPU_Transpose.h"
#include "../transpose/GPU_Transpose_uint8.h"
#include "../tools/debug_fx.h"

static const size_t BLOCK_SIZE = 128; // 96 for exp.

CGPU_Decoder_NMS_SIMD::CGPU_Decoder_NMS_SIMD(size_t _nb_frames, size_t n, size_t k, size_t m):
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
	cudaFuncGetAttributes(&attr, LDPC_Sched_Stage_1_NMS_SIMD);

  	int nMP      = devProp.multiProcessorCount; // NOMBRE DE STREAM PROCESSOR
  	int nWarp    = attr.maxThreadsPerBlock/32;  // PACKET DE THREADs EXECUTABLES EN PARALLELE
  	int nThreads = nWarp * 32;					// NOMBRE DE THREAD MAXI PAR SP
  	int nDOF     = nb_frames;
  	int nBperMP  = 65536 / (attr.numRegs); 	// Nr of blocks per MP
  	int minB     = min(nBperMP*nThreads,1024);
  	int nBlocks  = max(minB/nThreads * nMP, nDOF/nThreads);  //Total number of blocks
  	printf("(II) Nombre de Warp    : %d\n", nWarp);
  	printf("(II) Nombre de Threads           : %d\n", nThreads);

  	printf("(II) LDPC_Sched_Stage_1_MS_SIMD :\n");
  	printf("(II) - Nombre de regist/thr : %d\n", attr.numRegs);
  	printf("(II) - Nombre de local/thr  : %ld\n", attr.localSizeBytes);
  	printf("(II) - Nombre de shared/thr : %ld\n", attr.sharedSizeBytes);

  	printf("(II) Nombre de nDOF    : %d\n", nDOF);
  	printf("(II) Nombre de nBperMP : %d\n", nBperMP);
  	printf("(II) Nombre de nBperMP : %d\n", minB);
  	printf("(II) Nombre de nBperMP : %d\n", nBlocks);
  	printf("(II) Best BLOCK_SIZE   : %d\n", nThreads * nBperMP);
  	printf("(II) Best #codewords   : %d\n", 0);

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


CGPU_Decoder_NMS_SIMD::~CGPU_Decoder_NMS_SIMD()
{
}

void CGPU_Decoder_NMS_SIMD::initialize()
{
}


void CGPU_Decoder_NMS_SIMD::decode(float Intrinsic_fix[_N], int Rprime_fix[_N], int nombre_iterations)
{
    cudaError_t Status;

    size_t nb_blocks = nb_frames / BLOCK_SIZE;
	if( nb_frames % BLOCK_SIZE != 0 ){
		printf("(%ld - %ld)  (%ld - %ld)\n", nb_frames, BLOCK_SIZE, nb_frames/BLOCK_SIZE, nb_frames%BLOCK_SIZE);
		exit( 0 );
	}


	//
	// ON COPIE LES DONNEES DANS => device_V
	//
    Status = cudaMemcpy/*Async*/(d_MSG_C_2_V, Intrinsic_fix, sz_nodes * sizeof(float), cudaMemcpyHostToDevice);
    ERROR_CHECK(Status, __FILE__, __LINE__);
	{
		dim3 grid(1, nb_frames/32);
		dim3 threads(32, 32);
		Interleaver_uint8<<<grid, threads>>>((int*)d_MSG_C_2_V, (int*)device_V, _N, nb_frames);
	}

    LDPC_Sched_Stage_1_NMS_SIMD<<<nb_blocks, BLOCK_SIZE>>>((unsigned int*)device_V, (unsigned int*)d_MSG_C_2_V, d_transpose, nombre_iterations);

	//
	// DESENTRELACEMENT DES DONNEES POST-DECODAGE (device_V => device_R)
	//
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

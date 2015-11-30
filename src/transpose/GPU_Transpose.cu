
#include "GPU_Transpose.h"
#include "simd_functions.h"

__global__ void transposeDiagonal(float *odata, float *idata, int width, int height) {
	__shared__ float tile[TILE_DIM][TILE_DIM + 1];
	int blockIdx_x, blockIdx_y;

	// diagonal reordering
	if (width == height) {
		blockIdx_y = blockIdx.x;
		blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
	} else {
		int bid = blockIdx.x + gridDim.x * blockIdx.y;
		blockIdx_y = bid % gridDim.y;
		blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
	}

	int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex) * width;
	xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex) * height;

	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
	}

	__syncthreads();

	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
	}
}


__global__ void transposeDiagonal_and_hard_decision(float *odata, float *idata, int width, int height) {
	__shared__ float tile[TILE_DIM][TILE_DIM + 1];
	int blockIdx_x, blockIdx_y;

	// diagonal reordering
	if (width == height) {
		blockIdx_y = blockIdx.x;
		blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
	} else {
		int bid = blockIdx.x + gridDim.x * blockIdx.y;
		blockIdx_y = bid % gridDim.y;
		blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
	}

	int xIndex    = blockIdx_x * TILE_DIM + threadIdx.x;
	int yIndex    = blockIdx_y * TILE_DIM + threadIdx.y;
	int index_in  = xIndex + (yIndex) * width;
	xIndex        = blockIdx_y * TILE_DIM + threadIdx.x;
	yIndex        = blockIdx_x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex) * height;

	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		float data = idata[index_in + i * width];
		union{ float f; unsigned int u; } resu; resu.u = data > 0;
		tile[threadIdx.y + i][threadIdx.x] = resu.f;
	}

	__syncthreads();

	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
	}
}

__global__ void transposeDiagonal_and_hard_decision(unsigned int* _odata, unsigned int* _idata, int width, int height) {
	float *odata = (float*)_odata;
	float *idata = (float*)_idata;
	__shared__ float tile[TILE_DIM][TILE_DIM + 1];
	int blockIdx_x, blockIdx_y;

	// diagonal reordering
	if (width == height) {
		blockIdx_y = blockIdx.x;
		blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
	} else {
		int bid = blockIdx.x + gridDim.x * blockIdx.y;
		blockIdx_y = bid % gridDim.y;
		blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
	}

	int xIndex   = blockIdx_x * TILE_DIM + threadIdx.x;
	int yIndex   = blockIdx_y * TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex) * width;
	xIndex       = blockIdx_y * TILE_DIM + threadIdx.x;
	yIndex       = blockIdx_x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex) * height;

	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		union{ float f; unsigned int u; } data;
		data.f = idata[index_in + i * width];			//
		data.u = vsetgts4(data.u, 0x00000000);			// HARD DECISION HERE ...
		tile[threadIdx.y + i][threadIdx.x] = data.f;	//
	}

	__syncthreads();

	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
	}
}

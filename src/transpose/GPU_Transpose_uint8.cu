#include "GPU_Transpose_uint8.h"
#include "simd_functions.h"

union t_uchar4 {
	unsigned char c[4];
	unsigned int v;
};

__global__ void InvInterleaver_uint8(int *in, int* out, int taille_frame, int nb_frames) {
	__shared__ int s[4][32][32];
	const int t = threadIdx.x;
	const int i = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	int taille_frame_4 = taille_frame / 4;

//		printf("i            = %d\n", i);
//	return;

	int *limit = in + (taille_frame/4) * nb_frames;

	//
	// ON LOAD LES DATA
	//
	for (int q = 0; q < taille_frame; q += 4 * 32) {
		int baseAddr = (4 * nb_frames * threadIdx.y) + i + (q * nb_frames);
		t_uchar4 perm1, perm2, perm3, perm4, value;

//		if ( &in[baseAddr] < limit) {
		value.v = in[baseAddr];
		value.v = vsetgts4(value.v, 0x00000000);			// HARD DECISION HERE ...
		perm1.c[0] = value.c[0];
		perm2.c[0] = value.c[1];
		perm3.c[0] = value.c[2];
		perm4.c[0] = value.c[3];
//		}
//		if ( &in[baseAddr + (1 * nb_frames)] < limit) {
		value.v = in[baseAddr + (1 * nb_frames)];
		value.v = vsetgts4(value.v, 0x00000000);			// HARD DECISION HERE ...
		perm1.c[1] = value.c[0];
		perm2.c[1] = value.c[1];
		perm3.c[1] = value.c[2];
		perm4.c[1] = value.c[3];
//		}

//		if ( &in[baseAddr + (2 * nb_frames)] < limit) {
		value.v = in[baseAddr + (2 * nb_frames)];
		value.v = vsetgts4(value.v, 0x00000000);			// HARD DECISION HERE ...
		perm1.c[2] = value.c[0];
		perm2.c[2] = value.c[1];
		perm3.c[2] = value.c[2];
		perm4.c[2] = value.c[3];
//		}

//		if ( &in[baseAddr + (3 * nb_frames)] < limit) {
		value.v = in[baseAddr + (3 * nb_frames)];
		value.v = vsetgts4(value.v, 0x00000000);			// HARD DECISION HERE ...
		perm1.c[3] = value.c[0];
		perm2.c[3] = value.c[1];
		perm3.c[3] = value.c[2];
		perm4.c[3] = value.c[3];
//		}

		s[0][t][threadIdx.y /*frame_id + p*/] = perm1.v;
		s[1][t][threadIdx.y /*frame_id + p*/] = perm2.v;
		s[2][t][threadIdx.y /*frame_id + p*/] = perm3.v;
		s[3][t][threadIdx.y /*frame_id + p*/] = perm4.v;

		__syncthreads();

		int bAddr = threadIdx.y * (taille_frame) + i % 32 + 32 * (i / 32) * taille_frame + q / 4;
		if ((q / 4 + t) < (taille_frame_4)) {
			out[bAddr] = s[0][threadIdx.y][t];
			out[bAddr + taille_frame_4] = s[1][threadIdx.y][t];
			out[bAddr + 2 * taille_frame_4] = s[2][threadIdx.y][t];
			out[bAddr + 3 * taille_frame_4] = s[3][threadIdx.y][t];
		}
		__syncthreads();
	}
}

__global__ void Interleaver_uint8(int *in, int* out, int taille_frame, int nb_frames) {
	__shared__ int s[4][32][32];
	const int t = threadIdx.x;
	const int i = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	int taille_frame_4 = taille_frame / 4;

	for (int q = 0; q < taille_frame; q += 4 * 32) {
		int bAddr = threadIdx.y * (taille_frame) + i % 32 + 32 * (i / 32) * taille_frame + q / 4;
		t_uchar4 perm1, perm2, perm3, perm4, value;

		value.v = in[bAddr];
		perm1.c[0] = value.c[0];
		perm2.c[0] = value.c[1];
		perm3.c[0] = value.c[2];
		perm4.c[0] = value.c[3];

		value.v = in[bAddr + (1 * taille_frame_4)];
		perm1.c[1] = value.c[0];
		perm2.c[1] = value.c[1];
		perm3.c[1] = value.c[2];
		perm4.c[1] = value.c[3];

		value.v = in[bAddr + (2 * taille_frame_4)];
		perm1.c[2] = value.c[0];
		perm2.c[2] = value.c[1];
		perm3.c[2] = value.c[2];
		perm4.c[2] = value.c[3];

		value.v = in[bAddr + (3 * taille_frame_4)];
		perm1.c[3] = value.c[0];
		perm2.c[3] = value.c[1];
		perm3.c[3] = value.c[2];
		perm4.c[3] = value.c[3];

		s[0][t][threadIdx.y] = perm1.v;
		s[1][t][threadIdx.y] = perm2.v;
		s[2][t][threadIdx.y] = perm3.v;
		s[3][t][threadIdx.y] = perm4.v;
		__syncthreads();

		int baseAddr = (4 * nb_frames * (threadIdx.y)) + i + q * nb_frames;

		if ((q / 4 + threadIdx.y) < (taille_frame_4)) {
			out[baseAddr] = s[0][threadIdx.y][t];
			out[baseAddr + nb_frames] = s[1][threadIdx.y][t];
			out[baseAddr + 2 * nb_frames] = s[2][threadIdx.y][t];
			out[baseAddr + 3 * nb_frames] = s[3][threadIdx.y][t];
		}
		__syncthreads();
	}
}

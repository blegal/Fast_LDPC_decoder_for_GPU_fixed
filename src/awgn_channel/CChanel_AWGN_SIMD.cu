#include "CChanel_AWGN_SIMD.h"

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
      printf("Error (%d) at %s:%d\n", x, __FILE__,__LINE__);            \
      exit(0);}} while(0)

__global__ void GenerateNoiseAndTransform(const float *A, const float *B, int *C, float SigB, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
    {
		float vSin, vCos, x, y;
	    union {char c[4]; unsigned int i;} res_a = {0, 0, 0, 0};
	    union {char c[4]; unsigned int i;} res_b = {0, 0, 0, 0};

		for(int p=0; p<4; p++){
			x  = sqrt(-2.0 * log( A[i + p * N] ));
			y  = B[i + p * N];
			sincosf(_2pi * y, &vSin, &vCos);
			float v1   = (-1.0 + (x * vSin) * SigB);
			float v2   = (-1.0 + (x * vCos) * SigB);
			res_a.c[p] = (char)fminf( fmaxf(8.0f * v1, -31.0f), 31.0f);
			res_b.c[p] = (char)fminf( fmaxf(8.0f * v2, -31.0f), 31.0f);
		}
        C[i]   = res_a.i;
        C[i+N] = res_b.i;

    }
}

//#define SEQ_LEVEL 1

CChanel_AWGN_SIMD::CChanel_AWGN_SIMD(CTrame *t, int _BITS_LLR, bool QPSK, bool Es_N0) : CChanel(t, _BITS_LLR, QPSK, Es_N0){
	curandStatus_t Status;

	SEQ_LEVEL = 1 + ((_data > 10000) ? 3 : 0);

	unsigned int nb_ech = (_frames * _data) / SEQ_LEVEL;
	Status = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	CURAND_CALL(Status);
    Status = curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);
	CURAND_CALL(Status);
	CUDA_MALLOC_DEVICE(&device_A, nb_ech/2,__FILE__, __LINE__);
    CUDA_MALLOC_DEVICE(&device_B, nb_ech/2,__FILE__, __LINE__);
    CUDA_MALLOC_DEVICE(&device_R, nb_ech  ,__FILE__, __LINE__);
}

CChanel_AWGN_SIMD::~CChanel_AWGN_SIMD(){
	cudaError_t Status;
	Status = cudaFree(device_A);
	ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
	Status = cudaFree(device_B);
	ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
	Status = cudaFree(device_R);
	ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
	curandStatus_t eStatus;
    eStatus = curandDestroyGenerator(generator);
	CURAND_CALL(eStatus);
//    printf("(DD) STOP  CChanel_AWGN_SIMD::~CChanel_AWGN_SIMD() !\n");
}

void CChanel_AWGN_SIMD::configure(double _Eb_N0) {
    rendement = (float) (_vars) / (float) (_data);
    if (es_n0) {
        Eb_N0 = _Eb_N0 - 10.0 * log10(2 * rendement);
    } else {
        Eb_N0 = _Eb_N0;
    }
    double interm = 10.0 * log10(rendement);
    interm        = -0.1*((double)Eb_N0+interm);
    SigB          = sqrt(pow(10.0,interm)/2);
}

#include <limits.h>
#define MAX_RANDOM LONG_MAX    /* Maximum value of random() */


double CChanel_AWGN_SIMD::awgn(double amp)
{
    return 0.00;
}

#define QPSK 0.707106781
#define BPSK 1.0


void CChanel_AWGN_SIMD::generate()
{
	size_t nb_rand_data = _frames*_data / 2 / SEQ_LEVEL;
	CURAND_CALL( curandGenerateUniform( generator, device_A, nb_rand_data ) );
	CURAND_CALL( curandGenerateUniform( generator, device_B, nb_rand_data ) );

	for(int i=0; i<4 * SEQ_LEVEL; i++){

		// GENERATION DE DU BRUIT DU CANAL
		size_t nb_noise_sample = nb_rand_data / 2;
		int threadsPerBlock = 1024;
		size_t blocksPerGrid   = (nb_noise_sample  + threadsPerBlock - 1) / threadsPerBlock;
		GenerateNoiseAndTransform<<<blocksPerGrid, threadsPerBlock>>>(device_A, device_B, (int*)device_R, (float)SigB, nb_noise_sample/2);

		cudaError_t eStatus = cudaMemcpyAsync(&t_noise_data[i * nb_noise_sample], device_R, nb_noise_sample * sizeof(float), cudaMemcpyDeviceToHost);

		if( i != 3 ){
		CURAND_CALL( curandGenerateUniform( generator, device_A, nb_rand_data ) );
		CURAND_CALL( curandGenerateUniform( generator, device_B, nb_rand_data ) );
		}
		cudaDeviceSynchronize();
		ERROR_CHECK(cudaGetLastError(), __FILE__, __LINE__);

	}
}

#ifndef CLASS_CChanel_AWGN_SIMD
#define CLASS_CChanel_AWGN_SIMD

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "CChanel.h"


#include "../custom_api/custom_cuda.h"
#include <curand.h>

#define small_pi  3.1415926536
#define _2pi  (2.0 * small_pi)

class CChanel_AWGN_SIMD : public CChanel
{
private:
    double awgn(double amp);
    float *device_A;
    float *device_B;
    float *device_R;
	curandGenerator_t generator;

	unsigned int SEQ_LEVEL;

public:
	CChanel_AWGN_SIMD(CTrame *t, int _BITS_LLR, bool QPSK, bool Es_N0);
    ~CChanel_AWGN_SIMD();
    
    virtual void configure(double _Eb_N0);
    virtual void generate();
};

#endif


/*
 *  ldcp_decoder.h
 *  ldpc3
 *
 *  Created by legal on 02/04/11.
 *  Copyright 2011 ENSEIRB. All rights reserved.
 *
 */

/*----------------------------------------------------------------------------*/


#ifndef __CLASS_CGPUDecoder__
#define __CLASS_CGPUDecoder__

#include "../custom_api/custom_cuda.h"
#include "../matrix/constantes_gpu.h"

class CGPUDecoder{
protected:
    float* device_V;
    float* d_MSG_C_2_V;
    unsigned int* d_transpose;

    size_t nb_frames;
    size_t sz_nodes;
    size_t sz_checks;
    size_t sz_msgs;

public:
	CGPUDecoder(size_t _nb_frames, size_t n, size_t k, size_t m );
    virtual ~CGPUDecoder();
    virtual void initialize() = 0;
    virtual void decode(float var_nodes[_N], int Rprime_fix[_N], int nombre_iterations) = 0;
};

#endif

#ifndef CLASS_CTimer
#define CLASS_CTimer

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;

// includes, project
// includes, CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>


class CTimer
{
    
protected:
	cudaEvent_t _start;
	cudaEvent_t _stop;
	bool isRunning;
    
public:
    
    CTimer(bool _start);

    CTimer();

    ~CTimer();

    void start();

    void stop();

    void reset();

    long get_time_ns();

    long get_time_us();

    long get_time_ms();

    long get_time_sec();

};

#endif

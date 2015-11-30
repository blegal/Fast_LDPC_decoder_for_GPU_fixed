#include "CTimer.h"


CTimer::CTimer(bool _start_timer_){
	cudaEventCreate(&_start);
	cudaEventCreate(&_stop);

    if(_start_timer_ == true){
    	cudaEventRecord(_start, 0);
        isRunning = true;
    }else{
        isRunning = false;
    }
}


CTimer::CTimer(){
	cudaEventCreate(&_start);
	cudaEventCreate(&_stop);
	isRunning = false;
}


CTimer::~CTimer(){
	cudaEventDestroy(_start);
	cudaEventDestroy(_stop);
}


void CTimer::start(){
    if( isRunning == true ){
        cout << "(EE) CTimer :: trying to start a CTimer object that is already running !" << endl;
    }else{
        isRunning = true;
    	cudaEventRecord(_start, 0);
    }
}


void CTimer::stop(){
	cout << "CTimer::stop()" << endl;
    if( isRunning == false ){
        cout << "(EE) CTimer :: trying to stop a CTimer object that is not running !" << endl;
    }else{
        cudaEventRecord(_stop, 0);
        isRunning = false;
    }
}


void CTimer::reset(){
	cudaEventRecord(_start, 0);
}


long CTimer::get_time_ns(){
	float elapsedTime;
	if( isRunning == true ){
	    cudaEventRecord(_stop, 0);
	    cudaEventSynchronize(_stop);
	}
	cudaEventElapsedTime(&elapsedTime, _start, _stop); // that's our time!
	return (long)(1000.0 * 1000.0 * elapsedTime);
}


long CTimer::get_time_us(){
	float elapsedTime;
	if( isRunning == true ){
	    cudaEventRecord(_stop, 0);
	    cudaEventSynchronize(_stop);
	}
	cudaEventElapsedTime(&elapsedTime, _start, _stop); // that's our time!
	return (long)(1000.0 * elapsedTime);
}


long CTimer::get_time_ms(){
	float elapsedTime;
	if( isRunning == true ){
	    cudaEventRecord(_stop, 0);
	    cudaEventSynchronize(_stop);
	}
	cudaEventElapsedTime(&elapsedTime, _start, _stop); // that's our time!
	return (long)(elapsedTime);
}


long CTimer::get_time_sec(){
	return (long)(get_time_ms() / 1000.0);
}

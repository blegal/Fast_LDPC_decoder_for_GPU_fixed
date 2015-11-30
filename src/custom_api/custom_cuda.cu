#include "custom_cuda.h"

#define DEBUG 1

bool ERROR_CHECK(cudaError_t Status, string file, int line)
{
    if(Status != cudaSuccess)
    {
        printf("(EE) \n");
        printf("(EE) Error detected in the LDPC decoder (%s : %d)\n", file.c_str(), line);
        printf("(EE) MSG: %s\n", cudaGetErrorString(Status));
        printf("(EE) \n");
		exit(0);
        return false;
    }
    return true;
}

char* FilenamePtr(const char* filename){
	char* fname = (char*)filename;
	char* ptr = fname;
	while( *fname != 0 ){
		if( *fname == '\\' ) ptr = fname + 1;
		if( *fname == '/'  ) ptr = fname + 1;
		fname += 1;
	}
	return ptr;
}

void CUDA_MALLOC_HOST(float** ptr, size_t nbElements, const char * file, int line){
    cudaError_t Status;
    size_t nbytes = nbElements * sizeof(float);
    Status = cudaMallocHost(ptr, nbytes);
#if DEBUG == 1
	printf("(II)    + Allocating (%s:%d) Host   Memory, %ld elements (%ld ko) adr [%p, %p]\n", FilenamePtr(file), line, nbElements, nbytes/1024, *ptr, *ptr+nbElements-1);
#endif
    ERROR_CHECK(Status, __FILE__, __LINE__);
}

void CUDA_MALLOC_HOST(int** ptr, size_t nbElements, const char * file, int line){
    cudaError_t Status;
    size_t nbytes = nbElements * sizeof(int);
    Status = cudaMallocHost(ptr, nbytes);
#if DEBUG == 1
	printf("(II)    + Allocating (%s:%d) Host   Memory, %ld elements (%ld ko) adr [%p, %p]\n", FilenamePtr(file), line, nbElements, nbytes/1024, *ptr, *ptr+nbElements-1);
#endif
    ERROR_CHECK(Status, __FILE__, __LINE__);
}

void CUDA_MALLOC_HOST(unsigned int** ptr, size_t nbElements, const char * file, int line){
    cudaError_t Status;
    size_t nbytes = nbElements * sizeof(unsigned int);
    Status = cudaMallocHost(ptr, nbytes);
#if DEBUG == 1
	printf("(II)    + Allocating (%s:%d) Host   Memory, %ld elements (%ld ko) adr [%p, %p]\n", FilenamePtr(file), line, nbElements, nbytes/1024, *ptr, *ptr+nbElements-1);
#endif
    ERROR_CHECK(Status, __FILE__, __LINE__);
}

static size_t aDevice = 0;

void CUDA_MALLOC_HOST(char** ptr, size_t nbElements, const char * file, int line){
    cudaError_t Status;
    size_t nbytes = nbElements * sizeof(char);
    Status     = cudaMallocHost(ptr, nbytes);
	aDevice   += nbytes;
#if DEBUG == 1
	printf("(II)    + Allocating (%s:%d) Host   Memory, %ld elements (%ld ko) adr [%p, %p]\n", FilenamePtr(file), line,  nbElements, nbytes/1024, *ptr, *ptr+nbElements-1);
#endif
    ERROR_CHECK(Status, __FILE__, __LINE__);
}

void CUDA_MALLOC_DEVICE(float** ptr, size_t nbElements, const char * file, int line){
    cudaError_t Status;
    size_t nbytes = nbElements * sizeof(float);
    Status     = cudaMalloc(ptr, nbytes);
	aDevice   += nbytes;
#if DEBUG == 1
	printf("(II)    + Allocating (%s:%d) Device Memory, %ld elements (%ld ko) adr [%p, %p]\n", FilenamePtr(file), line, nbElements, nbytes/1024, *ptr, *ptr+nbElements-1);
//	printf("(II)    + Memory allocated on GPU device = %d Mo\n", aDevice/1024/1024);
#endif
    ERROR_CHECK(Status, __FILE__, __LINE__);
}

void CUDA_MALLOC_DEVICE(int** ptr, size_t nbElements, const char * file, int line){
    cudaError_t Status;
    size_t nbytes = nbElements * sizeof(int);
    Status     = cudaMalloc(ptr, nbytes);
	aDevice   += nbytes;
#if DEBUG == 1
	printf("(II)    + Allocating (%s:%d) Device Memory, %ld elements (%ld ko) adr [%p, %p]\n", FilenamePtr(file), line, nbElements, nbytes/1024, *ptr, *ptr+nbElements-1);
//	printf("(II)    + Memory allocated on GPU device = %d Mo\n", aDevice/1024/1024);
#endif
    ERROR_CHECK(Status, __FILE__, __LINE__);
}

void CUDA_MALLOC_DEVICE(unsigned int** ptr, size_t nbElements, const char * file, int line){
    cudaError_t Status;
    size_t nbytes = nbElements * sizeof(unsigned int);
    Status     = cudaMalloc(ptr, nbytes);
	aDevice   += nbytes;
#if DEBUG == 1
	printf("(II)    + Allocating (%s:%d) Device Memory, %ld elements (%ld ko) adr [%p, %p]\n", FilenamePtr(file), line, nbElements, nbytes/1024, *ptr, *ptr+nbElements-1);
//	printf("(II)    + Memory allocated on GPU device = %d Mo\n", aDevice/1024/1024);
#endif
    ERROR_CHECK(Status, __FILE__, __LINE__);
}

void CUDA_MALLOC_DEVICE(char** ptr, size_t nbElements, const char * file, int line){
    cudaError_t Status;
    size_t nbytes = nbElements * sizeof(char);
    Status     = cudaMalloc(ptr, nbytes);
	aDevice   += nbytes;
#if DEBUG == 1
	printf("(II)    + Allocating (%s:%d) Device Memory, %ld elements (%ld ko) adr [%p, %p]\n", FilenamePtr(file), line, nbElements, nbytes/1024, *ptr, *ptr+nbElements-1);
//	printf("(II)    + Memory allocated on GPU device = %d Mo\n", aDevice/1024/1024);
#endif
    ERROR_CHECK(Status, __FILE__, __LINE__);
}

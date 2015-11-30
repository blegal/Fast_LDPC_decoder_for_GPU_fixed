/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 3
 * of the programming guide with some additions like error checking.
 *
 */

#include "debug_fx.h"

void PrintIntegerMatrix(int *dataSet, int size){
    int sum = 0;
    for(int i=0; i<size; i++){
        if( i >= 2048 ) break;
        if( (i%64 == 0) && (i != 0) ){
            printf(" [%d]\n", sum);
            sum = 0;
        }
        sum += dataSet[i];
        printf("%d ", dataSet[i]);
    }
    printf("\n");
}


void PrintFloatMatrix(float *dataSet, int size){
    for(int i=0; i<size; i++){
        if( i >= 512 ) break;
        if( (i%32 == 0) && (i != 0) ) printf("\n");
        if( dataSet[i] >= 0.0 ) printf(" ");
        printf("%2.1f ", dataSet[i]);
    }
    printf("\n");
}


void PrintFloatMatrix(string name, float *dataSet, int size, int nb_lines){
    printf("(DD)\n");
    printf("(DD) PrintFloatMatrix(%s)\n", name.c_str());
    printf("(DD)\n");
	int nb_data_par_ligne = 16;
	printf("%4d : ", 0);
    for(int i=0; i<(nb_lines*nb_data_par_ligne); i++){
        if( (i%nb_data_par_ligne == 0) && (i != 0) ) printf("\n%4d : ", i);
        if( dataSet[i] >= 0.0 ) printf(" ");
        printf("%2.1f ", dataSet[i]);
    }
    printf("\n");

    printf(" ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...\n");
    printf(" ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...\n");
    printf(" ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...\n");

	printf("%4d : ", size-(nb_lines*nb_data_par_ligne));
    for(int i=0; i<(nb_lines*nb_data_par_ligne); i++){
        if( (i%nb_data_par_ligne == 0) && (i != 0) ) printf("\n%4d : ", (size-(nb_lines*nb_data_par_ligne)+i));
        if( dataSet[ (size-(nb_lines*nb_data_par_ligne)+i) ] >= 0.0 ) printf(" ");
        printf("%2.1f ", dataSet[ size-((nb_lines*nb_data_par_ligne)+i) ]);
    }
    printf("\n");
}

void PrintFloatMatrix(char*name, float *dataSet, int size){
    printf("(DD)\n");
    printf("(DD) PrintFloatMatrix(%s)\n", name);
    printf("(DD)\n");
	int nb_data_par_ligne = 16;
	printf("%4d : ", 0);
    for(int i=0; i<size; i++){
        if( (i%nb_data_par_ligne == 0) && (i != 0) ) printf("\n%4d : ", i);
        if( dataSet[i] >= 0.0 ) printf(" ");
        printf("%4.0f ", dataSet[i]);
    }
    printf("\n");
}

void PrintIntegerMatrix(char*name, unsigned int *dataSet, int size, int nb_lines){
    printf("(DD)\n");
    printf("(DD) PrintIntegerMatrix(%s)\n", name);
    printf("(DD)\n");
	int nb_data_par_ligne = 16;
	printf("%4d : ", 0);
    for(int i=0; i<(nb_lines*nb_data_par_ligne); i++){
        if( (i%nb_data_par_ligne == 0) && (i != 0) ) printf("\n%4d : ", i);
        if( dataSet[i] >= 0.0 ) printf(" ");
        printf("%5d ", dataSet[i]);
    }
    printf("\n");

    printf(" ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...\n");
    printf(" ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...\n");
    printf(" ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...\n");

    int start_pos = size - (nb_lines*nb_data_par_ligne);
	printf("%4d : ", size-(nb_lines*nb_data_par_ligne));
    for(int i=start_pos; i<size; i++){
        if( ((i-start_pos)%nb_data_par_ligne == 0) && ((i-start_pos) != 0) ){
        	printf("\n%4d : ", i);
        }
        if( dataSet[ i ] >= 0.0 ){
        	printf(" ");
        }
        printf("%5d ", dataSet[ i ]);
    }
    printf("\n");
}

void PrintIntegerMatrix(char*name, unsigned int *dataSet, int size){
    printf("(DD)\n");
    printf("(DD) PrintIntegerMatrix(%s)\n", name);
    printf("(DD)\n");
	int nb_data_par_ligne = 16;
	printf("%4d : ", 0);
    for(int i=0; i<size; i++){
        if( (i%nb_data_par_ligne == 0) && (i != 0) ) printf("\n%4d : ", i);
        if( dataSet[i] >= 0.0 ) printf(" ");
        printf("%5d ", dataSet[i]);
    }
    printf("\n");
}


bool CheckMemoryDataSet(char *name, const unsigned int* d_values, const unsigned int* h_values, int nb_data ){

    //
    // ON ALLOUE LA ZONE MEMOIRE POUR RECUPERE LES DONNEES PROVENANT DU GPU
    //
//    printf("(II) CheckMemoryDataSet(0x%8.8X, 0x%8.8X, %d)\n", (unsigned int)d_values, (unsigned int)h_values, (unsigned int)nb_data);
    cudaError_t Status;
    unsigned int* x_values;

    cudaMalloc(&x_values, nb_data * sizeof(int));
    Status = cudaGetLastError();
    if(Status != cudaSuccess)
    {
    	printf("\n1 %s\n", cudaGetErrorString(Status));
    }


    cudaMemcpy(x_values, d_values, nb_data * sizeof(int), cudaMemcpyDeviceToHost);
    Status = cudaGetLastError();
    if(Status != cudaSuccess)
    {
    	printf("\n1 %s\n", cudaGetErrorString(Status));
    }
//    cError = cuMemcpyDtoH(x_values, d_values, nb_data * sizeof(int));
//    if (cError != CUDA_SUCCESS){ /*checkCudaErrors( cError );*/ }

    int errors = 0;
    for(int i=0; i<nb_data; i++){
        if( x_values[i] != h_values[i] ){
            errors += 1;
        }
    }
    if( errors != 0 ){
        printf("(EE) DataSet are differents (%d errors)\n", errors);
    }else{
        printf("(II) DataSet are identical\n");
    }

    cudaFree(x_values);

    return (errors == 0);
}

bool DumpFloatMemoryDataSet(string name, float* device_values, int nb_data ){

    //
    // ON ALLOUE LA ZONE MEMOIRE POUR RECUPERE LES DONNEES PROVENANT DU GPU
    //
    printf("(II) DumpFloatMemoryDataSet(%s, %p, %d)\n", name.c_str(), device_values, nb_data);
    cudaError_t Status;
    float* host_values;
    CUDA_MALLOC_HOST(&host_values, nb_data, __FILE__, __LINE__);

    Status = cudaMemcpy(host_values, device_values, nb_data * sizeof(float), cudaMemcpyDeviceToHost);
    if(Status != cudaSuccess)
    {
    	printf("\n1 %s\n", cudaGetErrorString(Status));
    }

    PrintFloatMatrix(name, host_values, nb_data, 8);
    Status = cudaFreeHost(host_values);
    if(Status != cudaSuccess)
    {
    	printf("\n1 %s\n", cudaGetErrorString(Status));
    }

    return true;
}

bool DumpFullFloatMemoryDataSet(char *name, float* device_values, int nb_data ){

    //
    // ON ALLOUE LA ZONE MEMOIRE POUR RECUPERE LES DONNEES PROVENANT DU GPU
    //
    printf("(II) DumpFloatMemoryDataSet(%s, %p, %d)\n", name, device_values, nb_data);
    cudaError_t Status;
    float* host_values;
    CUDA_MALLOC_HOST(&host_values, nb_data, __FILE__, __LINE__);

    Status = cudaMemcpy(host_values, device_values, nb_data * sizeof(float), cudaMemcpyDeviceToHost);
    if(Status != cudaSuccess)
    {
    	printf("\n1 %s\n", cudaGetErrorString(Status));
    }

    PrintFloatMatrix(name, host_values, nb_data);
    Status = cudaFreeHost(host_values);
    if(Status != cudaSuccess)
    {
    	printf("\n1 %s\n", cudaGetErrorString(Status));
    }

    return true;
}

bool DumpIntegerMemoryDataSet(char *name, unsigned int* device_values, int nb_data ){

    //
    // ON ALLOUE LA ZONE MEMOIRE POUR RECUPERE LES DONNEES PROVENANT DU GPU
    //
    printf("(II) DumpFloatMemoryDataSet(%s, %p, %d)\n", name, device_values, nb_data);
    cudaError_t Status;
    unsigned int* host_values;
    CUDA_MALLOC_HOST(&host_values, nb_data, __FILE__, __LINE__);

    Status = cudaMemcpy(host_values, device_values, nb_data * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if(Status != cudaSuccess)
    {
    	printf("\n1 %s\n", cudaGetErrorString(Status));
    }

    PrintIntegerMatrix(name, host_values, nb_data, 8);
    // PrintIntegerMatrix(name, host_values, nb_data);
    Status = cudaFreeHost(host_values);
    if(Status != cudaSuccess)
    {
    	printf("\n1 %s\n", cudaGetErrorString(Status));
    }
    return true;
}


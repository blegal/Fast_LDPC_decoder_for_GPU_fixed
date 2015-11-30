#include "../custom_api/custom_cuda.h"

#ifndef DEBUG_FX
#define DEBUG_FX

void PrintIntegerMatrix(int *dataSet, int size);
void PrintFloatMatrix(float *dataSet, int size);
void PrintFloatMatrix(char*name, float *dataSet, int size, int nb_lines);
void PrintIntegerMatrix(char*name, unsigned int *dataSet, int size, int nb_lines);
void PrintIntegerMatrix(char*name, unsigned int *dataSet, int size);
bool CheckMemoryDataSet(char *name, const unsigned int* d_values, const unsigned int* h_values, int nb_data );
bool DumpFloatMemoryDataSet(string name, float* device_values, int nb_data );
bool DumpFullFloatMemoryDataSet(char *name, float* device_values, int nb_data );
bool DumpIntegerMemoryDataSet(char *name, unsigned int* device_values, int nb_data );

#endif //DEBUG_FX

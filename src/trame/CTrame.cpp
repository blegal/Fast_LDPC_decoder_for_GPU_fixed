#include "CTrame.h"

#include "../custom_api/custom_cuda.h"

CTrame::CTrame(int width, int height){
    _width        = width;
    _height       = height;
    _frame        = 1;
//    t_in_bits     = new int   [ nb_vars() ];
//    t_coded_bits  = new int   [ nb_data() ];

    CUDA_MALLOC_HOST(&t_noise_data, nb_data() + 1, __FILE__, __LINE__);
//    t_noise_data  = new float[ nb_data() + 1 ];
//    t_fpoint_data = new int   [ nb_data() ];
    CUDA_MALLOC_HOST(&t_fpoint_data, nb_data() + 1, __FILE__, __LINE__);
    CUDA_MALLOC_HOST(&t_decode_data, nb_data() + 1, __FILE__, __LINE__);
//    t_decode_data = new int   [ nb_data() ];
//    t_decode_bits = new int   [ nb_vars() ];
//    printf("t_in_bits     : %d MBytes\n", (sizeof(int) * nb_vars())/1024 );
//    printf("t_coded_bits  : %d MBytes\n", (sizeof(int) * nb_data())/1024 );
//    printf("t_fpoint_data : %d MBytes\n", (sizeof(int) * nb_data())/1024 );
//    printf("t_decode_data : %d MBytes\n", (sizeof(int) * nb_data())/1024 );
//    printf("t_coded_bits  : %d MBytes\n", (sizeof(int) * nb_vars())/1024 );
}

CTrame::CTrame(int width, int height, int frame){
    _width        = width;
    _height       = height;
    _frame        = frame;
//    printf("t_in_bits     0x%p : %f MBytes\n", t_coded_bits,  (sizeof(int) * nb_data() * frame)/1024.0/1024.0 );
//    printf("t_coded_bits  0x%p : %f MBytes\n", t_coded_bits,  (sizeof(int) * nb_data() * frame)/1024.0/1024.0 );
//    printf("t_noise_data  0x%p : %f MBytes\n", t_noise_data,  (sizeof(int) * nb_data() * frame)/1024.0/1024.0 );
//    printf("t_fpoint_data 0x%p : %f MBytes\n", t_fpoint_data, (sizeof(int) * nb_data() * frame)/1024.0/1024.0 );
//    printf("t_decode_data 0x%p : %f MBytes\n", t_decode_data, (sizeof(int) * nb_data() * frame)/1024.0/1024.0 );
//    printf("t_coded_bits  0x%p : %f MBytes\n", t_coded_bits,  (sizeof(int) * nb_vars() * frame)/1024.0/1024.0 );
//	t_in_bits     = NULL; // new int   [ nb_vars() * frame ];
//    t_coded_bits  = NULL; // new int   [ nb_data() * frame ];
    CUDA_MALLOC_HOST(&t_noise_data, nb_data()  * frame + 4, __FILE__, __LINE__);
//    t_noise_data  = new float[ nb_data() ];
    CUDA_MALLOC_HOST(&t_fpoint_data, nb_data() * frame + 4, __FILE__, __LINE__);
    CUDA_MALLOC_HOST(&t_decode_data, nb_data() * frame + 4, __FILE__, __LINE__);
//    t_fpoint_data = new int   [ nb_data() * frame ];
//    t_decode_data = new int   [ nb_data() * frame ];
//    t_decode_bits = new int   [ nb_vars() * frame ];
}


CTrame::~CTrame(){
//    if( t_in_bits    != NULL ) delete t_in_bits;
//    if( t_coded_bits != NULL ) delete t_coded_bits;
    //    delete t_noise_data;
	cudaFreeHost(t_noise_data);
	cudaFreeHost(t_fpoint_data);
	cudaFreeHost(t_decode_data);
//    delete t_fpoint_data;
//    delete t_decode_data;
//    delete t_decode_bits;
//    printf("t_in_bits     : %ld MBytes\n", (sizeof(int) * nb_vars())/1024 );
//    printf("t_coded_bits  : %ld MBytes\n", (sizeof(int) * nb_data())/1024 );
//    printf("t_fpoint_data : %ld MBytes\n", (sizeof(int) * nb_data())/1024 );
//    printf("t_decode_data : %ld MBytes\n", (sizeof(int) * nb_data())/1024 );
//    printf("t_coded_bits  : %ld MBytes\n", (sizeof(int) * nb_vars())/1024 );
}

int CTrame::nb_vars(){
    return  /*nb_frames() * */(nb_data()-nb_checks());
}

int CTrame::nb_frames(){
    return  _frame;
}

int CTrame::nb_checks(){
    return _height;
}

int CTrame::nb_data(){
    return _width;
}

//int* CTrame::get_t_in_bits(){
//    return t_in_bits;
//}

//int* CTrame::get_t_coded_bits(){
//    return t_coded_bits;
//}

float* CTrame::get_t_noise_data(){
    return t_noise_data;
}

int* CTrame::get_t_fpoint_data(){
    return t_fpoint_data;
}

int* CTrame::get_t_decode_data(){
    return t_decode_data;
}

//int* CTrame::get_t_decode_bits(){
//    return t_decode_bits;
//}

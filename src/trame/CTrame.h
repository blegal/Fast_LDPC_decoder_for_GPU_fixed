#include <stdlib.h>
#include <stdio.h>

#ifndef CLASS_TRAME
#define CLASS_TRAME

class CTrame
{
    
protected:
    int  _width;
    int  _height;
    int  _frame;
    
//    int*    t_in_bits;      // taille (var)
//    int*    t_coded_bits;   // taille (width)
    float*  t_noise_data;   // taille (width)
    int*    t_fpoint_data;  // taille (width/4)
    int*    t_decode_data;  // taille (var)
//    int*    t_decode_bits;  // taille (var)
    
public:
    CTrame(int width, int height);
    CTrame(int width, int height, int frame);
    ~CTrame();
    
    int nb_vars();
    int nb_checks();
    int nb_data();
    int nb_frames();
    
    //int*   get_t_in_bits();
    //int*   get_t_coded_bits();
    float* get_t_noise_data();
    int*   get_t_fpoint_data();
    int*   get_t_decode_data();
//    int*   get_t_decode_bits();
};

#endif

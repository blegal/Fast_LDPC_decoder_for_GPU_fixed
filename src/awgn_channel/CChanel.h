#ifndef CLASS_CChanel
#define CLASS_CChanel

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../trame/CTrame.h"

#define small_pi  3.1415926536
#define _2pi  (2.0 * small_pi)

class CChanel
{
    
protected:
	size_t  _vars;
	size_t  _checks;
	size_t  _data;
    int  BITS_LLR;
//    int* data_in;
    int* data_out;
    bool qpsk;
    bool es_n0;
    size_t _frames;
    
    float*  t_noise_data;   // taille (width)
    int*    t_coded_bits;   // taille (width)
    
    double rendement;
    double SigB;
    double Gauss;
    double Ph;
    double Qu;
    double Eb_N0;
    
public:
    CChanel(CTrame *t, int _BITS_LLR, bool QPSK, bool Es_N0);
    virtual ~CChanel();
    virtual void configure(double _Eb_N0) = 0;  // VIRTUELLE PURE
    virtual void generate() = 0;                // VIRTUELLE PURE    
};

#endif


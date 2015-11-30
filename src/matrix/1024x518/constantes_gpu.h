#ifndef CONSTANTES
#define CONSTANTES

#include <math.h>

#define NB_DEGRES            2

#define _N                   1024 // Nombre de Variables
#define _K                   518 // Nombre de Checks   
#define _M                   3815 // Nombre de Messages 

#define NOEUD   _N
#define MESSAGE _M

#define NmoinsK     (_N-_K)

#define DEG_1                8
#define DEG_2                7

#define DEG_1_COMPUTATIONS   189
#define DEG_2_COMPUTATIONS   329

extern const unsigned int PosNoeudsVariable[_M];

#endif

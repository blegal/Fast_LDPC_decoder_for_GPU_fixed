#ifndef CONSTANTES
#define CONSTANTES

#include <math.h>

#define NB_DEGRES            1

#define _N                   2048 // Nombre de Variables
#define _K                   384 // Nombre de Checks   
#define _M                   12288 // Nombre de Messages 

#define NOEUD   _N
#define MESSAGE _M

#define NmoinsK     (_N-_K)

#define DEG_1                32

#define DEG_1_COMPUTATIONS   384

extern const unsigned int PosNoeudsVariable[_M];

#endif

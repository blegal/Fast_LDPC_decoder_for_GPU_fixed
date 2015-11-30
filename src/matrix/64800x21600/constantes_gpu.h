#ifndef _CONSTANTES
#define _CONSTANTES

#include <math.h>

#define NB_DEGRES            2

#define _N                   64800 // Nombre de Variables
#define _K                   21600 // Nombre de Checks   
#define _M                   215999 // Nombre de Messages 

#define NOEUD   _N
#define MESSAGE _M

#define NmoinsK     (_N-_K)

#define DEG_1                10
#define DEG_2                9

#define DEG_1_COMPUTATIONS   21599
#define DEG_2_COMPUTATIONS   1

extern const unsigned int PosNoeudsVariable[_M];

#endif


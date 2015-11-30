#ifndef CONSTANTES
#define CONSTANTES

#include <math.h>

#define NB_DEGRES            2

#define _N                   64800 // Nombre de Variables
#define _K                   32400 // Nombre de Checks   
#define _M                   226799 // Nombre de Messages 

#define NmoinsK     (_N-_K)

#define DEG_1                7
#define DEG_2                6

#define DEG_1_COMPUTATIONS   32399
#define DEG_2_COMPUTATIONS   1

#define NB_ITERATIONS        30
#define NB_BITS_VARIABLES    8 //8
#define NB_BITS_MESSAGES     6 //6
#define SAT_POS_VAR  ( (0x0001<<(NB_BITS_VARIABLES-1))-1)
#define SAT_NEG_VAR  (-(0x0001<<(NB_BITS_VARIABLES-1))+1)
#define SAT_POS_MSG  ( (0x0001<<(NB_BITS_MESSAGES -1))-1)
#define SAT_NEG_MSG  (-(0x0001<<(NB_BITS_MESSAGES -1))+1)

#define BETA          0.15
#define FACTEUR_BETA  (0x0001<<(NB_BITS_MESSAGES/2))
#define BETA_FIX      ((int)(FACTEUR_BETA*BETA))

#define valeur_absolue(a)      ((a>0.00)?a:-a)
#define InversionCondSign(a,b) ((a==1)?b:-b)

#define _1D 0
#define _2D 1

#define _INTERLEAVED 1

extern const unsigned int PosNoeudsVariable[_M];

#endif


#ifndef CONSTANTES
#define CONSTANTES

#include <math.h>

#define NB_DEGRES            1

#define _N                   4896 // Nombre de Variables
#define _K                   2448 // Nombre de Checks   
#define _M                   14688 // Nombre de Messages 

#define NOEUD   _N
#define MESSAGE _M

#define NmoinsK     (_N-_K)

#define DEG_1                6

#define DEG_1_COMPUTATIONS   2448

#define NB_ITERATIONS        20
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


#ifndef CONSTANTES
#define CONSTANTES

#include <math.h>

#define cree_logs
#define gen_var_H_plasma
//#define AVC_BER
//#define test_gallager

#define largeur_bus 32 //en nb de bits

#define N                    1248 // Nombre de Variables
#define K                    624 // Nombre de Checks   
#define ONE_COUNT            3952 // Nombre de Messages 
#define nb_MAX_un_dans_ligne 7
#define nb_FU                24
#define optim_chrgt          0
#define nb_var_tot_par_ram   52
#define TAILLE_RAM_PNODE     8064 // = taille ram interleaver
#define TAILLE_VALID_RAM     1248
#define NB_CYCLE_EXEC        52

#define NmoinsK     (N-K)

#define NB_ITERATIONS        20
#define NB_BITS_VARIABLES    8 //8
#define NB_BITS_MESSAGES     6 //6
#define SAT_POS_VAR  ( (0x0001<<(NB_BITS_VARIABLES-1))-1)
#define SAT_NEG_VAR  (-(0x0001<<(NB_BITS_VARIABLES-1))+1)
#define SAT_POS_MSG  ( (0x0001<<(NB_BITS_MESSAGES -1))-1)
#define SAT_NEG_MSG  (-(0x0001<<(NB_BITS_MESSAGES -1))+1)

#define beta 0.15
#define FACTEUR_BETA (0x0001<<(NB_BITS_MESSAGES/2))
#define BETA_FIX ((int)(FACTEUR_BETA*beta))

#endif



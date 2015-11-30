#ifndef CONSTANTES_MANAGEMENT
#define CONSTANTES_MANAGEMENT

#include "code.h"

/*
	200x100
	
	1024x518
	1200x600
	1248x624
	2048x384
	16200x7560
	64800x21600
*/

#if CODE == 576
	#include "./576x288/constantes_gpu.h"
#endif

#if CODE == 816
	#include "./816x408/constantes_gpu.h"
#endif

#if CODE == 1024
	#include "./1024x518/constantes_gpu.h"
#endif

#if CODE == 1200
	#include "./1200x600/constantes_gpu.h"
#endif

#if CODE == 1944
	#include "./1944x972/constantes_gpu.h"
#endif

#if CODE == 2048
	#include "./2048x384/constantes_gpu.h"
#endif

#if CODE == 2304
	#include "./2304x1152/constantes_gpu.h"
#endif

#if CODE == 4000
	#include "./4000x2000/constantes_gpu.h"
#endif

#if CODE == 4896
	#include "./4896x2448/constantes_gpu.h"
#endif

#if CODE == 8000
	#include "./8000x4000/constantes_gpu.h"
#endif

#if CODE == 9972
	#include "./9972x4986/constantes_gpu.h"
#endif

#if CODE == 16200
	#include "./16200x7560/constantes_gpu.h"
#endif

#if CODE == 20000
	#include "./20000x10000/constantes_gpu.h"
#endif

#if CODE == 64800
	#include "./64800x32400/constantes_gpu.h"
#endif

#if CODE == 64801
	#include "./64800x21600/constantes_gpu.h"
#endif

#endif

// Includes
#include  <stdio.h>
#include  <stdlib.h>
#include  <iostream>
#include  <cstring>
//#include  <math.h>
//#include  <time.h>
#include  <string.h>
//#include  <limits.h>

#include <cuda.h>
#include <cuda_runtime.h>
//#include <builtin_types.h>
//#include <immintrin.h>
//#include <emmintrin.h>
//#include <xmmintrin.h>

using namespace std;

//#include "CFloodingGpuDecoder.h"

#include "./decoder_ms/CGPU_Decoder_MS_SIMD.h"
#include "./decoder_oms/CGPU_Decoder_OMS_SIMD.h"
#include "./decoder_nms/CGPU_Decoder_NMS_SIMD.h"
#include "./decoder_2nms/CGPU_Decoder_2NMS_SIMD.h"
#include "./decoder_oms_v2/CGPU_Decoder_MS_SIMD_v2.h"


//#define pi  3.1415926536

#include "./timer/CTimer.h"
#include "./trame/CTrame.h"
#include "./awgn_channel/CChanel_AWGN_SIMD.h"
#include "./ber_analyzer/CErrorAnalyzer.h"
#include "./terminal/CTerminal.h"

#include "./matrix/constantes_gpu.h"

//#define SINGLE_THREAD 1

int    QUICK_STOP           =  false;
bool   BER_SIMULATION_LIMIT =  false;
double BIT_ERROR_LIMIT      =  1e-7;

//int technique          = 0;
//int sChannel           = 1; // CHANNEL ON GPU

////////////////////////////////////////////////////////////////////////////////////

void show_info(){
	struct cudaDeviceProp devProp;
  	cudaGetDeviceProperties(&devProp, 0);
//  	printf("(II) Identifiant du GPU (CUDA)    : %s\n", devProp.name);
  	printf("(II) Nombre de Multi-Processor    : %d\n", devProp.multiProcessorCount);
  	printf("(II) + totalGlobalMem             : %ld Mo\n", (devProp.totalGlobalMem/1024/1024));
  	printf("(II) + sharedMemPerBlock          : %ld Ko\n", (devProp.sharedMemPerBlock/1024));
#ifdef CUDA_6
  	printf("(II) + sharedMemPerMultiprocessor : %ld Ko\n", (devProp.sharedMemPerMultiprocessor/1024));
  	printf("(II) + regsPerMultiprocessor      : %ld\n", devProp.regsPerMultiprocessor);
#endif
  	printf("(II) + regsPerBlock               : %d\n", (int)devProp.regsPerBlock);
  	printf("(II) + warpSize                   : %d\n", (int)devProp.warpSize);
  	printf("(II) + memoryBusWidth             : %d\n", (int)devProp.memoryBusWidth);
  	printf("(II) + memoryClockRate            : %d\n", (int)devProp.memoryClockRate);

  	{
  		struct cudaFuncAttributes attr;
  		cudaFuncGetAttributes(&attr, (const void*)LDPC_Sched_Stage_1_MS_SIMD);
  	  	printf("(II) CGPU_Decoder_MS_SIMD   (PTX version %d) : %2d regs, %4d shared bytes, %4d local bytes\n", attr.ptxVersion, attr.numRegs, (int)attr.localSizeBytes, (int)attr.sharedSizeBytes);
  	}
  	{
  		struct cudaFuncAttributes attr;
  		cudaFuncGetAttributes(&attr, (const void*)LDPC_Sched_Stage_1_OMS_SIMD);
  	  	printf("(II) CGPU_Decoder_OMS_SIMD   (PTX version %d) : %2d regs, %4d shared bytes, %4d local bytes\n", attr.ptxVersion, attr.numRegs, (int)attr.localSizeBytes, (int)attr.sharedSizeBytes);
  	}
  	{
  		struct cudaFuncAttributes attr;
  		cudaFuncGetAttributes(&attr, (const void*)LDPC_Sched_Stage_1_NMS_SIMD);
  	  	printf("(II) CGPU_Decoder_NMS_SIMD  (PTX version %d) : %2d regs, %4d shared bytes, %4d local bytes\n", attr.ptxVersion, attr.numRegs, (int)attr.localSizeBytes, (int)attr.sharedSizeBytes);
  	}
  	{
  		struct cudaFuncAttributes attr;
  		cudaFuncGetAttributes(&attr, (const void*)LDPC_Sched_Stage_1_2NMS_SIMD);
  	  	printf("(II) CGPU_Decoder_2NMS_SIMD (PTX version %d) : %2d regs, %4d shared bytes, %4d local bytes\n", attr.ptxVersion, attr.numRegs, (int)attr.localSizeBytes, (int)attr.sharedSizeBytes);
  	}
	fflush(stdout);
}

////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	int p;
    srand( 0 );
	printf("(II) LDPC DECODER - Flooding scheduled decoder\n");
	printf("(II) MANIPULATION DE DONNEES (IEEE-754 - %ld bits)\n", (long int)8*sizeof(int));
	printf("(II) GENEREE : %s - %s\n", __DATE__, __TIME__);

	double Eb_N0;
	double MinSignalSurBruit  = 0.50;
	double MaxSignalSurBruit  = 3.01;
	double PasSignalSurBruit  = 0.10;
    int    NOMBRE_ITERATIONS  = 20;
	int    STOP_TIMER_SECOND  = -1;
	bool   QPSK_CHANNEL       = false;
    bool   Es_N0              = false; // FALSE => MODE Eb_N0
    int    NB_THREAD_ON_GPU   = 1024;;
	int    FRAME_ERROR_LIMIT  =  200;

	char  defDecoder[] = "fMS";
    const char* type = defDecoder;

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaThreadSynchronize();

	//
	// ON VA PARSER LES ARGUMENTS DE LIGNE DE COMMANDE
	//
	for (p=1; p<argc; p++) {
		if( strcmp(argv[p], "-min") == 0 ){
			MinSignalSurBruit = atof( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-max") == 0 ){
			MaxSignalSurBruit = atof( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-pas") == 0 ){
			PasSignalSurBruit = atof( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-timer") == 0 ){
			STOP_TIMER_SECOND = atoi( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-iter") == 0 ){
			NOMBRE_ITERATIONS = atoi( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-fer") == 0 ){
			FRAME_ERROR_LIMIT = atoi( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-qef") == 0 ){
			BER_SIMULATION_LIMIT =  true;
			BIT_ERROR_LIMIT      = ( atof( argv[p+1] ) );
			p += 1;

		}else if( strcmp(argv[p], "-bpsk") == 0 ){
			QPSK_CHANNEL = false;

		}else if( strcmp(argv[p], "-qpsk") == 0 ){
			QPSK_CHANNEL = true;

		}else if( strcmp(argv[p], "-Eb/N0") == 0 ){
			Es_N0 = false;

		}else if( strcmp(argv[p], "-Es/N0") == 0 ){
			Es_N0 = true;

		}else if( strcmp(argv[p], "-n") == 0 ){
			NB_THREAD_ON_GPU = atoi( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-fMS") == 0 ){
			type      = "fMS";

		}else if( strcmp(argv[p], "-xMS") == 0 ){
			type      = "xMS";

		}else if( strcmp(argv[p], "-MS") == 0 ){
			type      = "MS";

		}else if( strcmp(argv[p], "-OMS") == 0 ){
			type      = "OMS";

		}else if( strcmp(argv[p], "-NMS") == 0 ){
			type      = "NMS";

		}else if( strcmp(argv[p], "-2NMS") == 0 ){
			type      = "2NMS";

		}else if( strcmp(argv[p], "-info") == 0 ){
			show_info();
			exit( 0 );

		}else{
			printf("(EE) Unknown argument (%d) => [%s]\n", p, argv[p]);
			exit(0);
		}
	}

	double rendement = (double)(NmoinsK)/(double)(_N);
	printf("(II) Code LDPC (N, K)     : (%d,%d)\n", _N, _K);
	printf("(II) Rendement du code    : %.3f\n", rendement);
	printf("(II) # ITERATIONs du CODE : %d\n", NOMBRE_ITERATIONS);
    printf("(II) FER LIMIT FOR SIMU   : %d\n", FRAME_ERROR_LIMIT);
	printf("(II) SIMULATION  RANGE    : [%.2f, %.2f], STEP = %.2f\n", MinSignalSurBruit,  MaxSignalSurBruit, PasSignalSurBruit);
	printf("(II) MODE EVALUATION      : %s\n", ((Es_N0)?"Es/N0":"Eb/N0") );
	printf("(II) MIN-SUM ALGORITHM    : %s\n", type );
	printf("(II) FAST STOP MODE       : %d\n", QUICK_STOP);

	CTimer simu_timer(true);


	//
	// ON CREE AUTANT DE TRAMES QUE L'ON A DE THREADS
	//
	CTrame simu_data(_N, _K, NB_THREAD_ON_GPU);

	CGPUDecoder* decoder;
	if( strcmp(type, "fMS") == 0 ){
			decoder = new CGPU_Decoder_MS_SIMD( NB_THREAD_ON_GPU, _N, _K, _M );
	}else if( strcmp(type, "MS") == 0 ){
		decoder = new CGPU_Decoder_MS_SIMD( NB_THREAD_ON_GPU, _N, _K, _M );
	}else if( strcmp(type, "OMS") == 0 ){
		decoder = new CGPU_Decoder_OMS_SIMD( NB_THREAD_ON_GPU, _N, _K, _M );
	}else if( strcmp(type, "NMS") == 0 ){
		decoder = new CGPU_Decoder_NMS_SIMD( NB_THREAD_ON_GPU, _N, _K, _M );
	}else if( strcmp(type, "2NMS") == 0 ){
		decoder = new CGPU_Decoder_2NMS_SIMD( NB_THREAD_ON_GPU, _N, _K, _M );
	}else if( strcmp(type, "xMS") == 0 ){
		decoder = new CGPU_Decoder_MS_SIMD_v2( NB_THREAD_ON_GPU, _N, _K, _M );
	}else{
		printf("(EE) Requested decoder does not exist !\n");
		exit( 0 );
	}
	decoder->initialize();


	CChanel_AWGN_SIMD noise(&simu_data, 4, QPSK_CHANNEL, Es_N0);


	Eb_N0 = MinSignalSurBruit;
	int temps = 0, fdecoding = 0;
	while (Eb_N0 <= MaxSignalSurBruit){

        //
        // ON CREE UN OBJET POUR LA MESURE DU TEMPS DE SIMULATION (REMISE A ZERO POUR CHAQUE Eb/N0)
        //
        CTimer temps_ecoule(true);
        CTimer term_refresh(true);

		noise.configure( Eb_N0 );

        CErrorAnalyzer errCounters(&simu_data, FRAME_ERROR_LIMIT, false, false);
        CErrorAnalyzer errCounter (&simu_data, FRAME_ERROR_LIMIT, true,  true);

        //
        // ON CREE L'OBJET EN CHARGE DES INFORMATIONS DANS LE TERMINAL UTILISATEUR
        //
		CTerminal terminal(&errCounters, &temps_ecoule, Eb_N0);

        //
        // ON GENERE LA PREMIERE TRAME BRUITEE
        //
        noise.generate();
        errCounter.store_enc_bits();

		while( 1 ){

			//
			//	ON LANCE LE TRAITEMENT SUR PLUSIEURS THREAD...
			//
			CTimer essai(true);
			decoder->decode( simu_data.get_t_noise_data(), simu_data.get_t_decode_data(), NOMBRE_ITERATIONS );
			temps += essai.get_time_ms();
			fdecoding += 1;
			#pragma omp sections
			{
				#pragma omp section
				{
					noise.generate();
				}
				#pragma omp section
				{
					errCounter.generate();
				}
			}

            //
			// ON COMPTE LE NOMBRE D'ERREURS DANS LA TRAME DECODE
            //
			errCounters.reset_internals();
			errCounters.accumulate( &errCounter );

            //
            // ON compare le Frame Error avec la limite imposee par l'utilisateur. Si on depasse
            // alors on affiche les resultats sur Eb/N0 courant.
            //
			if ( errCounters.fe_limit_achieved() == true ){
               break;
            }

            //
            // AFFICHAGE A L'ECRAN DE L'EVOLUTION DE LA SIMULATION SI NECESSAIRE
            //
			if( term_refresh.get_time_sec() >= 1 ){
				term_refresh.reset();
	           	terminal.temp_report();
			}

			if( (simu_timer.get_time_sec() >= STOP_TIMER_SECOND) && (STOP_TIMER_SECOND != -1) ){
        		printf("(II) THE SIMULATION HAS STOP DUE TO THE (USER) TIME CONTRAINT.\n");
        		printf("(II) PERFORMANCE EVALUATION WAS PERFORMED ON %d RUNS, TOTAL TIME = %dms\n", fdecoding, temps);
				temps /= fdecoding;
        		printf("(II) + TIME / RUN = %dms\n", temps);
        		int   workL = 4 * NB_THREAD_ON_GPU;
        		int   kbits = workL * _N / temps ;
        		float mbits = ((float)kbits) / 1000.0;
        		printf("(II) + DECODER LATENCY (ms)     = %d\n", temps);
        		printf("(II) + DECODER THROUGHPUT (Mbps)= %.1f\n", mbits);
        		printf("(II) + (%.2fdB, %dThd : %dCw, %dits) THROUGHPUT = %.1f\n", Eb_N0, NB_THREAD_ON_GPU, workL, NOMBRE_ITERATIONS, mbits);
				cout << endl << "Temps = " << temps << "ms : " << kbits;
				cout << "kb/s : " << ((float)temps/NB_THREAD_ON_GPU) << "ms/frame" << endl << endl;
        		break;
			}
		}

		terminal.final_report();

        if( (simu_timer.get_time_sec() >= STOP_TIMER_SECOND) && (STOP_TIMER_SECOND != -1) ){
        	break;
        }

		Eb_N0 = Eb_N0 + PasSignalSurBruit;

        if( BER_SIMULATION_LIMIT == true ){
        	if( errCounters.ber_value() < BIT_ERROR_LIMIT ){
        		printf("(II) THE SIMULATION HAS STOP DUE TO THE (USER) QUASI-ERROR FREE CONTRAINT.\n");
        		break;
        	}
        }
	}
    //printf("(II) Simulation is now terminated !\n");
	//delete decoder;
	//printf("(II) Simulation is now terminated !\n");
	return 0;
}

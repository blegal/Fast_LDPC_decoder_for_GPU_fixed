
#include "../decoder_template/CGPUDecoder.h"
#include "./cuda/CUDA_MS_SIMD.h"

class CGPU_Decoder_MS_SIMD : public CGPUDecoder{
public:
	CGPU_Decoder_MS_SIMD(size_t _nb_frames, size_t n, size_t k, size_t m );
    ~CGPU_Decoder_MS_SIMD();
    void initialize();
    void decode(float var_nodes[_N], int Rprime_fix[_N], int nombre_iterations);
};


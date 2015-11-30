
#include "../decoder_template/CGPUDecoder.h"

class CGPU_Decoder_MS_SIMD_v2 : public CGPUDecoder{
public:
	CGPU_Decoder_MS_SIMD_v2(size_t _nb_frames, size_t n, size_t k, size_t m );
    ~CGPU_Decoder_MS_SIMD_v2();
    void initialize();
    void decode(float var_nodes[_N], int Rprime_fix[_N], int nombre_iterations);
};


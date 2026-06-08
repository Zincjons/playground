// @file: ./task-1/src/xxx/f16-v2.cu
#include "playground/matmul.hpp"
#include "playground/utills.cuh"
#include <mma.h> // Tensor Core 必须引入的头文件

using namespace nvcuda; // WMMA API 所在的命名空间

namespace playground {

// Tensor Core 硬件指令通常采用 16x16x16 的分块大小
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
//每个warp分为16*16的块进行计算
//这里我有个疑问？我在K维度进行遍历时，切的每个块的大小一定要是WMMA_M WMMA_N WMMA_K吗，为什么不可以32*BK 和 BK*32去切，毕竟一个block只负责计算一个32*32的块

__global__ void matmul_f16_v2(const float16_t* A, const float16_t* B, float16_t* C, size_t M, size_t N, size_t K) {
    int warp_Id = threadIdx.y;//每个block负责计算32*32,一个block内四个warp，每个warp计算一个16*16
    int warp_Row = warp_Id / 2;
    int warp_Col = warp_Id % 2;


    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_c;

    wmma::fill_fragment(frag_c,__float2half(0.0f));

    int row = blockIdx.y * 32 + warp_Row * 16;//当前线程所在warp块的行号
    int col = blockIdx.x * 32 + warp_Col * 16;

    if(row < M && col < N){
        for(int split_K = 0; split_K < (K + WMMA_K - 1) / WMMA_K; split_K++){
            //搬运A中的对应块到fraga
            //当前是第几个K块
            wmma::load_matrix_sync(frag_a,(const half*)A + row * K + split_K * WMMA_K, K);
            //搬运B中的对应块到fragb
            wmma::load_matrix_sync(frag_b, (const half*)B + (split_K * WMMA_K) * K + col, N);
            //计算搬运到fragc
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }
        //最后fragc搬运回GM
        wmma::store_matrix_sync((half*)C + row * N + col, frag_c, N, wmma::mem_row_major);
        //这里有个问题，A不是被来就是float16_t吗，为什么还要手动转为(half*)?
    }



}
// 修复点：宏参数顺序改为 M, N, K, A, B, C
PLAYGROUND_MATMUL_DEC(float16_t, 2, M, N, K, A, B, C)
{
    // Block 配置：1 维排列 128 个线程 = 4 个 Warp
    dim3 block(32, 4); 
    // Grid 配置：按照 32x32 的尺度划分整个 M x N 矩阵
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    matmul_f16_v2<<<grid, block>>>(A, B, C, M, N, K);

    PLAYGOUND_CUDA_ERR_CHECK(cudaDeviceSynchronize());
}

}
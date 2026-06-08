// @file: ./task-1/src/xxx/f16-v3.cu
#include "playground/matmul.hpp"
#include "playground/utills.cuh"
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

namespace playground {

// Tensor Core 单次执行的硬件维度限制
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Block 级别的宏观 Tile 大小
const int BM = 64;
const int BN = 64;
const int BK = 32;

// 线程块的配置常量
const int BLOCK_DIM_X = 32; // 一个 Warp 的线程数
const int BLOCK_DIM_Y = 4;  // Warp 的数量
// const int T_NUM = BLOCK_DIM_X * BLOCK_DIM_Y; // 128

__global__ void matmul_f16_v3(const float16_t* A, const float16_t* B, float16_t* C, size_t M, size_t N, size_t K) {
    //一个block负责 BM * BN(64*64)的块的计算, 一个warp负责32*32
    // constexpr int warp_NUM = T_NUM / BLOCK_DIM_X; 
    int warp_Id = threadIdx.y;
    int warp_Row = warp_Id / 2;
    int warp_Col = warp_Id % 2;
    __shared__ half s_a[BM * (BK + 8)];
    __shared__ half s_b[BK * (BN + 8)];

    //当前block的首地址的行号，列号
    int row = blockIdx.y * BM;
    int col = blockIdx.x * BN;

    //单warp内的负责计算32*32所需的寄存器，以16*16为单位，所以
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a0, frag_a1;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b0, frag_b1;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_c00,frag_c01,frag_c10,frag_c11;
    
    wmma::fill_fragment(frag_c00, __float2half(0.0f));
    wmma::fill_fragment(frag_c01, __float2half(0.0f));
    wmma::fill_fragment(frag_c10, __float2half(0.0f));
    wmma::fill_fragment(frag_c11, __float2half(0.0f));
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    for(int split_K = 0; split_K < (K + BK -1) / BK; split_K++){
        //GM搬运到sa sb，这个和计算时的对应顺序不同，是将线程一维化，再进行均分，此处实现没有用ptx
        //每个线程搬 BM*BK/(32*4)
        #pragma unroll
        for(int ai = 0; ai < BM * BK; ai += 32 * 4 * 8){//8个half作为一个float4进行搬运
            int s_a_row = (ai + tid * 8) / BK;
            int s_a_col = (ai + tid * 8) % BK;

            float4 tmp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if(row + s_a_row < M && split_K * BK + s_a_col < K){
                tmp = (float4 &)A[(row + s_a_row)* K + split_K * BK + s_a_col];
            }
            (float4 &)s_a[s_a_row * (BK + 8) + s_a_col] = tmp;
        }
        #pragma unroll
        for(int bi =0; bi < BK * BN; bi += 32 * 4 * 8){
            int s_b_row = (bi + tid * 8) / BN;
            int s_b_col = (bi + tid * 8) % BN;

            float4 tmp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if(split_K * BK + s_b_row < K && col + s_b_col < N){
                tmp = (float4 &)B[(split_K * BK + s_b_row) * N + col + s_b_col];
            }
            (float4 &)s_b[s_b_row * (BN + 8) + s_b_col] = tmp;
        }

        __syncthreads();
        #pragma unroll
        for(int split_BK = 0; split_BK < 2; split_BK++){//
            //四个warp计算sa(64 * 32） sb32*64 所以每个warp，计算自己的warp_Id对应的32*32块，（该块由32*32与32*32的外积得到，），因此
            //把sa搬运到frag_a0,frag_a1
            wmma::load_matrix_sync(frag_a0,(const half*)s_a + warp_Row * 32 * (BK + 8) + split_BK * 16, BK + 8);//取出sa中的列向量的头部分
            wmma::load_matrix_sync(frag_a1,(const half*)s_a + (warp_Row * 32 + 16) * (BK + 8) + split_BK * 16, BK + 8);
            //把sb搬运到frag_b0,frag_b1
            wmma::load_matrix_sync(frag_b0, (const half*)s_b + (split_BK * 16 * (BN + 8)) + warp_Col * 32, BN + 8);
            wmma::load_matrix_sync(frag_b1, (const half*)s_b + split_BK * 16 * (BN + 8) + warp_Col * 32 + 16, BN + 8);

            //
            wmma::mma_sync(frag_c00, frag_a0, frag_b0, frag_c00);
            wmma::mma_sync(frag_c01, frag_a0, frag_b1, frag_c01);
            wmma::mma_sync(frag_c10, frag_a1, frag_b0, frag_c10);
            wmma::mma_sync(frag_c11, frag_a1, frag_b1, frag_c11);
        }
        //frag_a0, frag_a1, frag_b0, frag_b1计算
        __syncthreads();

    }
    //寄存器把当前warp的计算结果搬运回GM
    if(row + warp_Row * 32 < M && col + warp_Col * 32 < N)
        wmma::store_matrix_sync((half*)C + (row + warp_Row * 32) * N + col + warp_Col * 32, frag_c00, N, wmma::mem_row_major);
    if(row + warp_Row * 32 < M && col + warp_Col * 32 + 16 < N)
        wmma::store_matrix_sync((half*)C + (row + warp_Row * 32) * N + col + warp_Col * 32 + 16, frag_c01, N, wmma::mem_row_major); 
    if(row + warp_Row * 32 + 16 < M && col + warp_Col * 32 < N)
        wmma::store_matrix_sync((half*)C + (row + warp_Row * 32 + 16) * N + col + warp_Col * 32, frag_c10, N, wmma::mem_row_major);
    if(row + warp_Row * 32 +16  < M && col + warp_Col * 32 + 16 < N)
        wmma::store_matrix_sync((half*)C + (row + warp_Row * 32 + 16) * N + col + warp_Col * 32 + 16, frag_c11, N, wmma::mem_row_major);
    //
}

PLAYGROUND_MATMUL_DEC(float16_t, 3, M, N, K, A, B, C)
{
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y); 
    // Grid 计算公式通用化：(边界 + 块大小 - 1) / 块大小
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    matmul_f16_v3<<<grid, block>>>(A, B, C, M, N, K);
    
    PLAYGOUND_CUDA_ERR_CHECK(cudaDeviceSynchronize());
}

} // namespace playground
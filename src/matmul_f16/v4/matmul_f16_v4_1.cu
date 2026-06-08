// @file: ./task-1/src/xxx/f16-v3.cu
#include "playground/matmul.hpp"
#include "playground/utills.cuh"
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

namespace playground {

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// 128x128 的宏观分块
const int BM = 128;
const int BN = 128;
const int BK = 32;

const int BLOCK_DIM_X = 32;
const int BLOCK_DIM_Y = 8; // 8 个 Warp
const int THREAD_SUM = BLOCK_DIM_X * BLOCK_DIM_Y;

// Buffer 大小
const int SA_BUF_SIZE = BM * (BK + 8);
const int SB_BUF_SIZE = BK * (BN + 8);

__global__ void matmul_f16_v4(const float16_t* A, const float16_t* B, float16_t* C, size_t M, size_t N, size_t K) {
    int warp_id = threadIdx.y;
    int warp_Row = warp_id / 2;//块内warp行号
    int warp_Col = warp_id % 2;//块内warp列号

    int row = blockIdx.y * BM;//当前块首元素行号
    int col = blockIdx.x * BN;//当前块首元素列号

    extern __shared__ half s_m[];
    half* s_a = s_m;
    half* s_b = s_m + 3 * SA_BUF_SIZE;
    //8个warp，每个warp负责32 * 64的块的计算
    //由于寄存器只能按照16*16的单位进行申请
    //所以寄存器c 有c00 c01 c02 c03
    //            c10 c11 c12 c13
    //对应寄存器a 有a0 a1 (a在sharedmem中为128 * 32 对应每个warp分到的为32 * 16 且)
    //        b 有b0 b1 b2 b3
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a0, frag_a1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b0, frag_b1, frag_b2, frag_b3;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c00, frag_c01, frag_c02, frag_c03, frag_c10, frag_c11, frag_c12, frag_c13;
    
    wmma::fill_fragment(frag_c00, __float2half(0.0f));
    wmma::fill_fragment(frag_c01, __float2half(0.0f));
    wmma::fill_fragment(frag_c02, __float2half(0.0f));
    wmma::fill_fragment(frag_c03, __float2half(0.0f));
    wmma::fill_fragment(frag_c10, __float2half(0.0f));
    wmma::fill_fragment(frag_c11, __float2half(0.0f));
    wmma::fill_fragment(frag_c12, __float2half(0.0f));
    wmma::fill_fragment(frag_c13, __float2half(0.0f));

    //首块填入
    int split_K0 = 0;
    int split_K1 = 1;
    int a_shared_offset = 0;
    int b_shared_offset = 0;
    int a_shared1_offset = SA_BUF_SIZE;
    int b_shared1_offset = SB_BUF_SIZE;

    //搬运128*32的块从gm搬到a，搬运32*128的块从gm搬到b,第0块
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for(int a_id = 0; a_id < 128 * 32; a_id += (THREAD_SUM * 8)){//当前block的第一个开始搬运的线程在sa的首地址
        int sa_row = (a_id + tid * 8) / BK;
        int sa_col = (a_id + tid * 8) % BK;
        
        if(row + sa_row < M && split_K0 * BK + sa_col < K){
            uint32_t sa_tmp = static_cast<uint32_t>(__cvta_generic_to_shared(&s_a[a_shared_offset + sa_row * (BK + 8) + sa_col]));
            const void* gm_A_ptr = &A[(row + sa_row) * K + split_K0 * BK + sa_col];
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa_tmp), "l"(gm_A_ptr));
        }
        else{
            (float4&) s_a[a_shared_offset + sa_row * (BK + 8) + sa_col] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    for(int b_id = 0; b_id < 32* 128; b_id += (THREAD_SUM * 8)){
        int sb_row = (b_id + tid * 8) / BN;
        int sb_col = (b_id + tid * 8) % BN;

        if(split_K0 * BK + sb_row < K && col + sb_col < N){
            uint32_t sb_tmp = static_cast<uint32_t>(__cvta_generic_to_shared(&s_b[b_shared_offset + sb_row * (BN + 8) + sb_col]));
            const void* gm_B_ptr = &B[(split_K0 * BK + sb_row) * N + col + sb_col];
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sb_tmp), "l"(gm_B_ptr));
        }
        else{
            (float4&) s_b[b_shared_offset + sb_row * (BN + 8) + sb_col] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

    } 
    //K循环double buffer + 尾块计算，即每次K循环中都要判断下一块是否还有，如果有则继续搬下一块，并计算当前块
    //提交当前批次的所有DMA异步任务
    asm volatile(
        "cp.async.commit_group;\n" ::
    );

    //第1块
    for(int a_id = 0; a_id < 128 * 32; a_id += (THREAD_SUM * 8)){//当前block的第一个开始搬运的线程在sa的首地址
        int sa_row = (a_id + tid * 8) / BK;
        int sa_col = (a_id + tid * 8) % BK;
        
        if(row + sa_row < M && split_K1 * BK + sa_col < K){
            uint32_t sa_tmp = static_cast<uint32_t>(__cvta_generic_to_shared(&s_a[a_shared1_offset + sa_row * (BK + 8) + sa_col]));
            const void* gm_A_ptr = &A[(row + sa_row) * K + split_K1 * BK + sa_col];
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa_tmp), "l"(gm_A_ptr));
        }
        else{
            (float4&) s_a[a_shared1_offset + sa_row * (BK + 8) + sa_col] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    for(int b_id = 0; b_id < 32* 128; b_id += (THREAD_SUM * 8)){
        int sb_row = (b_id + tid * 8) / BN;
        int sb_col = (b_id + tid * 8) % BN;

        if(split_K1 * BK + sb_row < K && col + sb_col < N){
            uint32_t sb_tmp = static_cast<uint32_t>(__cvta_generic_to_shared(&s_b[b_shared1_offset + sb_row * (BN + 8) + sb_col]));
            const void* gm_B_ptr = &B[(split_K1 * BK + sb_row) * N + col + sb_col];
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sb_tmp), "l"(gm_B_ptr));
        }
        else{
            (float4&) s_b[b_shared1_offset + sb_row * (BN + 8) + sb_col] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

    } 
    //K循环double buffer + 尾块计算，即每次K循环中都要判断下一块是否还有，如果有则继续搬下一块，并计算当前块
    //提交当前批次的所有DMA异步任务
    asm volatile(
        "cp.async.commit_group;\n" ::
    );


    for(int split_K = 0; split_K < (K + BK - 1) / BK; split_K++){
        //若下下块可以搬运
        if(split_K + 2 < (K + BK - 1) / BK){
            int next_a_shared_offset = ((split_K + 2) % 3) * SA_BUF_SIZE;//因为absharedmem中内存大小一样，可以放在一起
            int next_b_shared_offset = ((split_K + 2) % 3) * SB_BUF_SIZE;
            for(int a_id = 0; a_id < 128 * 32; a_id += (THREAD_SUM * 8)){//当前block的第一个开始搬运的线程在sa的首地址
                int sa_row = (a_id + tid * 8) / BK;
                int sa_col = (a_id + tid * 8) % BK;
                
                if(row + sa_row < M && (split_K + 2) * BK + sa_col < K){
                    uint32_t sa_tmp = static_cast<uint32_t>(__cvta_generic_to_shared(&s_a[next_a_shared_offset + sa_row * (BK + 8) + sa_col]));
                    const void* gm_A_ptr = &A[(row + sa_row) * K + (split_K + 2) * BK + sa_col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa_tmp), "l"(gm_A_ptr));
                }
                else{
                    (float4&) s_a[next_a_shared_offset + sa_row * (BK + 8) + sa_col] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        
            for(int b_id = 0; b_id < 32* 128; b_id += (THREAD_SUM * 8)){
                int sb_row = (b_id + tid * 8) / BN;
                int sb_col = (b_id + tid * 8) % BN;
        
                if((split_K + 2) * BK + sb_row < K && col + sb_col < N){
                    uint32_t sb_tmp = static_cast<uint32_t>(__cvta_generic_to_shared(&s_b[next_b_shared_offset + sb_row * (BN + 8) + sb_col]));
                    const void* gm_B_ptr = &B[((split_K + 2) * BK + sb_row) * N + col + sb_col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sb_tmp), "l"(gm_B_ptr));
                }
                else{
                    (float4&) s_b[next_b_shared_offset + sb_row * (BN + 8) + sb_col] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
        
            } 

            asm volatile(
                "cp.async.commit_group;\n" ::
            );
        }

        if(split_K + 2 < (K + BK - 1) / BK){
            asm volatile("cp.async.wait_group 2;\n" ::);//
        }
        else if(split_K + 1 < (K + BK - 1) / BK){
            asm volatile("cp.async.wait_group 1;\n" ::);//
        }
        else{
            asm volatile("cp.async.wait_group 0;\n" ::);
        }

        __syncthreads();

        int now_a_shared_offset = (split_K % 3) * SA_BUF_SIZE;//
        int now_b_shared_offset = (split_K % 3) * SB_BUF_SIZE;

        //计算当前块
        for(int split_BK = 0; split_BK < 2; split_BK++){ //sa sb依次取出a的列块向量，和b的行块向量
            
            wmma::load_matrix_sync(frag_a0,(const half*)s_a + now_a_shared_offset + warp_Row * 32 * (BK + 8) + split_BK * 16, BK + 8);
            wmma::load_matrix_sync(frag_a1,(const half*)s_a + now_a_shared_offset + (warp_Row * 32 + 16) * (BK + 8) + split_BK * 16, BK + 8);
            wmma::load_matrix_sync(frag_b0, (const half*)s_b + now_b_shared_offset + (split_BK * 16) * (BN + 8) + warp_Col * 64, BN + 8);
            wmma::load_matrix_sync(frag_b1, (const half*)s_b + now_b_shared_offset + (split_BK * 16) * (BN + 8) + warp_Col * 64 + 16, BN + 8);
            wmma::load_matrix_sync(frag_b2, (const half*)s_b + now_b_shared_offset + (split_BK * 16) * (BN + 8) + warp_Col * 64 + 32, BN + 8);
            wmma::load_matrix_sync(frag_b3, (const half*)s_b + now_b_shared_offset + (split_BK * 16) * (BN + 8) + warp_Col * 64 + 48, BN + 8);

            wmma::mma_sync(frag_c00, frag_a0, frag_b0, frag_c00);
            wmma::mma_sync(frag_c01, frag_a0, frag_b1, frag_c01);
            wmma::mma_sync(frag_c02, frag_a0, frag_b2, frag_c02);
            wmma::mma_sync(frag_c03, frag_a0, frag_b3, frag_c03);
            wmma::mma_sync(frag_c10, frag_a1, frag_b0, frag_c10);
            wmma::mma_sync(frag_c11, frag_a1, frag_b1, frag_c11);
            wmma::mma_sync(frag_c12, frag_a1, frag_b2, frag_c12);
            wmma::mma_sync(frag_c13, frag_a1, frag_b3, frag_c13);
        }

        //为什么这个地方要加__syncthreads()
        __syncthreads();
    }
    
    
    wmma::store_matrix_sync((half*)C + (row + warp_Row * 32) * N + col + warp_Col * 64, frag_c00, N, wmma::mem_row_major);
    wmma::store_matrix_sync((half*)C + (row + warp_Row * 32) * N + col + warp_Col * 64 + 16, frag_c01, N, wmma::mem_row_major);
    wmma::store_matrix_sync((half*)C + (row + warp_Row * 32) * N + col + warp_Col * 64 + 32, frag_c02, N, wmma::mem_row_major);
    wmma::store_matrix_sync((half*)C + (row + warp_Row * 32) * N + col + warp_Col * 64 + 48, frag_c03, N, wmma::mem_row_major);
    wmma::store_matrix_sync((half*)C + (row + warp_Row * 32 + 16) * N + col + warp_Col * 64, frag_c10, N, wmma::mem_row_major);
    wmma::store_matrix_sync((half*)C + (row + warp_Row * 32 + 16) * N + col + warp_Col * 64 + 16, frag_c11, N, wmma::mem_row_major);
    wmma::store_matrix_sync((half*)C + (row + warp_Row * 32 + 16) * N + col + warp_Col * 64 + 32, frag_c12, N, wmma::mem_row_major);
    wmma::store_matrix_sync((half*)C + (row + warp_Row * 32 + 16) * N + col + warp_Col * 64 + 48, frag_c13, N, wmma::mem_row_major);
    //寄存器结果搬运回GM

    
    
}

PLAYGROUND_MATMUL_DEC(float16_t, 4, M, N, K, A, B, C)
{
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y); 
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    // 🚀 修改 2：计算动态 Shared Memory 的总字节数
    size_t shared_mem_bytes = (3 * SA_BUF_SIZE + 3 * SB_BUF_SIZE) * sizeof(half);

    // 🚀 修改 3：强行解锁驱动层面的 Shared Memory 上限（突破 48KB）
    PLAYGOUND_CUDA_ERR_CHECK(cudaFuncSetAttribute(
        matmul_f16_v4, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        shared_mem_bytes
    ));

    // 🚀 修改 4：在 Kernel 启动配置的第三个参数中传入动态大小
    matmul_f16_v4<<<grid, block, shared_mem_bytes>>>(A, B, C, M, N, K);
    
    PLAYGOUND_CUDA_ERR_CHECK(cudaDeviceSynchronize());
}

} // namespace playground
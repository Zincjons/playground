#include "playground/matmul.hpp"
#include "playground/utills.cuh"

namespace playground{

__global__ void matmul_f32_v2(const float32_t* A, const float32_t* B, float32_t* C, size_t M, size_t N, size_t K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    
    //知道每一个线程负责的C矩阵元素的对应索引
    if(row < M && col < N){
        float32_t sum = 0.0f;
        for(int k = 0; k < K; k++){
            sum += A[row * K + k] * B[k * N + col];
        }        
        C[row * N + col] = sum;
    }
}

PLAYGROUND_MATMUL_DEC(float32_t, 2, M, N, K, A, B, C)
{
    dim3 block(32, 32);
    //
    dim3 grid((N + block.x - 1) / block.x,(M + block.y - 1) / block.y);
    
    matmul_f32_v2<<<grid,block>>>(A, B, C, M, N, K);

    PLAYGOUND_CUDA_ERR_CHECK(cudaDeviceSynchronize());
}    

}


#include "playground/matmul.hpp"
#include "playground/utills.cuh"

namespace playground{

__global__ void matmul_f32_v3(const float32_t* A, const float32_t* B, float32_t* C, size_t M, size_t N, size_t K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    //这只是负责计算的C的元素位置
    
    constexpr int TILE_SIZE = 32;
    __shared__ float sa[TILE_SIZE][TILE_SIZE];//x是列而y是行
    __shared__ float sb[TILE_SIZE][TILE_SIZE];

    //知道每一个线程负责的C矩阵元素的对应索引
    //但是按照块去切分如果没有办法完全切分，比如说K不为32的倍数，如4097,那么多出来的维度，要再单独划分一个block去处理吗

    float32_t sum = 0.0f;
    for(int k_tile = 0; k_tile < (K + TILE_SIZE - 1) / TILE_SIZE; k_tile++){
        //先搬运,每个线程搬运block内自己的负责的元素，无论AB
        //添加越界判断，即一定可以访问到A或B的对应数据
        //A的行row
        //A的列k_tile * TILE_SIZE + threadIdx.x]
        //B的行(threadIdx.y + k_tile * TILE_SIZE)
        //B的列col
        if(k_tile * TILE_SIZE + threadIdx.x < K && row < M){
            sa[threadIdx.y][threadIdx.x] = A[row * K + k_tile * TILE_SIZE + threadIdx.x];
        }
        else{
            sa[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if(threadIdx.y + k_tile * TILE_SIZE < K && col < N){
            sb[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k_tile * TILE_SIZE) * N + col];
        }
        else{
            sb[threadIdx.y][threadIdx.x] = 0.0f;
        }

        //block内的线程同步
        __syncthreads();
        //再计算
        for(int k = 0; k < TILE_SIZE; k++){
            sum += sa[threadIdx.y][k] * sb[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < M && col < N){
        C[row * N + col] = sum;
    }

}

PLAYGROUND_MATMUL_DEC(float32_t, 3, M, N, K, A, B, C)
{
    dim3 block(32, 32);
    //
    dim3 grid((N + block.x - 1) / block.x,(M + block.y - 1) / block.y);
    
    matmul_f32_v3<<<grid,block>>>(A, B, C, M, N, K);

    PLAYGOUND_CUDA_ERR_CHECK(cudaDeviceSynchronize());
}    

}


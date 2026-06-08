#include "playground/matmul.hpp"
#include "playground/utills.cuh"

namespace playground{

__global__ void matmul_f32_v6(const float32_t* A, const float32_t* B, float32_t* C, size_t M, size_t N, size_t K){
    constexpr int BM = 128;
    constexpr int BK = 32;
    constexpr int BN = 128;
    constexpr int FRAG_C_SIZE = 8;
    //引入float4 量化，原本一个线程搬运16个，现在只用搬运 16/4 = 4 个
    constexpr int FLOAT4 = 4;

    __shared__ float sa[BM][BK];
    __shared__ float sb[BK][BN];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    float frag_c[FRAG_C_SIZE][FRAG_C_SIZE] = {0.0f};

    for(int k_Tile = 0; k_Tile < (K + BK - 1) / BK; k_Tile++){

        for(int copy_Num = 0; copy_Num < BM * BK / (blockDim.x * blockDim.y) / FLOAT4; copy_Num++){//每个线程搬运的float4的块的数量，当然这些块还是跳着搬的
            int thread_Ele_Id = copy_Num * blockDim.x * blockDim.y * FLOAT4 + tid * FLOAT4;//每个线程取的当前数据块的首地址
            int A_Row = blockIdx.y * BM + thread_Ele_Id / BK;//当前A块块首元素的行号 + 块内要搬运的float4块的首元素的行号，但这个是一次要搬运四个，这个该怎么写
            int A_Col = k_Tile * BK + thread_Ele_Id % BK;//当前A块块首元素的列号 + 块内要搬运元素的列号
            int B_Row = k_Tile * BK + thread_Ele_Id / BN;//B同理
            int B_Col = blockIdx.x * BN + thread_Ele_Id % BN;
            int A_Loc = A_Row * K + A_Col;
            int B_Loc = B_Row * N + B_Col;

            // 这一部分应该要怎么改成float4?
            float4 temp_a = reinterpret_cast<const float4*>(&A[A_Loc])[0];
            reinterpret_cast<float4*>(&sa[thread_Ele_Id / BK][thread_Ele_Id % BK])[0] = temp_a;

            float4 temp_b = reinterpret_cast<const float4*>(&B[B_Loc])[0];
            reinterpret_cast<float4*>(&sb[thread_Ele_Id / BN][thread_Ele_Id % BN])[0] = temp_b;

            // sa[thread_Ele_Id / BK][thread_Ele_Id % BK] = A[A_Loc];                
            // sb[thread_Ele_Id / BN][thread_Ele_Id % BN] = B[B_Loc];

        }
        __syncthreads();
        //计算部分
        for(int k = 0; k < BK; k++){//用矩阵外积去算，即k维度表示当前是sa的第k列，和sb的第k行
            //为什么线程会只截取属于的对应行号或列号的数据，因为k循环下，每个需要计算的数据都会被枚举到
            //当前线程负责的行号，threadIdx.y * 2 ,threadIdx.y * 2 + 1 ,列号threadIdx.x * 2 ,threadIdx.x * 2 + 1
            float frag_a[FRAG_C_SIZE];//这边的计算是要用常量宏吧
            float frag_b[FRAG_C_SIZE];
            #pragma unroll
            for(int i = 0;i < FRAG_C_SIZE; i++){
                frag_a[i] = sa[threadIdx.y * FRAG_C_SIZE + i][k];
                frag_b[i] = sb[k][threadIdx.x * FRAG_C_SIZE + i];
            }
            #pragma unroll
            for(int i = 0;i < FRAG_C_SIZE; i++){
                #pragma unroll
                for(int j = 0;j < FRAG_C_SIZE; j++){
                    frag_c[i][j] += frag_a[i] * frag_b[j];
                }
            }
        }

        __syncthreads();
    }

    //再将c的结果搬运回GM，要计算这四个点的对应GM地址，先算线程的对应行号和列号，再对应到矩阵C中负责的2*2矩阵的行列号上
    //threadIdx.y + blockIdx.y * blockDim.y 
    //threadIdx.x + blockIdx.x * blockDim.x

    int thread_Row = threadIdx.y + blockIdx.y * blockDim.y;
    int thread_Col = threadIdx.x + blockIdx.x * blockDim.x;
    #pragma unroll
    for(int i = 0; i < FRAG_C_SIZE; i++){
        #pragma unroll
        for(int j = 0; j < FRAG_C_SIZE; j++){
            if(thread_Row * FRAG_C_SIZE + i < M && thread_Col * FRAG_C_SIZE + j < N){
                C[(thread_Row * FRAG_C_SIZE + i)* N + thread_Col * FRAG_C_SIZE + j]=frag_c[i][j];
            }
        }
    }
}

PLAYGROUND_MATMUL_DEC(float32_t, 6, M, N, K, A, B, C)
{
    dim3 block(16, 16);
    //每个block负责计算128*128的矩阵块，一个线程负责计算一个8*8的矩阵块
    dim3 grid((N + 128 - 1) / 128,(M + 128 - 1) / 128);
    
    matmul_f32_v6<<<grid,block>>>(A, B, C, M, N, K);

    PLAYGOUND_CUDA_ERR_CHECK(cudaDeviceSynchronize());
}    

}


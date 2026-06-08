#include "playground/matmul.hpp"
#include "playground/utills.cuh"

namespace playground{

__global__ void matmul_f32_v4(const float32_t* A, const float32_t* B, float32_t* C, size_t M, size_t N, size_t K){
    
    constexpr int TILE_SIZE = 32;
    __shared__ float sa[TILE_SIZE][TILE_SIZE];
    __shared__ float sb[TILE_SIZE][TILE_SIZE];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    float frag_c[2][2] = {0.0f};//这个一定要初始化吗，会不会自动初始化为0

    for(int k_Tile = 0; k_Tile < (K + TILE_SIZE - 1) / TILE_SIZE; k_Tile++){
//线程搬运，将32*32的矩阵块搬运到sharedmem,sa,sb均需要搬运，每个线程负责搬运四个数据，将矩阵块索引拉长为一维，
//那么为什么要跳256再搬运
//是因为一个block里的32个线程同时搬运的数据最好地址连续，这样可以转化为一次访存指令，
//所以不用一个线程搬运连续的4个数据，而是每个线程搬运四次，
//每次32倍数的线程搬运地址连续的对应数据即可，
        for(int copy_Num = 0; copy_Num < 4; copy_Num++){//这个只是每个线程搬运的其中一个块
            int thread_Ele_Id = copy_Num * 256 + tid;//当前搬运的块内的线程对应的元素id
            int A_Row = blockIdx.y * TILE_SIZE + thread_Ele_Id / TILE_SIZE;//当前A块块首元素的行号 + 块内要搬运的元素的行号 
            int A_Col = k_Tile * TILE_SIZE + thread_Ele_Id % TILE_SIZE;//当前A块块首元素的列号 + 块内要搬运元素的列号
            int B_Row = k_Tile * TILE_SIZE + thread_Ele_Id / TILE_SIZE;//B同理
            int B_Col = blockIdx.x * TILE_SIZE + thread_Ele_Id % TILE_SIZE;
            int A_Loc = A_Row * K + A_Col;
            int B_Loc = B_Row * N + B_Col;
            if(A_Row < M && A_Col < K){
                sa[thread_Ele_Id / TILE_SIZE][thread_Ele_Id % TILE_SIZE] = A[A_Loc];                
            }
            else{
                sa[thread_Ele_Id / TILE_SIZE][thread_Ele_Id % TILE_SIZE] = 0.0f;
            }
            if(B_Row < K && B_Col < N){
                sb[thread_Ele_Id / TILE_SIZE][thread_Ele_Id % TILE_SIZE] = B[B_Loc];
            }
            else{
                sb[thread_Ele_Id / TILE_SIZE][thread_Ele_Id % TILE_SIZE] = 0.0f;
            }

            //这个地方我有个疑问，因为A_Row A_Col不会大于M K吗，而且我记得cuda的for循环里是不建议写if else分支的是吗
            //会导致线程分叉，即一个warp里有些线程在if分支里跑，另一部分满足else分支的，只能等这些线程跑完了，才再else分支里跑
            //但这个不可避免
        }
        __syncthreads();
        //计算部分
        for(int k = 0; k < TILE_SIZE; k++){//用矩阵外积去算，即k维度表示当前是sa的第k列，和sb的第k行
            //为什么线程会只截取属于的对应行号或列号的数据，因为k循环下，每个需要计算的数据都会被枚举到
            //当前线程负责的行号，threadIdx.y * 2 ,threadIdx.y * 2 + 1 ,列号threadIdx.x * 2 ,threadIdx.x * 2 + 1
            float frag_a[2];
            float frag_b[2];

            frag_a[0] = sa[threadIdx.y * 2][k];
            frag_a[1] = sa[threadIdx.y * 2 + 1][k];
            frag_b[0] = sb[k][threadIdx.x * 2];
            frag_b[1] = sb[k][threadIdx.x * 2 + 1];

            //取出每个线程要的数到寄存器中，再进行结果类加
            frag_c[0][0] += frag_a[0] * frag_b[0];
            frag_c[0][1] += frag_a[0] * frag_b[1];
            frag_c[1][0] += frag_a[1] * frag_b[0];
            frag_c[1][1] += frag_a[1] * frag_b[1];
        }

        __syncthreads();
    }

    //再将c的结果搬运回GM，要计算这四个点的对应GM地址，先算线程的对应行号和列号，再对应到矩阵C中负责的2*2矩阵的行列号上
    //threadIdx.y + blockIdx.y * blockDim.y 
    //threadIdx.x + blockIdx.x * blockDim.x

    int thread_Row = threadIdx.y + blockIdx.y * blockDim.y;
    int thread_Col = threadIdx.x + blockIdx.x * blockDim.x;
    if(thread_Row * 2 < M && thread_Col * 2 < N){
        C[thread_Row * 2 * N + thread_Col * 2]=frag_c[0][0];
    }
    if(thread_Row * 2 + 1 < M && thread_Col * 2 < N){
        C[(thread_Row * 2 + 1 )* N + thread_Col * 2]=frag_c[1][0];
    }
    if(thread_Row * 2 < M && thread_Col * 2 + 1 < N){
        C[thread_Row * 2 * N + thread_Col * 2 + 1]=frag_c[0][1];
    }
    if(thread_Row * 2 + 1 < M && thread_Col * 2 + 1 < N){
        C[(thread_Row * 2 + 1 ) * N + thread_Col * 2 + 1]=frag_c[1][1];
    }

}

PLAYGROUND_MATMUL_DEC(float32_t, 4, M, N, K, A, B, C)
{
    dim3 block(16, 16);
    //每个block负责计算32*32的矩阵块，一个线程负责计算一个2*2的矩阵块
    dim3 grid((N + 32 - 1) / 32,(M + 32 - 1) / 32);
    
    matmul_f32_v4<<<grid,block>>>(A, B, C, M, N, K);

    PLAYGOUND_CUDA_ERR_CHECK(cudaDeviceSynchronize());
}    

}


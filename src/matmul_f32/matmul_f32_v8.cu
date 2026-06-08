#include "playground/matmul.hpp"
#include "playground/utills.cuh"

namespace playground{

__global__ void matmul_f32_v8(const float32_t* A, const float32_t* B, float32_t* C, size_t M, size_t N, size_t K){
    constexpr int BM = 128;
    constexpr int BK = 32;
    constexpr int BN = 128;
    constexpr int FRAG_C_SIZE = 8;
    //引入float4 量化，原本一个线程搬运16个，现在只用搬运 16/4 = 4 个
    constexpr int FLOAT4 = 4;
    constexpr int BLOCKDIM_X = 16;
    constexpr int BLOCKDIM_Y = 16;
    constexpr int COPY_NUM = BM * BK / (BLOCKDIM_X * BLOCKDIM_Y) / FLOAT4;

    __shared__ float sa[BM][BK + 1];
    __shared__ float sb[BK][BN];
    
    int warpdim_y = 4;
    int warpdim_x = 8;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warpid = tid / 32;
    int warp_Row = warpid / (blockDim.x / warpdim_x); 
    int warp_Col = warpid % (blockDim.x / warpdim_x); 
    int thread_Ele_Row = warp_Row * warpdim_y + (tid % 32) / warpdim_x;//块内的元素行号
    int thread_Ele_Col = warp_Col * warpdim_x + (tid % 32) % warpdim_x;
    

    float frag_c[FRAG_C_SIZE][FRAG_C_SIZE] = {0.0f};//这个一定要初始化吗，会不会自动初始化为0

    

    for(int k_Tile = 0; k_Tile < (K + BK - 1) / BK; k_Tile++){
        //在此处要进行a矩阵的转置  128*32/ 16/ 16 /4
        #pragma unroll
        for(int copy_Num = 0; copy_Num < COPY_NUM; copy_Num++){//每个线程搬运的float4的块的数量，当然这些块还是跳着搬的
            int thread_Ele_Id = copy_Num * 1024 + tid * FLOAT4;//每个线程取的当前数据块的首地址
            int A_Row = blockIdx.y * BM + thread_Ele_Id / BK;//当前A块块首元素的行号 + 块内要搬运的float4块的首元素的行号，但这个是一次要搬运四个，这个该怎么写
            int A_Col = k_Tile * BK + thread_Ele_Id % BK;//当前A块块首元素的列号 + 块内要搬运元素的列号
            int A_Loc = A_Row * K + A_Col;

            //这个转置是指在将A的内容搬运到sharedmem上时，要存A矩阵对应块的转置
            float4 temp_a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (A_Row < M && A_Col + 3< K){
                temp_a = reinterpret_cast<const float4*>(&A[A_Loc])[0];
            }
            // reinterpret_cast<float4*>(&sa[thread_Ele_Id / BK][thread_Ele_Id % BK])[0] = temp_a;

            sa[thread_Ele_Id / BK][thread_Ele_Id % BK] = temp_a.x;
            sa[thread_Ele_Id / BK][thread_Ele_Id % BK + 1] = temp_a.y;
            sa[thread_Ele_Id / BK][thread_Ele_Id % BK + 2] = temp_a.z;
            sa[thread_Ele_Id / BK][thread_Ele_Id % BK + 3] = temp_a.w;
        }
        #pragma unroll
        for(int copy_Num = 0; copy_Num < COPY_NUM; copy_Num++){//每个线程搬运的float4的块的数量，当然这些块还是跳着搬的
            int thread_Ele_Id = copy_Num * 1024 + tid * FLOAT4;//每个线程取的当前数据块的首地址

            int B_Row = k_Tile * BK + thread_Ele_Id / BN;//B同理
            int B_Col = blockIdx.x * BN + thread_Ele_Id % BN;
            int B_Loc = B_Row * N + B_Col;
            //这个转置是指在将A的内容搬运到sharedmem上时，要存A矩阵对应块的转置
            float4 temp_b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if(B_Row < K && B_Col + 3< N){
                temp_b = reinterpret_cast<const float4*>(&B[B_Loc])[0];
            }
            reinterpret_cast<float4*>(&sb[thread_Ele_Id / BN][thread_Ele_Id % BN])[0] = temp_b;

        }
        __syncthreads();
        //计算部分
        for(int k = 0; k < BK; k++){//用矩阵外积去算，即k维度表示当前是sa的第k列，和sb的第k行
            float frag_a[FRAG_C_SIZE];
            float frag_b[FRAG_C_SIZE];
            #pragma unroll
            for(int i = 0;i < FRAG_C_SIZE; i++){
                frag_a[i] = sa[thread_Ele_Row * FRAG_C_SIZE + i][k];
            }
            #pragma unroll
            for(int i = 0;i < FRAG_C_SIZE; i++){
                //b float4  
                frag_b[i] = sb[k][thread_Ele_Col * FRAG_C_SIZE + i];
                // float4 temp_sb = reinterpret_cast<float4*>(&sb[k][thread_Ele_Col * FRAG_C_SIZE + i])[0];
                // reinterpret_cast<float4*>(&frag_b[i])[0] = temp_sb;
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

    int thread_Row = thread_Ele_Row + blockIdx.y * blockDim.y;
    int thread_Col = thread_Ele_Col + blockIdx.x * blockDim.x;
    #pragma unroll
    for(int i = 0; i < FRAG_C_SIZE; i++){
        int global_Row = thread_Row * FRAG_C_SIZE + i; 
        if(global_Row < M){

            #pragma unroll
            for(int j = 0; j < FRAG_C_SIZE; j += FLOAT4){
                int global_Col = thread_Col * FRAG_C_SIZE + j;
                if(global_Col + 3 < N){
                    float4 temp_c;
                    temp_c.x = frag_c[i][j];
                    temp_c.y = frag_c[i][j + 1];
                    temp_c.z = frag_c[i][j + 2];
                    temp_c.w = frag_c[i][j + 3];
                    reinterpret_cast<float4*>(&C[(thread_Row * FRAG_C_SIZE + i)* N + thread_Col * FRAG_C_SIZE + j])[0] = temp_c;
                }
                else{
                    #pragma unroll
                    for(int j_Epilogue = 0; global_Col + j_Epilogue < N; j_Epilogue++){
                        C[global_Row * N + global_Col + j_Epilogue] = frag_c[i][j + j_Epilogue];
                    }
                }
            }
        }

    }
}

PLAYGROUND_MATMUL_DEC(float32_t, 8, M, N, K, A, B, C)
{
    dim3 block(16, 16);
    //每个block负责计算128*128的矩阵块，一个线程负责计算一个8*8的矩阵块
    dim3 grid((N + 128 - 1) / 128,(M + 128 - 1) / 128);
    
    matmul_f32_v8<<<grid,block>>>(A, B, C, M, N, K);

    PLAYGOUND_CUDA_ERR_CHECK(cudaDeviceSynchronize());
}    

}


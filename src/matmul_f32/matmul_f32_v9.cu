#include "playground/matmul.hpp"
#include "playground/utills.cuh"

namespace playground {

__global__ void matmul_f32_v9_final(const float32_t* A, const float32_t* B, float32_t* C, int M, int N, int K) {
    // 严格匹配目标版参数
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int BK = 8; 
    constexpr int BM = 128; // blockDim_y in target
    constexpr int BN = 128; // blockDim_x in target

    int row = (blockIdx.y * 16) * TM; // 16 is blockDim.y，块首行号
    int col = (blockIdx.x * 16) * TN; // 16 is blockDim.x，块首列号

    // 采用一维打平的共享内存，消除多维索引计算开销
    __shared__ float S_A[BK * BM * 2]; 
    __shared__ float S_B[BK * BN * 2]; 

    float result[TM][TN] = {0.0f};
    float tmp_load[4]; // 用于中转的寄存器
    float tmp_S_A[TM];
    float tmp_S_B[TN];

    int tid = threadIdx.y * 16 + threadIdx.x;

    //负责搬运的块内行列号
    int S_A_Row = tid / (8 / 4);//块大小是128 * 8，一个block里的线程数256，每个线程负责搬运一个float4刚刚好
    int S_A_Col = tid % (8 / 4);
    int S_B_Row = tid / (128 / 4);//块大小是8 * 128
    int S_B_Col = tid % (128 / 4);
    
    //先放入K维度第一块的计算
    int split_K_0 = 0;
    //tmp_load用于float4读取GM中的A到sa
    (float4 &)(tmp_load[0]) = (float4 &)(A[(row + S_A_Row) * K + split_K_0 * BK + S_A_Col * 4]);
    //转置写回S_A
    #pragma unroll
    for(int i = 0; i < 4; i++){
        S_A[(S_A_Col * 4 + i) * BM + S_A_Row] = tmp_load[i];
    }
    //float4 读取GM中的B+写回到sb
    (float4 &)(S_B[S_B_Row * BN + S_B_Col * 4]) = (float4 &)(B[(split_K_0 * BK + S_B_Row) * N  + col + S_B_Col * 4]);

    //doublebuffer,在主循环的k维度中，针对sharedmem开启doublebuffer，这个doublebuffer该怎么开？ping_pong flag可以吗
    __syncthreads();
    
    for(int split_K = 1; split_K < (K + BK -1) / BK; split_K++){//算当前块，+搬运下一块
        //因为A转置了，所以读取sa sb的地址一样
        int s_Read_Offset = ((split_K - 1) % 2) * BM * BK;//BM == BN
        int s_Wirte_Offset = (split_K % 2) * BM * BK;

        //计算，将结果搬运到寄存器中，float4,每个线程计算自己负责的对应8*8的块,这里是不是不能用S_A_Row和S_B_Row等，因为这些是线程负责从GM搬运的sa sb块内行列号,
        //而计算时，是按照线程的block内的排列顺序分配的
        #pragma unroll
        for(int k = 0; k < BK; k++){//

            (float4 &)(tmp_S_A[0]) = (float4 &)(S_A[s_Read_Offset + k * BM + threadIdx.y * TM]);//S_A存的是转置
            (float4 &)(tmp_S_B[0]) = (float4 &)(S_B[s_Read_Offset + k * BN + threadIdx.x * TN]);
            (float4 &)(tmp_S_A[4]) = (float4 &)(S_A[s_Read_Offset + k * BM + threadIdx.y * TM + 4]);
            (float4 &)(tmp_S_B[4]) = (float4 &)(S_B[s_Read_Offset + k * BN + threadIdx.x * TN + 4]);

            #pragma unroll
            for(int TM_i = 0; TM_i < TM; TM_i++){
                #pragma unroll
                for(int TN_j = 0; TN_j < TN; TN_j++){
                    result[TM_i][TN_j] += tmp_S_A[TM_i] * tmp_S_B[TN_j];
                }
            }
        }

        //搬运下一块
        (float4 &)(tmp_load[0]) = (float4 &)(A[(row + S_A_Row) * K + split_K * BK + S_A_Col * 4]);
        #pragma unroll
        for(int i = 0; i < 4; i++){
            S_A[s_Wirte_Offset + (S_A_Col * 4 + i) * BM + S_A_Row] = tmp_load[i];
        }
        (float4 &)(S_B[s_Wirte_Offset + S_B_Row * BN + S_B_Col * 4]) = (float4 &)(B[(split_K * BK + S_B_Row) * N  + col + S_B_Col * 4]);
        __syncthreads();
    }

    //最后部分计算最后一块
    int s_Final_Read_Offset = (((K + BK - 1) / BK - 1) % 2) * BM * BK;

    #pragma unroll
    for(int k = 0; k < BK; k++){//

        (float4 &)(tmp_S_A[0]) = (float4 &)(S_A[s_Final_Read_Offset + k * BM + threadIdx.y * TM]);//S_A存的是转置
        (float4 &)(tmp_S_B[0]) = (float4 &)(S_B[s_Final_Read_Offset + k * BN + threadIdx.x * TN]);
        (float4 &)(tmp_S_A[4]) = (float4 &)(S_A[s_Final_Read_Offset + k * BM + threadIdx.y * TM + 4]);
        (float4 &)(tmp_S_B[4]) = (float4 &)(S_B[s_Final_Read_Offset + k * BN + threadIdx.x * TN + 4]);

        #pragma unroll
        for(int TM_i = 0; TM_i < TM; TM_i++){
            #pragma unroll
            for(int TN_j = 0; TN_j < TN; TN_j++){
                result[TM_i][TN_j] += tmp_S_A[TM_i] * tmp_S_B[TN_j];
            }
        }
    }

    __syncthreads();

    #pragma unroll
    for(int i = 0; i < TM; i++) {
        #pragma unroll
        for(int j = 0; j < TN; j++) {
            C[(row + threadIdx.y * TM + i) * N + col + threadIdx.x * TN + j] = result[i][j];
        }
    }
    
    //为什么不用float4搬运回，比以下用float4搬运回，的TFLOPS是18.39，若用float4搬运回，则为17.5

    // //最后将寄存器结果搬运回GM,此处用float4搬运回
    // int thread_Row = threadIdx.y + blockIdx.y * blockDim.y;
    // int thread_Col = threadIdx.x + blockIdx.x * blockDim.x;
    // #pragma unroll
    // for(int i = 0; i < TM; i++){
    //     #pragma unroll
    //     for(int j = 0; j < TN; j += 4){ // 注意这里是 += 4
    //         if(thread_Row * TM + i < M && thread_Col * TN + j + 3 < N){
    //             float4 temp_c = make_float4(result[i][j], result[i][j+1], result[i][j+2], result[i][j+3]);
    //             reinterpret_cast<float4*>(&C[(thread_Row * TM + i)* N + thread_Col * TN + j])[0] = temp_c;
    //         }
    //         // (注：为保持精简，这里忽略了矩阵边缘不能被4整除的残余处理，假设 M N 是 128 的倍数)
    //     }
    // }




    
}

PLAYGROUND_MATMUL_DEC(float32_t, 9, M, N, K, A, B, C) {
    dim3 block(16, 16);
    dim3 grid((N + 127) / 128, (M + 127) / 128);
    matmul_f32_v9_final<<<grid, block>>>(const_cast<float*>(A), const_cast<float*>(B), const_cast<float*>(C), M, N, K);
}

} // namespace playground
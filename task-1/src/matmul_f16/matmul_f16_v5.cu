#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define BM 256
#define BN 128
#define BK 32

#define SB_PADDING 8
#define SPLIT_K_STAGE 3
#define SPLIT_BK_STAGE 2

#define SA_BUFFER_SIZE 256 * 32
#define SB_BUFFER_SIZE 40 * 128

#define THREAD_SUM 256
namespace playground
{

__global__ void matmul_f16_v5(half *A, half *B, half *C, int M, int K, int N)
{
    int warp_Id = threadIdx.x >> 5;//
    int warp_Row = warp_Id >> 1;
    int warp_Col = warp_Id & 1;

    int Row = blockIdx.y * BM;//当前块首元素的行号
    int Col = blockIdx.x * BN;

    //共享内存triple buffer
    extern __shared__ half s_Mem[];
    half* s_a = s_Mem;
    half* s_b = s_Mem + SA_BUFFER_SIZE * 3;

    // //寄存器double buffer
    // uint32_t frag_a[2][4][4];//第一维 double buffer 第二维K维度有几个16*16的块（a为行维度），第三维（每个块内有几个uint32_t）
    // //64*32 其中K维度一次存储64*16,算上doublebuffer应该是[2][64][16],由于一个16*16的块为单位，且每个线程在16*16块内 16 * 16 / 32 = 8个half 用4个uint32_t
    // //所以维度为[2][64/16][4]
    // uint32_t frag_b[2][8][2];//第一维 double buffer 第二维K维度有几个16*8的块（b为列维度），第三维（每个块内有几个uint32_t）
    // //32 * 64 其中K维度一次存储16*64，算上doublebuffer应该是[2][64][16]，由于一个16*8的块为单位，且每个线程在16*8块内用2个uint32_t
    // //所以维度为[2][64/8][2]
    uint32_t frag_c[4][8][2];
    // //frag_c不需要doublebuffer 第一维 M维度有几个16*8的块 第二维N维度有几个16*8的块，第三维每个块内有几个uint32_t
    // //所以维度为[64/16][64/8][2]
    uint32_t frag_a0[4][4];
    uint32_t frag_a1[4][4];
    uint32_t frag_b0[8][2];
    uint32_t frag_b1[8][2];



    //triple buffer
    //先把第0块，第1块放入
    int split_K0 = 0;
    int split_K1 = 1;
    int s_a_Offset_K0 = 0;
    int s_a_Offset_K1 = SA_BUFFER_SIZE;
    int s_b_Offset_K0 = 0;
    int s_b_Offset_K1 = SB_BUFFER_SIZE;

    int tid = threadIdx.x; //块内线程编号
    //每个线程搬运256 * 32 / 32 / 8 = 32个half，一个float4为8个half所以，一个线程搬运4个float4
    #pragma unroll
    for(int float4_Id = 0; float4_Id < 4; float4_Id++){
        int sa_Id_Before_Swizzle = float4_Id * THREAD_SUM * 8 + tid * 8;
        int sa_Row_Before_Swizzle = sa_Id_Before_Swizzle / BK;//float4块的行号,用于异或，为了确保映射在0-3内，（一行最多只能容纳4个swizzle块）
        int sa_Col_Before_Swizzle = sa_Id_Before_Swizzle % BK;//这个列号是指float4块在原本sa中的列号，但不是floa4的块号，所以要除以8
        
        
        int sa_Col_Group_Swizzle = ((sa_Col_Before_Swizzle >> 3) + ((sa_Row_Before_Swizzle & 7) >> 1)) & 3;//swizzle后的块的组号还要 * 3 
        int sa_Col_Swizzle = (sa_Col_Group_Swizzle << 3) + (sa_Col_Before_Swizzle & 7);
        

        uint32_t sa_tmp = static_cast<uint32_t>(__cvta_generic_to_shared(&s_a[s_a_Offset_K0 + sa_Row_Before_Swizzle * BK + sa_Col_Swizzle]));//这个swizzle逻辑该怎么写
        const void* gm_A_ptr = &A[(Row + sa_Row_Before_Swizzle) * K + split_K0 * BK + sa_Col_Before_Swizzle];
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa_tmp), "l"(gm_A_ptr));

    }

    //每个线程搬运32*128/32/8 = 16个half,每个线程搬运2个float4,
    #pragma unroll
    for(int float4_Id = 0; float4_Id < 2; float4_Id++){ 
        int sb_Id = float4_Id * THREAD_SUM * 8 +tid * 8;
        int sb_Row = sb_Id / BN;
        int sb_Col = sb_Id % BN;

        uint32_t sb_tmp = static_cast<uint32_t>(__cvta_generic_to_shared(&s_b[s_b_Offset_K0 + sb_Row * (BN + 8) + sb_Col]));
        const void* gm_B_ptr = &B[(split_K0 * BK + sb_Row) * N + Col + sb_Col];
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sb_tmp), "l"(gm_B_ptr));

    }
    //搬运第一块

    asm volatile(
        "cp.async.commit_group;\n" ::
    );
    #pragma unroll
    for(int float4_Id = 0; float4_Id < 4; float4_Id++){
        int sa_Id_Before_Swizzle = float4_Id * THREAD_SUM * 8 + tid * 8;
        int sa_Row_Before_Swizzle = sa_Id_Before_Swizzle / BK;
        int sa_Col_Before_Swizzle = sa_Id_Before_Swizzle % BK;
        
        int sa_Col_Group_Swizzle = ((sa_Col_Before_Swizzle >> 3) + ((sa_Row_Before_Swizzle & 7) >> 1)) & 3;//swizzle后的块的组号还要 * 3 
        int sa_Col_Swizzle = (sa_Col_Group_Swizzle << 3) + (sa_Col_Before_Swizzle & 7);


        uint32_t sa_tmp = static_cast<uint32_t>(__cvta_generic_to_shared(&s_a[s_a_Offset_K1 + sa_Row_Before_Swizzle * BK + sa_Col_Swizzle]));
        const void* gm_A_ptr = &A[(Row + sa_Row_Before_Swizzle) * K + split_K1 * BK + sa_Col_Before_Swizzle];
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa_tmp), "l"(gm_A_ptr));

    }
    #pragma unroll
    for(int float4_Id = 0;float4_Id < 2; float4_Id++){
        int sb_Id = float4_Id * THREAD_SUM * 8 + tid * 8;
        int sb_Row = sb_Id / BN;
        int sb_Col = sb_Id % BN;


        uint32_t sb_tmp = static_cast<uint32_t>(__cvta_generic_to_shared(&s_b[s_b_Offset_K1 + sb_Row * (BN + 8) + sb_Col]));
        const void* gm_B_ptr = &B[(split_K1 * BK + sb_Row) * N + Col + sb_Col];
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sb_tmp), "l"(gm_B_ptr));

    }
    //
    //K循环double buffer + 尾块计算，即每次K循环中都要判断下一块是否还有，如果有则继续搬下一块，并计算当前块
    //提交当前批次的所有DMA异步任务
    asm volatile(
        "cp.async.commit_group;\n" ::
    );

    //还要提前搬运split_K0数据到寄存器中,所以要等split_K0先搬运到sa,sb中
    asm volatile(
        "cp.async.wait_group 1;\n" ::
    );

    __syncthreads();//很关键
    //提前搬运s_a中warp0a中的64*16的矩阵到frag_a[2][4][4]中的ping区域
    int lane_Id = tid & 31;
    //每个线程要搬运的half块在16*16的块内部的行号和列号
    int lane_a_Row = lane_Id & 15;//%16
    int lane_a_Col = lane_Id / 16 * 8;//
    #pragma unroll
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 8; j++){
            for(int k = 0; k < 2; k++){
                frag_c[i][j][k] = 0;
            }
        }
    }
    #pragma unroll
    for(int i = 0; i < 4; i++){//这个是循环搬运sa中的16*16的块到fraga中
        int sa_1616_Row_Per_Lane_Before_Swizzle = warp_Row * 64 + i * 16 + lane_a_Row;//取出的a的16*16块中的那一个由lane负责的8个half块在s_a中的行号
        int sa_1616_Col_Per_Lane_Before_Swizzle = lane_a_Col;//列号，为什么要有这个列号，是因为不同的lane，如lane0,和lane16负责的8*8的矩阵是不同的
        
        int sa_1616_Col_Inner_Before_Swizzle = sa_1616_Col_Per_Lane_Before_Swizzle & 7;
        int sa_1616_Col_Group_Swizzle = ((sa_1616_Col_Per_Lane_Before_Swizzle >> 3) + ((sa_1616_Row_Per_Lane_Before_Swizzle & 7) >> 1)) & 3;

        int sa_1616_Col_Per_Lane_Swizzle = (sa_1616_Col_Group_Swizzle << 3) + sa_1616_Col_Inner_Before_Swizzle;

        uint32_t sa_1616_Ptr_Per_Lane = static_cast<uint32_t>
        (__cvta_generic_to_shared(&s_a[s_a_Offset_K0 + sa_1616_Row_Per_Lane_Before_Swizzle * BK + sa_1616_Col_Per_Lane_Swizzle])); 

        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
            "{%0, %1, %2, %3}, [%4];\n"
            :"=r"(frag_a0[i][0]),
             "=r"(frag_a0[i][1]),
             "=r"(frag_a0[i][2]),
             "=r"(frag_a0[i][3])
            :"r"(sa_1616_Ptr_Per_Lane)
        );
    }
    //提前搬运s_b中warp0b中的16*64的矩阵到frag_b[2][8][2]中的ping区域

    //每个lane要搬运的half块在16*8的块内部的行号和列号
    int lane_b_Row = lane_Id & 15;//
    // int lane_b_Col = lane_Id / 16 * 4;
    //不需要每个lane的对应Col，传入时只需要这个lane负责的16*8的矩阵中的对应行的行首地址
    #pragma unroll
    for(int i = 0; i < 8; i++){//循环搬运sb中16*8块到fragb中
        int sb_168_Row_Per_Lane = lane_b_Row;  //取出的b的16*8块中的那一个由lane负责的4个half块在s_b中的行号
        // int sb_168_Col_Per_Lane = warp_Col * 64 + i * 8 + lane_b_Col;
        int sb_168_Col_Per_Lane = warp_Col * 64 + i * 8 ;

        uint32_t sb_168_Ptr_Per_Lane = static_cast<uint32_t>
        (__cvta_generic_to_shared(&s_b[s_b_Offset_K0 + sb_168_Row_Per_Lane * (BN + 8) + sb_168_Col_Per_Lane]));

        asm volatile(
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 "
            "{%0, %1}, [%2];\n"
            :"=r"(frag_b0[i][0]),
             "=r"(frag_b0[i][1])
            :"r"(sb_168_Ptr_Per_Lane)
        );
    }


    //split_K循环
    int K_Tiles = (K + BK - 1) / BK;
    for(int split_K = 0; split_K < K_Tiles - 2; split_K++){
        int sa_Stage0_Offset = (split_K % 3) * SA_BUFFER_SIZE;//stage0计算
        int sb_Stage0_Offset = (split_K % 3) * SB_BUFFER_SIZE;
        int sa_Stage1_Offset = ((split_K + 1) % 3) * SA_BUFFER_SIZE;//stage1用于split_BK的最后提前搬运下一个splitK的sa sb到fraga fragb,两级流水线的衔接点
        int sb_Stage1_Offset = ((split_K + 1) % 3) * SB_BUFFER_SIZE;//; 
        int sa_Stage2_Offset = ((split_K + 2) % 3) * SA_BUFFER_SIZE;//;//stage2,搬运下下个splitK的块，从gm到sa,sb
        int sb_Stage2_Offset = ((split_K + 2) % 3) * SB_BUFFER_SIZE;//;

        

        //搬运当前splitK块中的sa sb的下一半
        //搬运下一块到寄存器
        #pragma unroll
        for(int i = 0; i < 4; i++){//这个是循环搬运sa中的16*16的块到fraga中
            int sa_1616_Row_Per_Lane_Before_Swizzle = warp_Row * 64 + i * 16 + lane_a_Row;//取出的a的16*16块中的那一个由lane负责的8个half块在s_a中的行号
            int sa_1616_Col_Per_Lane_Before_Swizzle = 16 + lane_a_Col;//列号，为什么要有这个列号，是因为不同的lane，如lane0,和lane16负责的8*8的矩阵是不同的
            
            
            int sa_1616_Col_Inner_Before_Swizzle = sa_1616_Col_Per_Lane_Before_Swizzle & 7;
            int sa_1616_Col_Group_Swizzle = ((sa_1616_Col_Per_Lane_Before_Swizzle >> 3) + ((sa_1616_Row_Per_Lane_Before_Swizzle & 7) >> 1)) & 3;
    
            int sa_1616_Col_Per_Lane_Swizzle = (sa_1616_Col_Group_Swizzle << 3) + sa_1616_Col_Inner_Before_Swizzle;
    
            uint32_t sa_1616_Ptr_Per_Lane = static_cast<uint32_t>
            (__cvta_generic_to_shared(&s_a[sa_Stage0_Offset + sa_1616_Row_Per_Lane_Before_Swizzle * BK + sa_1616_Col_Per_Lane_Swizzle])); 
    
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "{%0, %1, %2, %3}, [%4];\n"
                :"=r"(frag_a1[i][0]),
                    "=r"(frag_a1[i][1]),
                    "=r"(frag_a1[i][2]),
                    "=r"(frag_a1[i][3])
                :"r"(sa_1616_Ptr_Per_Lane)
            );
        }
        #pragma unroll
        for(int i = 0; i < 8; i++){//循环搬运sb中16*8块到fragb中
            int sb_168_Row_Per_Lane = lane_b_Row + 16;  //取出的b的16*8块中的那一个由lane负责的4个half块在s_b中的行号
            // int sb_168_Col_Per_Lane = warp_Col * 64 + i * 8 + lane_b_Col;
            int sb_168_Col_Per_Lane = warp_Col * 64 + i * 8 ;
    
            uint32_t sb_168_Ptr_Per_Lane = static_cast<uint32_t>
            (__cvta_generic_to_shared(&s_b[sb_Stage0_Offset + sb_168_Row_Per_Lane * (BN + 8) + sb_168_Col_Per_Lane]));
    
            asm volatile(
                "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 "
                "{%0, %1}, [%2];\n"
                :"=r"(frag_b1[i][0]),
                    "=r"(frag_b1[i][1])
                :"r"(sb_168_Ptr_Per_Lane)
            );
        }
        //计算当前寄存器
        #pragma unroll
        for(int i = 0; i < 4; i++){
            #pragma unroll
            for(int j = 0; j < 8; j++){
                int j_S = (i & 1) ? (8 - j - 1) : j; 

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                    "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" //这里a是四个寄存器，b是两个寄存器，c也是两个寄存器，
                    : "=r"(frag_c[i][j_S][0]), "=r"(frag_c[i][j_S][1])
                    : "r"(frag_a0[i][0]), "r"(frag_a0[i][1]), "r"(frag_a0[i][2]), "r"(frag_a0[i][3]),
                      "r"(frag_b0[j_S][0]), "r"(frag_b0[j_S][1]),
                      "r"(frag_c[i][j_S][0]), "r"(frag_c[i][j_S][1])
                );
            }
        }

        //搬运后一块（判断能否搬）
        #pragma unroll
        for(int float4_Id = 0; float4_Id < 4; float4_Id++){
            int sa_Id_Before_Swizzle = float4_Id * THREAD_SUM * 8 + tid * 8;
            int sa_Row_Before_Swizzle = sa_Id_Before_Swizzle / BK;//float4块的行号,用于异或，为了确保映射在0-3内，（一行最多只能容纳4个swizzle块）
            int sa_Col_Before_Swizzle = sa_Id_Before_Swizzle % BK;//这个列号是指float4块在原本sa中的列号，但不是floa4的块号，所以要除以8
            
            
            
            int sa_Col_Group_Swizzle = ((sa_Col_Before_Swizzle >> 3) + ((sa_Row_Before_Swizzle & 7) >> 1)) & 3;//swizzle后的块的组号还要 * 3 
            int sa_Col_Swizzle = (sa_Col_Group_Swizzle << 3) + (sa_Col_Before_Swizzle & 7);
            
            uint32_t sa_tmp = static_cast<uint32_t>(__cvta_generic_to_shared(&s_a[sa_Stage2_Offset + sa_Row_Before_Swizzle * BK + sa_Col_Swizzle]));//这个swizzle逻辑该怎么写
            const void* gm_A_ptr = &A[(Row + sa_Row_Before_Swizzle) * K + (split_K + 2) * BK + sa_Col_Before_Swizzle];
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa_tmp), "l"(gm_A_ptr));

        }

        //每个线程搬运32*128/32/8 = 16个half,每个线程搬运2个float4,
        #pragma unroll
        for(int float4_Id = 0; float4_Id < 2; float4_Id++){ 
            int sb_Id = float4_Id * THREAD_SUM * 8 +tid * 8;
            int sb_Row = sb_Id / BN;
            int sb_Col = sb_Id % BN;

            uint32_t sb_tmp = static_cast<uint32_t>(__cvta_generic_to_shared(&s_b[sb_Stage2_Offset + sb_Row * (BN + 8) + sb_Col]));
            const void* gm_B_ptr = &B[((split_K + 2) * BK + sb_Row) * N + Col + sb_Col];
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sb_tmp), "l"(gm_B_ptr));

        }
        asm volatile(
            "cp.async.commit_group;\n" ::
        );

        //搬运下一个splitK块中的sa sb的前一半
        asm volatile(
            "cp.async.wait_group 1;\n" ::
        );//等下一块的已经被搬入了
        
        __syncthreads();//确保所有线程完成了上述的计算，要开始接下来的搬运了
        #pragma unroll
        for(int i = 0; i < 4; i++){//这个是循环搬运sa中的16*16的块到fraga中
            int sa_1616_Row_Per_Lane_Before_Swizzle = warp_Row * 64 + i * 16 + lane_a_Row;//取出的a的16*16块中的那一个由lane负责的8个half块在s_a中的行号
            int sa_1616_Col_Per_Lane_Before_Swizzle = lane_a_Col;//列号，为什么要有这个列号，是因为不同的lane，如lane0,和lane16负责的8*8的矩阵是不同的
            
            
            int sa_1616_Col_Inner_Before_Swizzle = sa_1616_Col_Per_Lane_Before_Swizzle & 7;
            int sa_1616_Col_Group_Swizzle = ((sa_1616_Col_Per_Lane_Before_Swizzle >> 3) + ((sa_1616_Row_Per_Lane_Before_Swizzle & 7) >> 1)) & 3;
    
            int sa_1616_Col_Per_Lane_Swizzle = (sa_1616_Col_Group_Swizzle << 3) + sa_1616_Col_Inner_Before_Swizzle;
    
            uint32_t sa_1616_Ptr_Per_Lane = static_cast<uint32_t>
            (__cvta_generic_to_shared(&s_a[sa_Stage1_Offset + sa_1616_Row_Per_Lane_Before_Swizzle * BK + sa_1616_Col_Per_Lane_Swizzle])); 
    
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "{%0, %1, %2, %3}, [%4];\n"
                :"=r"(frag_a0[i][0]),
                    "=r"(frag_a0[i][1]),
                    "=r"(frag_a0[i][2]),
                    "=r"(frag_a0[i][3])
                :"r"(sa_1616_Ptr_Per_Lane)
            );
        }
        #pragma unroll
        for(int i = 0; i < 8; i++){//循环搬运sb中16*8块到fragb中
            int sb_168_Row_Per_Lane = lane_b_Row;  //取出的b的16*8块中的那一个由lane负责的4个half块在s_b中的行号
            // int sb_168_Col_Per_Lane = warp_Col * 64 + i * 8 + lane_b_Col;
            int sb_168_Col_Per_Lane = warp_Col * 64 + i * 8 ;
    
            uint32_t sb_168_Ptr_Per_Lane = static_cast<uint32_t>
            (__cvta_generic_to_shared(&s_b[sb_Stage1_Offset + sb_168_Row_Per_Lane * (BN + 8) + sb_168_Col_Per_Lane]));
    
            asm volatile(
                "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 "
                "{%0, %1}, [%2];\n"
                :"=r"(frag_b0[i][0]),
                    "=r"(frag_b0[i][1])
                :"r"(sb_168_Ptr_Per_Lane)
            );
        }
        //计算当前寄存器
        #pragma unroll
        for(int i = 0; i < 4; i++){
            #pragma unroll
            for(int j = 0; j < 8; j++){
                int j_S = (i & 1) ? (8 - j - 1) : j; 

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                    "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" //这里a是四个寄存器，b是两个寄存器，c也是两个寄存器，
                    : "=r"(frag_c[i][j_S][0]), "=r"(frag_c[i][j_S][1])
                    : "r"(frag_a1[i][0]), "r"(frag_a1[i][1]), "r"(frag_a1[i][2]), "r"(frag_a1[i][3]),
                      "r"(frag_b1[j_S][0]), "r"(frag_b1[j_S][1]),
                      "r"(frag_c[i][j_S][0]), "r"(frag_c[i][j_S][1])
                );
            }
        }
        //计算当前块已经完成
    }
//================================================================================================================================================
    //尾块处理部分，因为将循环变为小于（K + BK - 1）/ BK - 2，将最后两块的splitK搬运拿到循环外面
    //倒数第二块，
    int sa_Stage0_Offset_Last1 = ((K_Tiles - 2) % 3) * SA_BUFFER_SIZE;//stage0计算
    int sb_Stage0_Offset_Last1 = ((K_Tiles - 2) % 3) * SB_BUFFER_SIZE;
    int sa_Stage1_Offset_Last1 = ((K_Tiles - 1) % 3) * SA_BUFFER_SIZE;//stage1用于split_BK的最后提前搬运下一个splitK的sa sb到fraga fragb,两级流水线的衔接点
    int sb_Stage1_Offset_Last1 = ((K_Tiles - 1) % 3) * SB_BUFFER_SIZE;//; 
    #pragma unroll
    for(int i = 0; i < 4; i++){//这个是循环搬运sa中的16*16的块到fraga中
        int sa_1616_Row_Per_Lane_Before_Swizzle = warp_Row * 64 + i * 16 + lane_a_Row;//取出的a的16*16块中的那一个由lane负责的8个half块在s_a中的行号
        int sa_1616_Col_Per_Lane_Before_Swizzle = 16 + lane_a_Col;//列号，为什么要有这个列号，是因为不同的lane，如lane0,和lane16负责的8*8的矩阵是不同的
        
        
        int sa_1616_Col_Inner_Before_Swizzle = sa_1616_Col_Per_Lane_Before_Swizzle & 7;
        int sa_1616_Col_Group_Swizzle = ((sa_1616_Col_Per_Lane_Before_Swizzle >> 3) + ((sa_1616_Row_Per_Lane_Before_Swizzle & 7) >> 1)) & 3;

        int sa_1616_Col_Per_Lane_Swizzle = (sa_1616_Col_Group_Swizzle << 3) + sa_1616_Col_Inner_Before_Swizzle;

        uint32_t sa_1616_Ptr_Per_Lane = static_cast<uint32_t>
        (__cvta_generic_to_shared(&s_a[sa_Stage0_Offset_Last1 + sa_1616_Row_Per_Lane_Before_Swizzle * BK + sa_1616_Col_Per_Lane_Swizzle])); 

        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
            "{%0, %1, %2, %3}, [%4];\n"
            :"=r"(frag_a1[i][0]),
                "=r"(frag_a1[i][1]),
                "=r"(frag_a1[i][2]),
                "=r"(frag_a1[i][3])
            :"r"(sa_1616_Ptr_Per_Lane)
        );
    }
    #pragma unroll
    for(int i = 0; i < 8; i++){//循环搬运sb中16*8块到fragb中
        int sb_168_Row_Per_Lane = lane_b_Row + 16;  //取出的b的16*8块中的那一个由lane负责的4个half块在s_b中的行号
        // int sb_168_Col_Per_Lane = warp_Col * 64 + i * 8 + lane_b_Col;
        int sb_168_Col_Per_Lane = warp_Col * 64 + i * 8 ;

        uint32_t sb_168_Ptr_Per_Lane = static_cast<uint32_t>
        (__cvta_generic_to_shared(&s_b[sb_Stage0_Offset_Last1 + sb_168_Row_Per_Lane * (BN + 8) + sb_168_Col_Per_Lane]));

        asm volatile(
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 "
            "{%0, %1}, [%2];\n"
            :"=r"(frag_b1[i][0]),
                "=r"(frag_b1[i][1])
            :"r"(sb_168_Ptr_Per_Lane)
        );
    }
    //计算当前寄存器
    #pragma unroll
    for(int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 8; j++){
            int j_S = (i & 1) ? (8 - j - 1) : j; 

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" //这里a是四个寄存器，b是两个寄存器，c也是两个寄存器，
                : "=r"(frag_c[i][j_S][0]), "=r"(frag_c[i][j_S][1])
                : "r"(frag_a0[i][0]), "r"(frag_a0[i][1]), "r"(frag_a0[i][2]), "r"(frag_a0[i][3]),
                  "r"(frag_b0[j_S][0]), "r"(frag_b0[j_S][1]),
                  "r"(frag_c[i][j_S][0]), "r"(frag_c[i][j_S][1])
            );
        }
    }

    //搬运下一个splitK块中的sa sb的前一半
    asm volatile(
        "cp.async.wait_group 0;\n" ::
    );//等下一块的已经被搬入了
    
    __syncthreads();//确保所有线程完成了上述的计算，要开始接下来的搬运了
    #pragma unroll
    for(int i = 0; i < 4; i++){//这个是循环搬运sa中的16*16的块到fraga中
        int sa_1616_Row_Per_Lane_Before_Swizzle = warp_Row * 64 + i * 16 + lane_a_Row;//取出的a的16*16块中的那一个由lane负责的8个half块在s_a中的行号
        int sa_1616_Col_Per_Lane_Before_Swizzle = lane_a_Col;//列号，为什么要有这个列号，是因为不同的lane，如lane0,和lane16负责的8*8的矩阵是不同的
        
        
        int sa_1616_Col_Inner_Before_Swizzle = sa_1616_Col_Per_Lane_Before_Swizzle & 7;
        int sa_1616_Col_Group_Swizzle = ((sa_1616_Col_Per_Lane_Before_Swizzle >> 3) + ((sa_1616_Row_Per_Lane_Before_Swizzle & 7) >> 1)) & 3;

        int sa_1616_Col_Per_Lane_Swizzle = (sa_1616_Col_Group_Swizzle << 3) + sa_1616_Col_Inner_Before_Swizzle;

        uint32_t sa_1616_Ptr_Per_Lane = static_cast<uint32_t>
        (__cvta_generic_to_shared(&s_a[sa_Stage1_Offset_Last1 + sa_1616_Row_Per_Lane_Before_Swizzle * BK + sa_1616_Col_Per_Lane_Swizzle])); 

        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
            "{%0, %1, %2, %3}, [%4];\n"
            :"=r"(frag_a0[i][0]),
                "=r"(frag_a0[i][1]),
                "=r"(frag_a0[i][2]),
                "=r"(frag_a0[i][3])
            :"r"(sa_1616_Ptr_Per_Lane)
        );
    }
    #pragma unroll
    for(int i = 0; i < 8; i++){//循环搬运sb中16*8块到fragb中
        int sb_168_Row_Per_Lane = lane_b_Row;  //取出的b的16*8块中的那一个由lane负责的4个half块在s_b中的行号
        // int sb_168_Col_Per_Lane = warp_Col * 64 + i * 8 + lane_b_Col;
        int sb_168_Col_Per_Lane = warp_Col * 64 + i * 8 ;

        uint32_t sb_168_Ptr_Per_Lane = static_cast<uint32_t>
        (__cvta_generic_to_shared(&s_b[sb_Stage1_Offset_Last1 + sb_168_Row_Per_Lane * (BN + 8) + sb_168_Col_Per_Lane]));

        asm volatile(
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 "
            "{%0, %1}, [%2];\n"
            :"=r"(frag_b0[i][0]),
                "=r"(frag_b0[i][1])
            :"r"(sb_168_Ptr_Per_Lane)
        );
    }
    //计算当前寄存器
    #pragma unroll
    for(int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 8; j++){
            int j_S = (i & 1) ? (8 - j - 1) : j; 

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" //这里a是四个寄存器，b是两个寄存器，c也是两个寄存器，
                : "=r"(frag_c[i][j_S][0]), "=r"(frag_c[i][j_S][1])
                : "r"(frag_a1[i][0]), "r"(frag_a1[i][1]), "r"(frag_a1[i][2]), "r"(frag_a1[i][3]),
                  "r"(frag_b1[j_S][0]), "r"(frag_b1[j_S][1]),
                  "r"(frag_c[i][j_S][0]), "r"(frag_c[i][j_S][1])
            );
        }
    }

    //计算最后一块
    int sa_Stage0_Offset_Last0 = ((K_Tiles - 1) % 3) * SA_BUFFER_SIZE;//stage0计算
    int sb_Stage0_Offset_Last0 = ((K_Tiles - 1) % 3) * SB_BUFFER_SIZE;


    //结果搬运回GM（中间经过sharedmem）
    #pragma unroll
    for(int i = 0; i < 4; i++){//这个是循环搬运sa中的16*16的块到fraga中
        int sa_1616_Row_Per_Lane_Before_Swizzle = warp_Row * 64 + i * 16 + lane_a_Row;//取出的a的16*16块中的那一个由lane负责的8个half块在s_a中的行号
        int sa_1616_Col_Per_Lane_Before_Swizzle = 16 + lane_a_Col;//列号，为什么要有这个列号，是因为不同的lane，如lane0,和lane16负责的8*8的矩阵是不同的
        
        
        int sa_1616_Col_Inner_Before_Swizzle = sa_1616_Col_Per_Lane_Before_Swizzle & 7;
        int sa_1616_Col_Group_Swizzle = ((sa_1616_Col_Per_Lane_Before_Swizzle >> 3) + ((sa_1616_Row_Per_Lane_Before_Swizzle & 7) >> 1)) & 3;

        int sa_1616_Col_Per_Lane_Swizzle = (sa_1616_Col_Group_Swizzle << 3) + sa_1616_Col_Inner_Before_Swizzle;

        uint32_t sa_1616_Ptr_Per_Lane = static_cast<uint32_t>
        (__cvta_generic_to_shared(&s_a[sa_Stage0_Offset_Last0 + sa_1616_Row_Per_Lane_Before_Swizzle * BK + sa_1616_Col_Per_Lane_Swizzle])); 

        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
            "{%0, %1, %2, %3}, [%4];\n"
            :"=r"(frag_a1[i][0]),
                "=r"(frag_a1[i][1]),
                "=r"(frag_a1[i][2]),
                "=r"(frag_a1[i][3])
            :"r"(sa_1616_Ptr_Per_Lane)
        );
    }
    #pragma unroll
    for(int i = 0; i < 8; i++){//循环搬运sb中16*8块到fragb中
        int sb_168_Row_Per_Lane = lane_b_Row + 16;  //取出的b的16*8块中的那一个由lane负责的4个half块在s_b中的行号
        // int sb_168_Col_Per_Lane = warp_Col * 64 + i * 8 + lane_b_Col;
        int sb_168_Col_Per_Lane = warp_Col * 64 + i * 8 ;

        uint32_t sb_168_Ptr_Per_Lane = static_cast<uint32_t>
        (__cvta_generic_to_shared(&s_b[sb_Stage0_Offset_Last0 + sb_168_Row_Per_Lane * (BN + 8) + sb_168_Col_Per_Lane]));

        asm volatile(
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 "
            "{%0, %1}, [%2];\n"
            :"=r"(frag_b1[i][0]),
                "=r"(frag_b1[i][1])
            :"r"(sb_168_Ptr_Per_Lane)
        );
    }
    //计算当前寄存器
    #pragma unroll
    for(int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 8; j++){
            int j_S = (i & 1) ? (8 - j - 1) : j; 

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" //这里a是四个寄存器，b是两个寄存器，c也是两个寄存器，
                : "=r"(frag_c[i][j_S][0]), "=r"(frag_c[i][j_S][1])
                : "r"(frag_a0[i][0]), "r"(frag_a0[i][1]), "r"(frag_a0[i][2]), "r"(frag_a0[i][3]),
                  "r"(frag_b0[j_S][0]), "r"(frag_b0[j_S][1]),
                  "r"(frag_c[i][j_S][0]), "r"(frag_c[i][j_S][1])
            );
        }
    }

    __syncthreads();//确保所有线程完成了上述的计算，要开始接下来的最后一块的计算了

    //计算当前块的最后一半寄存器，此时已经是最后一块，不需要继续搬运了
    #pragma unroll
    for(int i = 0; i < 4; i++){
        #pragma unroll
        for(int j = 0; j < 8; j++){
            int j_S = (i & 1) ? (8 - j - 1) : j; 

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" //这里a是四个寄存器，b是两个寄存器，c也是两个寄存器，
                : "=r"(frag_c[i][j_S][0]), "=r"(frag_c[i][j_S][1])
                : "r"(frag_a1[i][0]), "r"(frag_a1[i][1]), "r"(frag_a1[i][2]), "r"(frag_a1[i][3]),
                  "r"(frag_b1[j_S][0]), "r"(frag_b1[j_S][1]),
                  "r"(frag_c[i][j_S][0]), "r"(frag_c[i][j_S][1])
            );
        }
    }

    //frag_c搬回sm最后回到gm
    //====================================================================================================================================
    half* s_c = s_Mem;

    //将fragc搬运到s_c
    #pragma unroll
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 8; j++){
            //遍历frag_c中的每一个16*8矩阵
            int fragc_168_Row = lane_Id / 4;
            int s_c_Swizzle_Offset = fragc_168_Row * 8;
            int fragc_168_Col = (lane_Id % 4) * 2 + s_c_Swizzle_Offset;
            //搬运上一半
            *(uint32_t*)&s_c[(warp_Row * 64 + i * 16 + fragc_168_Row) * BN + (warp_Col * 64 + j * 8 + fragc_168_Col) % BN] = frag_c[i][j][0];
            //搬运下一半
            *(uint32_t*)&s_c[(warp_Row * 64 + i * 16 + fragc_168_Row + 8) * BN + (warp_Col * 64 + j * 8 + fragc_168_Col) % BN] = frag_c[i][j][1]; 
        }
    }
    //每个lane搬运16*8矩阵中自己的寄存器内容
    __syncthreads();

    //将s_c搬运回gm,
    #pragma unroll
    for(int float4_Id = 0; float4_Id < 16; float4_Id++){//每个线程搬运16个float4
        //但是因为s_c中存储的是swizzle后的数据所以在算出逻辑行和列后，要再swizzle找到实际物理存储地址
        int sc_Id_Before_Swizzle = float4_Id * THREAD_SUM * 8 + tid * 8;
        int sc_Row_Before_Swizzle = sc_Id_Before_Swizzle / BN;
        int sc_Col_Before_Swizzle = sc_Id_Before_Swizzle % BN;
        int sc_Col_Swizzle = (sc_Col_Before_Swizzle + (sc_Row_Before_Swizzle % 8) * 8) % BN;

        if(Row + sc_Row_Before_Swizzle < M && Col + sc_Col_Before_Swizzle < N){
            (float4&)C[(Row + sc_Row_Before_Swizzle) * N + Col + sc_Col_Before_Swizzle] = (float4&)s_c[sc_Row_Before_Swizzle * BN + sc_Col_Swizzle]; 
        }

    }


}

PLAYGROUND_MATMUL_DEC(float16_t, 5, M, N, K, A, B, C)
{
    const int BLOCK_DIM_y = 1;
    const int BLOCK_DIM_x = 256;

    int num_Block_Y = (M + BM - 1) / BM;
    int num_Block_X = (N + BN - 1) / BN;

    dim3 block_Dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_Dim(num_Block_X, num_Block_Y, 1);
    // int buffer_num = 2;
    size_t smem_Max_Size = std::max((BM * BK + BN * (BK +  SB_PADDING)) * sizeof(half) * SPLIT_K_STAGE, BM * BN * sizeof(half));
    cudaFuncSetAttribute(matmul_f16_v5, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_Max_Size);
    
    matmul_f16_v5<<<grid_Dim, block_Dim, smem_Max_Size>>>(
        const_cast<float16_t*>(A), 
        const_cast<float16_t*>(B), 
        const_cast<float16_t*>(C), 
        M, K, N
    );

    cudaDeviceSynchronize();
}

}  // namespace playground
#include <cuda.h>
#include <stdio.h>
#include <chrono>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define M 16   // anything
#define N 16   // anything
#define K 16   // anything

// unset for no print final comparison
#define DEBUG 1

// enable/disable CPU matmul
#define ENABLECPU 1

__global__ void wmma_kernel(half *a, half *b, float *c) {
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);

   // Load the inputs
   wmma::load_matrix_sync(a_frag, a, 16);
   wmma::load_matrix_sync(b_frag, b, 16);

   // Perform the matrix multiplication
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   // Store the output
   wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}

__global__ void float2half(float* float_array, half* half_array, int num_elements){
    int globalThreadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    half_array[globalThreadIndex] = (half)float_array[globalThreadIndex];
}

__global__ void half2float(half* half_array, float* float_array, int num_elements){
    int globalThreadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    float_array[globalThreadIndex] = (float)half_array[globalThreadIndex];
}

__global__ void doNothingKernel(){
    return;
}

int main(){
    doNothingKernel<<<1,1>>>();
    printf("[M] = %d\n[N] = %d\n[K] = %d\n\n",M,N,K);

    double number_of_flops = (double)M*(double)N*(2.0*(double)K-1.0) + (double) M * (double) N;
    printf("Number of FLOPS: %lf\n", number_of_flops);

    float* matrixA;
    float* matrixB;
    float* matrixC;
    float* matrixOut;
    float alpha, beta;

    alpha = 1.0f;
    beta = 1.0f;

    matrixA     = (float*)malloc(sizeof(float) * M * K); printf("matrixA allocated with space for %d floats\n", M * K);
    matrixB     = (float*)malloc(sizeof(float) * K * N); printf("matrixB allocated with space for %d floats\n", K * N);
    matrixC     = (float*)malloc(sizeof(float) * M * N); printf("matrixC allocated with space for %d floats\n", M * N);
    matrixOut   = (float*)malloc(sizeof(float) * M * N); printf("matrixOut allocated with space for %d floats\n", M * N);

    srand(0);

    // intialise matrixA
    printf("Setting matrixA...\n");
    for (int i = 0; i < M; i++){
        for (int j = 0; j < K; j++){
            matrixA[i * K + j] = (float)rand()/(float)RAND_MAX;
            printf("%f ", matrixA[i * K + j]);
        }
        printf("\n");
    }

    // initialise matrixB
    printf("Setting matrixB...\n");
    for (int i = 0; i < K; i++){
        for (int j = 0; j < N; j++){
            matrixB[i * N + j] = (float)rand()/(float)RAND_MAX;
            printf("%f ", matrixB[i * N + j]);
        }
        printf("\n");
    }

    // initialise matrixC
    printf("Setting matrixC...\n");
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            matrixC[i * N + j] = 0.0f;
            printf("%f ", matrixC[i * M + j]);
        }
        printf("\n");
    }

    // set output to 0
    memset(matrixOut, 0, sizeof(float) * M * N);
    printf("matrixOut set to all zeroes\n");

    // do the matrix multiply

#ifdef ENABLECPU
    auto start_cpu_mmul = std::chrono::high_resolution_clock::now();

    // step through rows of the output
    for (int i = 0; i < M; i++){
        // step through columns of the output
        for (int j = 0; j < N; j++){
            float tempSum = 0.0f;
            for (int step = 0; step < K; step++){
                // step through row[i] of matrixA
                // step through column[j] of matrixB
                //printf("Accessing element %d of matrixA[%d] and element %d of matrixB[%d]\n", i*K + step, M * K, step * N + j, K * N);
                tempSum += matrixA[i*K + step] * matrixB[step * N + j] * alpha;
            }
            // add [i][j] of matrixC
            tempSum += beta * matrixC[i * N + j];
            matrixOut[i * N + j] = tempSum;
            //printf("Accessing element %d of matrixOut[%d]\n", i * N + j, M * N);
        }
    }

    auto stop_cpu_mmul = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_cpu_mmul = stop_cpu_mmul - start_cpu_mmul;
    printf("CPU MMUL took %f ms @ %lf GFLOPs/s\n", duration_cpu_mmul.count(), number_of_flops / ((double) duration_cpu_mmul.count()/1000) / 1024 / 1024 / 1024);
#endif

    float* GPU_matrixA;
    float* GPU_matrixB;
    float* GPU_matrixC;
    half* GPU_matrixA_half;
    half* GPU_matrixB_half;
    float* hostCheck_matrixOut;

    cudaError_t errorCheckVariable;

    errorCheckVariable = cudaMalloc((void**)&GPU_matrixA,   sizeof(float) * M * K);
    errorCheckVariable = cudaMalloc((void**)&GPU_matrixB,   sizeof(float) * K * N);
    errorCheckVariable = cudaMalloc((void**)&GPU_matrixC,   sizeof(float) * M * N);

    errorCheckVariable = cudaMalloc((void**)&GPU_matrixA_half,   sizeof(half) * M * K);
    errorCheckVariable = cudaMalloc((void**)&GPU_matrixB_half,   sizeof(half) * K * N);

    printf("\nSetting hostCheck_matrixOut to all zeroes\n");
    hostCheck_matrixOut = (float*)malloc(sizeof(float) * M * N);
    printf("\nSet hostCheck_matrixOut to all zeroes\n");

    errorCheckVariable = cudaMemcpy(GPU_matrixA, matrixA,       sizeof(float) * M * K, cudaMemcpyHostToDevice);
    errorCheckVariable = cudaMemcpy(GPU_matrixB, matrixB,       sizeof(float) * K * N, cudaMemcpyHostToDevice);
    errorCheckVariable = cudaMemcpy(GPU_matrixC, matrixC,       sizeof(float) * M * N, cudaMemcpyHostToDevice);

    int numThreadsCast;
    int numBlocksCast;
    numThreadsCast = 1024;
    
    // cast GPU_matrixA into GPU_matrixA_half
    numBlocksCast = (M * K + numThreadsCast - 1) / numThreadsCast;
    float2half<<<numBlocksCast,numThreadsCast>>>(GPU_matrixA, GPU_matrixA_half, M*K);

    // cast GPU_matrixB into GPU_matrixB_half
    numBlocksCast = (K * N + numThreadsCast - 1) / numThreadsCast;
    float2half<<<numBlocksCast,numThreadsCast>>>(GPU_matrixB, GPU_matrixB_half, K*N);

    int blockSizeX = 16;
    int blockSizeY = 16;
    dim3 blockSize =    dim3(blockSizeX,blockSizeY);

    int gridSizeX = ceil((double) N / (double) blockSizeX);
    int gridSizeY = ceil((double) M / (double) blockSizeY);

    dim3 gridSize =     dim3(gridSizeX,gridSizeY);

    printf("blockSize   = [%d,  %d]\n", blockSizeX, blockSizeY);
    printf("gridSize    = [%d,  %d]\n", gridSizeX, gridSizeY);

    cudaEvent_t start_mmul, stop_mmul;
    float milliseconds_mmul, seconds_mmul;
    cudaEventCreate(&start_mmul);
    cudaEventCreate(&stop_mmul);

    double nRepeats = 10.0;

    // wmma Kernel
    cudaEventRecord(start_mmul);
    for (double i = 0; i < nRepeats; i+=1.0) wmma_kernel<<<gridSize,blockSize>>>(GPU_matrixA_half, GPU_matrixB_half, GPU_matrixC);

    cudaDeviceSynchronize();
    cudaEventRecord(stop_mmul);
    cudaEventSynchronize(stop_mmul);
    cudaEventElapsedTime(&milliseconds_mmul, start_mmul, stop_mmul);
    seconds_mmul = milliseconds_mmul/ 1000;
    printf("GPU MMUL wmma took %f ms @ %lf GFLOPS/s\n", milliseconds_mmul, number_of_flops * nRepeats / (double) seconds_mmul / 1024 / 1024 / 1024 );

    // copy output back to host
    cudaMemcpy(hostCheck_matrixOut, GPU_matrixC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    float maxDiscrepancy = 0.0f;
    float tempDiscrepancy = 0.0f;

    // print matrixOut
    printf("\nmatrixOut:\n");
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            tempDiscrepancy = matrixOut[i * N + j] - hostCheck_matrixOut[i * N + j];
#ifdef DEBUG
            printf("matrixC[%d][%d]: %f vs ",i,j, matrixOut[i * N + j]);
            printf("%f\n", hostCheck_matrixOut[i * N + j]);
#endif
            if (abs(tempDiscrepancy) > maxDiscrepancy){
                maxDiscrepancy = abs(tempDiscrepancy);
            }
        }
    }

    printf("maxDiscrepancy = %f\n", maxDiscrepancy);
}

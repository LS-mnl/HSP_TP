#include <stdio.h>
#include <stdlib.h>

void MatrixInit(float *M, int n, int p) {
    for (int i = 0; i < n * p; i++) {
        M[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

void MatrixPrint(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%1.1f ", M[i * p + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    for (int i = 0; i < n * p; i++) {
        Mout[i] = M1[i] + M2[i];
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    int idx = threadIdx.x * blockDim.y + threadIdx.y;
    if (idx < n * p) {
        Mout[idx] = M1[idx] + M2[idx];
    }
}

void MatrixMult(float *M1, float *M2, float *Mout, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int k = 0; k < n; k++) {
                s += M1[i * n + k] * M2[k * n + j];
            }
            Mout[i * n + j] = s;
        }
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float val = 0;
        for (int k = 0; k < n; ++k) {
            val += M1[row * n + k] * M2[k * n + col];
        }
        Mout[row * n + col] = val;
    }
}

__global__ void cudaMatrixMultGeneral(float *M1, float *M2, float *Mout, int n, int p, int m) {
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float s = 0.0f;

    if (lig < n && col < m) {
        for (int i = 0; i < p; i++) {
            s += M1[lig * p + i] * M2[i * m + col];
        }
        Mout[lig * m + col] = s;
    }
}









int main() {
    float *M;
    int n = 3;
    int p = 3;
    int m = 3;

    M = (float *)malloc(n * p * sizeof(float));
    MatrixInit(M, n, p);

    free(M);

    float *M1, *M2, *Mout;
    M1 = (float *)malloc(n * p * sizeof(float));
    M2 = (float *)malloc(p * m * sizeof(float));
    Mout = (float *)malloc(n * m * sizeof(float));

    MatrixInit(M1, n, p);
    MatrixInit(M2, p, m);

    MatrixAdd(M1, M2, Mout, n, p);

    printf("\nMatrix 1\n");
    MatrixPrint(M1, n, p);
    printf("Matrix 2\n");
    MatrixPrint(M2, n, p);
    printf("Result Matrix from Addition on CPU:\n");
    MatrixPrint(Mout, n, p);

    MatrixMult(M1, M2, Mout, n);

    printf("\nMatrix 1\n");
    MatrixPrint(M1, n, p);
    printf("Matrix 2\n");
    MatrixPrint(M2, n, p);
    printf("Result Matrix from Multiplication on CPU:\n");
    MatrixPrint(Mout, n, p);

    float *d_M1, *d_M2, *d_Mout;
    cudaMalloc((void **)&d_M1, sizeof(float) * n * p);
    cudaMalloc((void **)&d_M2, sizeof(float) * p * m);
    cudaMalloc((void **)&d_Mout, sizeof(float) * n * m);

    cudaMemcpy(d_M1, M1, sizeof(float) * n * p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, sizeof(float) * p * m, cudaMemcpyHostToDevice);

    dim3 block_size(n, m);
    dim3 grid_size(1, 1);

    cudaMatrixAdd<<<grid_size, block_size>>>(d_M1, d_M2, d_Mout, n, p);
    cudaDeviceSynchronize();

    cudaMemcpy(Mout, d_Mout, sizeof(float) * n * m, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("\nMatrix 1\n");
    MatrixPrint(M1, n, p);
    printf("Matrix 2\n");
    MatrixPrint(M2, p, m);
    printf("Result Matrix from Addition on GPU:\n");
    MatrixPrint(Mout, n, m);

    cudaMatrixMultGeneral<<<grid_size, block_size>>>(d_M1, d_M2, d_Mout, n, p, m);
    cudaDeviceSynchronize();

    cudaMemcpy(Mout, d_Mout, sizeof(float) * n * m, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("\nMatrix 1\n");
    MatrixPrint(M1, n, p);
    printf("Matrix 2\n");
    MatrixPrint(M2, p, m);
    printf("Result Matrix from Multiplication on GPU:\n");
    MatrixPrint(Mout, n, m);

    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    free(M1);
    free(M2);
    free(Mout);

    return 0;
}

    

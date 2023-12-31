#include <stdio.h>
#include <stdlib.h>

/*
*** Function Name : MatrixInit ***

Initializes a matrix of size NxPxD with specified values based on the 'type':
   type == 0: Fills the matrix with zeros.
   type == 1: Fills the matrix with ones.
   type == 2: Creates a kernel matrix with a central value of 2, others being zero (assumes a 3D matrix).
   type == 3: Fills the matrix with random values between 0 and 1.
Parameters:
   M: Pointer to the matrix (in row-major order).
   n: Number of rows.
   p: Number of columns.
   d: Depth (third dimension).
   type: Initialization mode (0, 1, 2, or 3).
*/


void MatrixInit(float *M, int n, int p, int d, int type){
    
    float random_value;
    
    if (type == 0){
        for (int i = 0; i < n * p * d; i++){
            M[i] =  0;
        }
    }
    if (type == 1){
        for (int i = 0; i < n * p * d; i++){
            M[i] =  1;
        }
    }
    else if (type == 2){
        for (int i = 0; i < n * p * d; i++){
            M[i] =  0;
        }
        for (int k = 0; k < d; k++){
            M[k * (n * p) + 12] = 2;
        }
    }
    else{
        //Valeurs entre 0 et 1
        for (int i = 0; i < n * p * d; i++){
            random_value = (float)rand() / (float)(RAND_MAX/1.0);
            M[i] =  random_value;
        }
    }
}

/*
*** Function Name : MatrixPrint2D ***
Prints a 2D matrix in conventional format.
Parameters:
   M: Pointer to the matrix (in row-major order).
   n: Number of rows.
   p: Number of columns.
*/

void MatrixPrint2D(float *M, int n, int p){
    
    printf("\n");
    for (int lig = 0; lig < p; lig++){
        for(int col = lig * n; col < n * (lig+1); col++){
            printf("%1.1f ", M[col]);
        }
        printf("\n");
    }
    printf("\n");
}

// Layer 2 - Convolution 2D

/*
*** Function Name : cudaConv2D ***

Performs 2D convolution on a matrix M using a specified number of 5x5 kernels.

Parameters:
   M: Input matrix in device memory.
   kernel: Convolution kernels in device memory.
   Mout: Output matrix in device memory to store the result.
   M_ligne: Number of rows in the input matrix.
   M_colonne: Number of columns in the input matrix.
   kernel_size: Size of one side of the square convolution kernel.
   nb_kernel: Number of convolution kernels.
   Mout_ligne: Number of rows in the output matrix.
   Mout_colonne: Number of columns in the output matrix.

Note:
   The dimensions of the output matrix Mout are computed as:
   Mout_ligne = (M_ligne - kernel_size) + 1
   Mout_colonne = (M_colonne - kernel_size) + 1
*/


__global__ void cudaConv2D(float* M, float* kernel, float* Mout, int M_ligne, int M_colonne, int kernel_size, int nb_kernel, int Mout_ligne, int Mout_colonne){
    
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float s;

    if (lig < Mout_ligne && col < Mout_colonne){
        
        int tot_kernel = kernel_size * kernel_size;
        int tot_Mout = Mout_ligne * Mout_colonne;
        
        for (int n_k = 0; n_k < nb_kernel; n_k++){
            s = 0.0;
            
            for (int kernel_lig = 0; kernel_lig < kernel_size; kernel_lig++) {
                for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                    
                    s += M[(lig + kernel_lig) * M_colonne + (col + kernel_col)] * kernel[kernel_lig * kernel_size + kernel_col + n_k * tot_kernel];
                    
                }
            }
            
            Mout[lig * Mout_colonne + col + n_k * tot_Mout] = s;
        }
    }
}

// Layer 3 - Sous-échantillonnage 


/*
*** Function Name : cudaMeanPool ***

Performs mean pooling on the input matrix M using a 2x2 kernel.

Example:
    Given a sub-matrix:   1 2
                      	  3 4
    The mean pool result is: (1 + 2 + 3 + 4) / 4 = 2.5

Parameters:
    M: Pointer to the input matrix.
    Mout: Pointer to the output matrix.
    M_ligne: Number of rows in the input matrix M.
    M_colonne: Number of columns in the input matrix M.
    M_prof: Depth of the input matrix M.
    meanpool_size: The size of the mean pooling window (both rows and columns).
    Mout_ligne: Number of rows in the output matrix Mout.
    Mout_colonne: Number of columns in the output matrix Mout.

Note:
    The relationship between the input and output dimensions is:
    Mout_ligne = M_ligne / meanpool_size
    Mout_colonne = M_colonne / meanpool_size
*/



__global__ void cudaMeanPool(float* M, float* Mout, int M_ligne, int M_colonne, int M_prof, int meanpool_size, int Mout_ligne, int Mout_colonne){
    
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (lig % meanpool_size == 0 && col % meanpool_size == 0){
        
        float s;
        int tot_meanpool = meanpool_size * meanpool_size;
        int tot_M = M_ligne * M_colonne;
        int tot_Mout = Mout_ligne * Mout_colonne;
        
        for (int n_prof = 0; n_prof < M_prof; n_prof++){
            s = 0.0;
            
            for (int meanpool_lig = 0; meanpool_lig < meanpool_size; meanpool_lig++) {
                for (int meanpool_col = 0; meanpool_col < meanpool_size; meanpool_col++) {
                    s += M[(lig + meanpool_lig) * M_colonne + col + meanpool_col + n_prof * tot_M] / tot_meanpool;
            
                }
            }
            if (lig == 0){
                Mout[lig * Mout_colonne + (col / meanpool_size) + n_prof * tot_Mout] = s;
            }
            else if (col == 0){
                Mout[(lig / meanpool_size) * Mout_colonne + col + n_prof * tot_Mout] = s;
            }
            else{
                Mout[(lig / meanpool_size) * Mout_colonne + (col / meanpool_size) + n_prof * tot_Mout] = s;
            }
        }
    }
}

/*
*** Function Name : activation_tanh ***

Applies the hyperbolic tangent (tanh) activation function to each element of matrix M on the GPU.

Note: This is a __device__ function and must be called from a __global__ function on the GPU.

Parameters:
    M: Pointer to the matrix on which to apply the tanh function.
    M_ligne: Number of rows in the matrix M.
    M_colonne: Number of columns in the matrix M.
    M_prof: Depth of the matrix M.
*/


__device__ float* activation_tanh(float* M, int M_ligne, int M_colonne, int M_prof){
    
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (lig < M_ligne && col < M_colonne){
        
        int tot_M = M_ligne * M_colonne;
        
        for (int n_prof = 0; n_prof < M_prof; n_prof++){
            M[lig * M_colonne + col + n_prof * tot_M] = tanh(M[lig * M_colonne + col + n_prof * tot_M]);
        }
            
    }
            
    return M;
}

/*
Kernel function to call the activation_tanh __device__ function.

Parameters:
    M: Pointer to the matrix on the GPU.
    M_ligne: Number of rows in the matrix M.
    M_colonne: Number of columns in the matrix M.
    M_prof: Depth of the matrix M.
*/


__global__ void cudaTanh(float* M, int M_ligne, int M_colonne, int M_prof){
    activation_tanh(M, M_ligne, M_colonne, M_prof);
}

/*
*** Function Name : cudaTanh ***

/*
The cudaTanh kernel launches the device-level function activation_tanh across GPU threads.

This kernel function serves as a wrapper to invoke the activation_tanh function, which 
applies the hyperbolic tangent (tanh) activation function to each element of the input matrix M.

Parameters:
   M: Pointer to the input matrix in device memory.
   M_ligne: The number of rows in the input matrix.
   M_colonne: The number of columns in the input matrix.
   M_prof: The depth of the input matrix, indicating the number of matrices in the case of a 3D matrix.

Note: 
   This kernel should be configured with an appropriate number of blocks and threads to match the size of the input matrix.
   It is assumed that the input matrix M is stored in a flat, row-major format.
*/



int main(){
    
  // CPU \\ 
    
    
    // Création de l'image d'entrée à convoluer
    float *raw_data;    
    raw_data = (float*)malloc(32 * 32 * 1 * sizeof(float));
    
    MatrixInit(raw_data, 32, 32, 1, 1);
    
    // Création de la sortie de la conv2D
    float *C1_data;    
    C1_data = (float*)malloc(28 * 28 * 6 * sizeof(float));
    
    MatrixInit(C1_data, 28, 28, 6, 0);
    
    // Création de la sortie du sous-échantillonnage
    float *S1_data;    
    S1_data = (float*)malloc(14 * 14 * 6 * sizeof(float));
    
    MatrixInit(S1_data, 14, 14, 6, 0);
    
    
    // Création des premiers noyaux de convolution
    float *C1_kernel;    
    C1_kernel = (float*)malloc(5 * 5 * 6 * sizeof(float));
    
    MatrixInit(C1_kernel, 5, 5, 6, 2);

    
// GPU \\ 

    // Définition des matrices cuda
    float *d_raw_data, *d_C1_data, *d_C1_kernel, *d_S1_data;
    
    // Allocation des mémoires des matrices pour cuda
    cudaMalloc((void**)&d_raw_data, sizeof(float) * 32 * 32 * 1);
    cudaMalloc((void**)&d_C1_kernel, sizeof(float) * 5 * 5 * 6);
    cudaMalloc((void**)&d_C1_data, sizeof(float) * 28 * 28 * 6);
    cudaMalloc((void**)&d_S1_data, sizeof(float) * 14 * 14 * 6);
    
    // Copie des valeurs des matrices initialisées sur le CPU dans leur homonyme GPU
    cudaMemcpy(d_raw_data, raw_data, sizeof(float) * 32 * 32 * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * 5 * 5 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, sizeof(float) * 28 * 28 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, sizeof(float) * 14 * 14 * 6, cudaMemcpyHostToDevice);
  
// GPU \\ 

    // Process sur GPU
    dim3 block_size(32, 32);
    dim3 grid_size(1,1);
    
    cudaConv2D<<<grid_size, block_size>>>(d_raw_data, d_C1_kernel, d_C1_data, 32, 32, 5, 6, 28, 28);
    cudaDeviceSynchronize();
    
    cudaTanh<<<grid_size, block_size>>>(d_C1_data, 28, 28, 6);
    cudaDeviceSynchronize();
    
    cudaMeanPool<<<grid_size, block_size>>>(d_C1_data, d_S1_data, 28, 28, 6, 2, 14, 14);
    cudaDeviceSynchronize();
    
    
    // Copie des résultats sur CPU
    cudaMemcpy(C1_data, d_C1_data, sizeof(float) * 28 * 28 * 6, cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, sizeof(float) * 14 * 14 * 6, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    printf("\nMatrice de base raw_data:");
    MatrixPrint2D(raw_data, 32, 32);
    printf("Noyau de convolution C1_kernel:");
    MatrixPrint2D(C1_kernel, 5, 5);
    printf("Matrice résultante de la convolution et de la fonction d'activation:");
    MatrixPrint2D(C1_data, 28, 28);
    printf("Matrice résultante du MeanPooling:");
    MatrixPrint2D(S1_data, 14, 14);
    
    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);
    
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
}

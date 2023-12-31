#include <stdio.h>
#include <stdlib.h>
 
// Layer 1 - Génération des données de test 

/*
*** Function Name : MatrixInit ***

Initializes a matrix of size NxPxD with various options:
   type == 0: All elements set to 0.
   type == 1: A central element set to 1, others set to 0 (assumes 3x3 matrix).
   type == 2: Random initialization between 0 and 1 for each element.
Parameters:
   n: Number of rows in the matrix.
   p: Number of columns in the matrix.
   d: Depth of the matrix (used in 3D matrices).
   M: Pointer to the matrix.
   type: Type of initialization.
*/


void MatrixInit(float *M, int n, int p, int d, int type){
    
    float random_value;
    
    if (type == 0){
        for (int i = 0; i < n * p * d; i++){
            M[i] =  0;
        }
    }
    else if (type == 1){
        for (int i = 0; i < n * p * d; i++){
            M[i] =  0;
        }
        M[4] = 1;
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
*** Function Name : MatrixPrint ***

Prints a NxP matrix in a conventional format.
Parameters:
   n: Number of rows in the matrix.
   p: Number of columns in the matrix.
   M: Pointer to the matrix.
*/

void MatrixPrint2D(float *M, int n, int p){
        
    for (int lig = 0; lig < p; lig++){
        for(int col = lig * n; col < n * (lig+1); col++){
            printf("%1.1f ", M[col]);
        }
        printf("\n");
    }
}

// Layer 2 - Convolution 2D

/*
*** Function Name : cudaConv2D ***

Performs convolution on matrix M using multiple 5x5 convolution kernels.

Parameters:
   M: Pointer to the input matrix.
   kernel: Pointer to the convolution kernels.
   Mout: Pointer to the output matrix.
   M_ligne: Number of rows in M.
   M_colonne: Number of columns in M.
   kernel_size: Size of the convolution kernel.
   nb_kernel: Number of convolution kernels.
   Mout_ligne: Number of rows in the output matrix.
   Mout_colonne: Number of columns in the output matrix.

Note:
   Output matrix dimensions are calculated as: Mout_ligne = (M_ligne - kernel_size) + 1, and similarly for columns.
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
   Sub-matrix [1, 2; 3, 4] would be pooled to 2.5.
     1 2    ==>  2.5 = (1 + 2 + 3 + 4) / 4 = 2.5
     3 4 

Parameters:
   M: Pointer to the input matrix.
   Mout: Pointer to the output matrix.
   M_ligne: Number of rows in M.
   M_colonne: Number of columns in M.
   M_prof: Depth of M.
   meanpool_size: Size of the pooling kernel.
   Mout_ligne: Number of rows in the output matrix.
   Mout_colonne: Number of columns in the output matrix.

Note:
   Output dimensions are calculated as: Mout_ligne = M_ligne / meanpool_size, and similarly for columns.
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

Applies the tanh activation function to the matrix M on the GPU.
Note: This is a __device__ function and should be called from a __global__ function.
Parameters:
   M: Pointer to the matrix.
   M_ligne: Number of rows in M.
   M_colonne: Number of columns in M.
   M_prof: Depth of M.
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
*** Function Name : activation_tanh ***
Wrapper for calling activation_tanh on the GPU.
Parameters:
   M: Pointer to the matrix.
   M_ligne: Number of rows in M.
   M_colonne: Number of columns in M.
   M_prof: Depth of M.
*/

/*
*** Function Name : cudaTanh ***

Sert simplement à appeler la fonction activation_tanh définie juste avant.

Paramètres : 
    M_ligne : nombre de lignes de la matrice M
    M_colonne : nombre de colonnes de la matrice M
    M_prof : profondeur de la matrice M
    M : pointeur de la matrice
*/

__global__ void cudaTanh(float* M, int M_ligne, int M_colonne, int M_prof){
    activation_tanh(M, M_ligne, M_colonne, M_prof);
}

// Layer 4 - Dense | Linear

/*
*** Function Name : cudaMatrixMultGeneral ***

Performs matrix multiplication (dot product) of a NxP matrix with a PxM matrix on the GPU.
Parameters:
   n: Number of rows in matrix M1.
   p: Number of columns in M1 and number of rows in M2.
   m: Number of columns in M2.
   M1: Pointer to the first matrix of size NxP.
   M2: Pointer to the second matrix of size PxM.
   Mout: Pointer to the output matrix of size NxM.
Note:
   Grid and block dimensions in the kernel launch should correspond to the dimensions of the output matrix.
*/


__device__ float* cudaMatrixMultGeneral(float *M1, float *M2, float *Mout, int n, int p, int m){
    
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float s = 0.0f;
    
    if (lig < n && col < m){
        for (int i = 0; i < p; i++){
            s += M1[lig * p + i] * M2[i * m + col];
        }
        Mout[lig * m + col] = s;
    }
    
    return Mout;
}

/*
*** Function Name : cudaMatrixAdd ***

Sert à additionner deux matrices de même taille NxP sur le GPU 

Paramètres : 
    n : nombre de lignes des matrice,
    p : nombre de colonnes des matrices si n différent de p,
    M1 : pointeur de la matrice 1 de taille NxP,
    M2 : pointeur de la matrice 2 de taille NxP,
    Mout : pointeur vers la matrice résultante de l'addition de taille NxP,
    
On peut considérer les dimensions des matrices comme les paramètres gridDim et blockDim pour l'appel de la fonction:
    les lignes correspondent aux blocks,
    les colonnes correspondent aux threads
*/

__device__ float* cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (lig < n && col < p){
        Mout[lig * p + col] = M1[lig * p + col] + M2[lig * p + col];
    }
    
    return Mout;
}


__global__ void cudaDense(float* d_M, float* d_Mout, float* d_W, float* d_b, int n, int p, int m){
    
    d_Mout = cudaMatrixMultGeneral(d_M, d_W, d_Mout, n, p, m);
    d_Mout = cudaMatrixAdd(d_Mout, d_b, d_Mout, n, m);
    
}

// Fonction main
int main(){
    
// CPU \\ 

    

    
    // Création de l'image d'entrée à convoluer
    float *raw_data;    
    raw_data = (float*)malloc(32 * 32 * 1 * sizeof(float));
    
    MatrixInit(raw_data, 32, 32, 1, 2);
    
    // Création de la sortie de la conv2D
    float *C1_data;    
    C1_data = (float*)malloc(28 * 28 * 6 * sizeof(float));
    
    MatrixInit(C1_data, 28, 28, 6, 0);
    
    // Création de la sortie du sous-échantillonnage
    float *S1_data;    
    S1_data = (float*)malloc(14 * 14 * 6 * sizeof(float));
    
    MatrixInit(S1_data, 14, 14, 6, 0);
    
    // Création de la sortie de la conv2D
    float *C2_data;    
    C2_data = (float*)malloc(10 * 10 * 6 * sizeof(float));
    
    MatrixInit(C2_data, 10, 10, 6, 0);
    
    // Création de la sortie du sous-échantillonnage
    float *S2_data;    
    S2_data = (float*)malloc(5 * 5 * 6 * sizeof(float));
    
    MatrixInit(S1_data, 5, 5, 6, 0);
    
    // Création des premiers noyaux de convolution
    float *C1_kernel;    
    C1_kernel = (float*)malloc(5 * 5 * 6 * sizeof(float));
    
    MatrixInit(C1_kernel, 5, 5, 6, 1);
    
    // Création des poids pour la fin du réseau
    float *W1_kernel;    
    W1_kernel = (float*)malloc(400 * 120 * sizeof(float));
    MatrixInit(W1_kernel, 400, 120, 1, 1);
    
    float *B1_kernel;    
    B1_kernel = (float*)malloc(120 * sizeof(float));
    MatrixInit(B1_kernel, 1, 120, 1, 1);
    
    float *D1_data;    
    D1_data = (float*)malloc(120 * sizeof(float));
    MatrixInit(D1_data, 1, 120, 1, 0);

    
    //  GPU \\
    
    // Définition des matrices cuda
    float *d_raw_data, *d_C1_data, *d_C1_kernel, *d_S1_data, *d_C2_data, *d_S2_data, *d_D1_data, *d_W1_kernel, *d_B1_kernel;
    
    // Allocation des mémoires des matrices pour cuda
    cudaMalloc((void**)&d_raw_data, sizeof(float) * 32 * 32 * 1);
    cudaMalloc((void**)&d_C1_kernel, sizeof(float) * 5 * 5 * 6);
    cudaMalloc((void**)&d_C1_data, sizeof(float) * 28 * 28 * 6);
    cudaMalloc((void**)&d_S1_data, sizeof(float) * 14 * 14 * 6);
    cudaMalloc((void**)&d_C2_data, sizeof(float) * 10 * 10 * 6);
    cudaMalloc((void**)&d_S2_data, sizeof(float) * 5 * 5 * 6);
    cudaMalloc((void**)&d_W1_kernel, sizeof(float) * 400 * 120);
    cudaMalloc((void**)&d_B1_kernel, sizeof(float) * 120);
    cudaMalloc((void**)&d_D1_data, sizeof(float) * 400);
    
    // Copie des valeurs des matrices initialisées sur le CPU dans leur homonyme GPU
    cudaMemcpy(d_raw_data, raw_data, sizeof(float) * 32 * 32 * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * 5 * 5 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, sizeof(float) * 28 * 28 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, sizeof(float) * 14 * 14 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2_data, C2_data, sizeof(float) * 10 * 10 * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S2_data, S2_data, sizeof(float) * 5 * 5 * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1_kernel, W1_kernel, sizeof(float) * 120 * 400, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B1_kernel, B1_kernel, sizeof(float) * 120, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D1_data, D1_data, sizeof(float) * 120, cudaMemcpyHostToDevice);
  
    
    
    // CPU \\
  
    
    // Process sur GPU
    dim3 block_size(32, 32);
    dim3 grid_size(1,1);
    
    cudaConv2D<<<grid_size,block_size>>>(d_raw_data, d_C1_kernel, d_C1_data, 32, 32, 5, 6, 28, 28);
    cudaDeviceSynchronize();
    
    cudaTanh<<<grid_size, block_size>>>(d_C1_data, 28, 28, 6);
    cudaDeviceSynchronize();
    
    cudaMeanPool<<<grid_size, block_size>>>(d_C1_data, d_S1_data, 28, 28, 6, 2, 14, 14);
    cudaDeviceSynchronize();
    
    cudaConv2D<<<grid_size,block_size>>>(d_S1_data, d_C1_kernel, d_C2_data, 14, 14, 5, 16, 10, 10);
    cudaDeviceSynchronize();
    
    cudaTanh<<<grid_size, block_size>>>(d_C2_data, 10, 10, 16);
    cudaDeviceSynchronize();
    
    cudaMeanPool<<<grid_size, block_size>>>(d_C2_data, d_S2_data, 10, 10, 16, 2, 5, 5);
    cudaDeviceSynchronize();
    
//    cudaDense<<<grid_size, block_size>>>(d_C2_data, d_D1_data, d_W1_kernel, d_B1_kernel, 1, 400, 120);
//    cudaDeviceSynchronize();
    
    
    // Copie des résultats sur CPU
    cudaMemcpy(C1_data, d_C1_data, sizeof(float) * 28 * 28 * 6, cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, sizeof(float) * 14 * 14 * 6, cudaMemcpyDeviceToHost);
    cudaMemcpy(C2_data, d_C2_data, sizeof(float) * 10 * 10 * 6, cudaMemcpyDeviceToHost);
    cudaMemcpy(S2_data, d_S2_data, sizeof(float) * 5 * 5 * 6, cudaMemcpyDeviceToHost);
    cudaMemcpy(D1_data, d_D1_data, sizeof(float) * 120, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    // Affichage de la matrice résultat
    MatrixPrint2D(C2_data, 5, 5);
    
    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);
    cudaFree(d_C2_data);
    cudaFree(d_S2_data);
    cudaFree(d_D1_data);
    cudaFree(d_W1_kernel);
    cudaFree(d_B1_kernel);
    
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
    free(C2_data);
    free(S2_data);
    free(D1_data);
    free(W1_kernel);
    free(B1_kernel);
}

#include <stdio.h>
#include <stdlib.h>
 
__global__ void kernelA(){
    printf("Hello, from kernelA\n");
}
 
__global__ void kernelB(){
    printf("Hello, from kernelB\n");
}
 
int main()
{
    cudaSetDevice(0);
     
    printf("--------- Example 1 ---------\n");
     
    int blockCount = 1; 
    int threadCount = 4;
 
    // Calling a kenel with 1 block that contains 4 threads
    // Launching a total of 4 threads
    kernelA<<<blockCount,threadCount>>>();
     
    cudaDeviceSynchronize();
 
    printf("\n--------- Example 2 ---------\n");
 
    blockCount = 3; 
    threadCount = 2;
 
    // Calling a kenel with 3 blocks that each contain 2 threads
    // Launching a total of 6 threads
    kernelB<<<blockCount,threadCount>>>();
 
    cudaDeviceSynchronize();
 
   return 0;
}

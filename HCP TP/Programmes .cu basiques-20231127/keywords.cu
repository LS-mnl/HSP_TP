#include <stdio.h>
#include <stdlib.h>
 
// __device__ keyword specifies a function that is run on the device and called from a kernel (1a)
__device__ void GPUFunction(){ 
    printf("\tHello, from the GPU! (1a)\n");
}
 
// This is a kernel that calls a decive function (1b)
__global__ void kernelA(){
    GPUFunction();
}
 
// __host__ __device__ keywords can be specified if the function needs to be 
//                     available to both the host and device (2a)
__host__ __device__ void versatileFunction(){
    printf("\tHello, from the GPU or CPU! (2a)\n");
}
 
// This is a kernel that calls a function on the device (2b)
__global__ void kernelB(){
    versatileFunction();
}
 
int main()
{
    cudaSetDevice(0);
 
    //  Launch a kernel, that will print from a function called by device code (1b -> 1a)
    printf("\nLaunching kernel 1b\n");
    kernelA<<<1,1>>>();
 
    cudaDeviceSynchronize();
 
    // Call a function from the host (2a)
    printf("\nCalling host function 2a\n");
    versatileFunction();
 
    // Call the same function from the device (2b -> 2a)
    printf("\nLaunching kernel 2b\n");
    kernelB<<<1,1>>>();
 
    cudaDeviceSynchronize();
 
   return 0;
}

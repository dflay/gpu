// add two numbers 

#include <stdio.h>
#include <cuda.h> 
#include <cuda_runtime.h> 
#include <curand_kernel.h>  

const int NBLOCK  = 1; 
const int NTHREAD = 1; 

__global__ void add(int a,int b,int *c){
   *c = a + b;
}

int main(void){

   int a = 2;
   int b = 7; 
   int c;
 
   int *c_dev; 

   cudaMalloc( (void**)&c_dev,sizeof(int) ); 
   add<<<NBLOCK,NTHREAD>>>(a,b,c_dev);
   cudaMemcpy(&c,c_dev,sizeof(int),cudaMemcpyDeviceToHost); 

   printf("%d + %d = %d \n",a,b,c); 
   
   cudaFree(c_dev); 

   return 0;
}

// add two vectors  

#include <stdio.h>
#include <cuda.h> 
#include <cuda_runtime.h> 
#include <curand_kernel.h>  

#define N (2048*2048) 

const int THREADS_PER_BLOCK = 512; 

__global__ void add(int npts,int *a,int *b,int *c){
   int i   = threadIdx.x + blockIdx.x*blockDim.x;  // combining blocks and threads; this gives an absolute index  
   if(i<npts){
      c[i] = a[i] + b[i];
   } 
}

int main(void){

   const int SIZE = N*sizeof(int); 

   int *a = (int *)malloc(SIZE); 
   int *b = (int *)malloc(SIZE); 
   int *c = (int *)malloc(SIZE);
 
   int *a_dev,*b_dev,*c_dev; 

   int i=0;
   for(i=0;i<N;i++){
      a[i] = -1*i; 
      b[i] = 2*i; 
   }

   cudaMalloc( (void**)&a_dev,SIZE); 
   cudaMalloc( (void**)&b_dev,SIZE); 
   cudaMalloc( (void**)&c_dev,SIZE); 

   cudaMemcpy(a_dev,a,SIZE,cudaMemcpyHostToDevice); 
   cudaMemcpy(b_dev,b,SIZE,cudaMemcpyHostToDevice); 

   add<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(N,a_dev,b_dev,c_dev);

   cudaMemcpy(c,c_dev,SIZE,cudaMemcpyDeviceToHost); 

   for(i=0;i<N;i++){
      printf("i = %d, %d + %d = %d \n",i,a[i],b[i],c[i]);
   } 
  
   free(a);
   free(b);
   free(c);  
   cudaFree(a_dev); 
   cudaFree(b_dev); 
   cudaFree(c_dev); 

   return 0;
}

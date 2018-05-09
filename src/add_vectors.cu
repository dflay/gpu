// add two vectors  

#include <stdio.h>
#include <cuda.h> 
#include <cuda_runtime.h> 
#include <curand_kernel.h>  

#define N 512 

const int NBLOCK  = 1; 
const int NTHREAD = N; 

__global__ void add(int *a,int *b,int *c){
   int tid = threadIdx.x;  // parse data at this index 
   if(tid<N){
      c[tid] = a[tid] + b[tid];
   } 
}

int main(void){

   int a[N],b[N],c[N];
 
   int *a_dev,*b_dev,*c_dev; 

   int i=0;
   for(i=0;i<N;i++){
      a[i] = -1*i; 
      b[i] = 2*i; 
   }

   cudaMalloc( (void**)&a_dev,N*sizeof(int) ); 
   cudaMalloc( (void**)&b_dev,N*sizeof(int) ); 
   cudaMalloc( (void**)&c_dev,N*sizeof(int) ); 

   cudaMemcpy(a_dev,a,N*sizeof(int),cudaMemcpyHostToDevice); 
   cudaMemcpy(b_dev,b,N*sizeof(int),cudaMemcpyHostToDevice); 

   add<<<NBLOCK,NTHREAD>>>(a_dev,b_dev,c_dev);

   cudaMemcpy(&c,c_dev,N*sizeof(int),cudaMemcpyDeviceToHost); 

   for(i=0;i<N;i++){
      printf("i = %d, %d + %d = %d \n",i,a[i],b[i],c[i]);
   } 
  
   // free(a);
   // free(b);
   // free(c);  
   cudaFree(a_dev); 
   cudaFree(b_dev); 
   cudaFree(c_dev); 

   return 0;
}
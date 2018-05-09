// zero crossing algorithm on the GPU   

#include <stdio.h>
#include <cuda.h> 
#include <cuda_runtime.h> 
#include <curand_kernel.h>  

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

#define N 512 

const int NBLOCK  = N; 
const int NTHREAD = 1; 

__global__ void findTimeOfZC(double *p1,double *p2,double *p3,double *p4,double *p5,){
   // find the time of the zero crossing
   // p1,...,p5 corresponds to a given point in a zero crossing
   // the index of an array corresponds to the ith crossing.  
   // that is, a row of p1[0],...,p5[0] is the zeroth crossing, and so on. 
   int tid = blockIdx.x;  // parse data at this index 
   if(tid<N){
      c[tid] = a[tid] + b[tid];
   } 
}

int findZeroCrossings(int N,double *time,double *ampl,int *zc); 

int main(void){

   int a[N],b[N],c[N];
 
   int *a_dev,*b_dev,*c_dev; 

   for(i=0;i<N;i++){
      a[i] = -1*i; 
      b[i] = i*i*i; 
   }

   cudaMalloc( (void**)&a_dev,N*sizeof(int) ); 
   cudaMalloc( (void**)&b_dev,N*sizeof(int) ); 
   cudaMalloc( (void**)&c_dev,N*sizeof(int) ); 

   cudaMemcpy(a_dev,a,N*sizeof(int),cudaMemcpyHostToDevice); 
   cudaMemcpy(b_dev,b,N*sizeof(int),cudaMemcpyHostToDevice); 

   add<<<NBLOCK,NTHREAD>>>(a_dev,b_dev,c_dev);

   cudaMemcpy(&c,c_dev,sizeof(int),cudaMemcpyDeviceToHost); 

   for(i=0;i<N;i++){
      printf("i = %d, %d + %d = %d \n",i,a[i],b[i],c[i]);
   } 
   
   cudaFree(a_dev); 
   cudaFree(b_dev); 
   cudaFree(c_dev); 

   return 0;
}
//______________________________________________________________________________
int findZeroCrossings(int N,double *time,double *ampl,int *zc){
   // find rough location of the zero crossings 
   int i=0,j=0;
   double prod=0;
   double t_prev=0,t_next=0;
   double v_prev=0,v_next=0; 
   for(i=0;i<N-1;i++){
      prod = ampl[i]*ampl[i+1];
      if(prod<0){
	 // found a crossing!  Mark its location
	 zc[j] = i;
	 j++; 
      }
   }
   return 0;
}

// testing usage of thrust vectors    

#include <iostream>
#include <cmath> 

#include <cuda.h> 
#include <cuda_runtime.h> 
#include <curand_kernel.h>  

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

const int NBLOCK  = N; 
const int NTHREAD = 1; 

// square<T> computes the square of the number f(x) -> x*x 
template <typename T> 
struct square {
   __host__ __device__ T operator()(const T& x) const {
      return x*x; 
   }
}

int main(void){

   const int N = 10;
   thrust::host_vector<double> H(N,1.); // initialize to size N, each entry is 1 
   thrust::device_vector<double> D = H;  

   // arguments to reduce: start, end, initial value, operation 
   double sum = thrust::reduce( D.begin(),D.end(),0.,thrust::plus<double>() ); 

   std::cout << "sum = " << sum << std::endl;

   // now get its norm 
   square<double> SQUARE; 
   thrust::plus<double> PLUS; 
   double init = 0; 
   
   // transform reduce arguments: start, end, unary operator, init value, binary operator 
   double norm = sqrt( thrust::transform_reduce(D.begin(),D.end(),SQUARE,init,PLUS) );

   std::cout << "norm = " << norm << std::endl;  

   return 0;
}


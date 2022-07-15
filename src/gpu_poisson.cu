#include <iostream>
#include <stdio.h>
#include <assert.h>

#include <helper_cuda.h>
#include <cooperatigve_groups.h>



// ----------------------------------------------------------------- 
// For creating shared memory
template<class T>
struct SharedMemory
{
    __device__ inline operator T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};
// ----------------------------------------------------------------- 



// Check memory allocation
void check_cuda(cudaError_t cudaStatus, const char* error) {
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "%s failed: %s\n", error, cudaGetErrorString(cudaStatus));
      exit(EXIT_FAILURE);
  }
  else{
      // printf("Successfully allocated %s\n", error);
      fprintf(stderr, "%s succeeded: %s\n", error, cudaGetErrorString(cudaStatus));

  }
}


// Allocate Memory on GPU
void allocate_matrix_free_GPU(double* out_d2x, double* out_d2y, 
  double* out_bysy1, double* out_v2f_1, double* out_hyinv1,
 double* out_bysy2, double* out_v2f_2, double* out_hyinv2,
 double* out_bxsx3, double* out_v2f_3, double* out_hxinv3_a, double* out_hxinv3_b,
 double* out_bxsx4, double* out_v2f_4, double* out_hxinv4_a, double* out_hxinv4_b,
 double* out_all, double* out_hy, double* out_hx, 
 double** d_out_d2x, double** d_out_d2y, 
  double** d_out_bysy1, double** d_out_v2f_1, double** d_out_hyinv1,
 double** d_out_bysy2, double** d_out_v2f_2, double** d_out_hyinv2,
 double** d_out_bxsx3, double** d_out_v2f_3, double** d_out_hxinv3_a, double** d_out_hxinv3_b,
 double** d_out_bxsx4, double** d_out_v2f_4, double** d_out_hxinv4_a, double** d_out_hxinv4_b,
 double** d_out_all, double** d_out_hy, double** d_out_hx,
 int N)
{
  // Copy data for D2x and D2y
  CopyData<double>(out_d2x, N, sizeof(double), d_out_d2x);
  CopyData<double>(out_d2y, N, sizeof(double), d_out_d2y);

  // Copy data for boundary 1
  CopyData<double>(out_bysy1, N, sizeof(double), d_out_bysy1);
  CopyData<double>(out_v2f_1, N, sizeof(double), d_out_v2f_1);
  CopyData<double>(out_hyinv1, N, sizeof(double), d_out_hyinv1);

  // Copy data for boundary 2
  CopyData<double>(out_bysy2, N, sizeof(double), d_out_bysy2);
  CopyData<double>(out_v2f_2, N, sizeof(double), d_out_v2f_2);
  CopyData<double>(out_hyinv2, N, sizeof(double), d_out_hyinv2);

  // Copy data for boundary 3
  CopyData<double>(out_bxsx3, N, sizeof(double), d_out_bxsx3);
  CopyData<double>(out_v2f_3, N, sizeof(double), d_out_v2f_3);
  CopyData<double>(out_hxinv3_a, N, sizeof(double), d_out_hxinv3_a);
  CopyData<double>(out_hxinv3_b, N, sizeof(double), d_out_hxinv3_b);
  
  // Copy data for boundary 4
  CopyData<double>(out_bxsx4, N, sizeof(double), d_out_bxsx4);
  CopyData<double>(out_v2f_4, N, sizeof(double), d_out_v2f_4);
  CopyData<double>(out_hxinv4_a, N, sizeof(double), d_out_hxinv4_a);
  CopyData<double>(out_hxinv4_b, N, sizeof(double), d_out_hxinv4_b);

  // Copy data for other operators
  CopyData<double>(out_all, N, sizeof(double), d_out_all);
  CopyData<double>(out_hy, N, sizeof(double), d_out_hy);
  CopyData<double>(out_hx, N, sizeof(double), d_out_hx);
}



// Copy result from GPU array to CPU array

void get_result_gpu(double* dev_b, double* b, int m)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    printf("Inside get_result_gpu\n");
    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaMemcpy(b, dev_b, sizeof(double) * m,  
                        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Pinned Host to Device bandwidth (GB/s): %f\n",
         (m * sizeof(double)) * 1e-6 / elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


// Template function for copying data from CPU to GPU

template <class T>
void CopyData(
  T* input,
  unsigned int N,
  unsigned int dsize,
  T** d_in)
{
  // timers
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;

  // Allocate pinned memory on host (for faster HtoD copy)
  T* h_in_pinned = NULL;
  checkCudaErrors(cudaMallocHost((void**) &h_in_pinned, N * dsize));
  assert(h_in_pinned);
  memcpy(h_in_pinned, input, N * dsize);

  // copy data
  checkCudaErrors(cudaMalloc((void**) d_in, N * dsize));
  checkCudaErrors(cudaEventRecord(start, 0));
  checkCudaErrors(cudaMemcpy(*d_in, h_in_pinned,
                             N * dsize, cudaMemcpyHostToDevice));
//    // test copy data back
//     checkCudaErrors(cudaMemcpy(input, *d_in, N*dsize, cudaMemcpyDeviceToHost));
//    //
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("  Pinned Device to Host bandwidth (GB/s): %f\n",
         (N * dsize) * 1e-6 / elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}



// ----------------------------------------------------------------- 
// Other GPU Kernels
template <class T>
__global__ void
vec_add_kernel(T c, T* x, T* y, T* z, int n)
{
    // COMPLETE THIS
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        z[idx] = c * x[idx] + y[idx];
    }
}


void vec_add_gpu(const int n, const double a, double* x, double* y, double* z)
{
    unsigned int threads = 256;
    unsigned int blocks = (n + threads - 1) / threads;
    unsigned int shared = 0;
    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);
   
    // vec_add_kernel<double><<<dimGrid, dimBlock, shared>>>(a, x, y, z, n); 
    vec_add_kernel<double><<<dimGrid, dimBlock>>>(a, x, y, z, n); 
}
// ----------------------------------------------------------------- 






// ----------------------------------------------------------------- 
// Kernels needed for dnrm2
template <class T>
__global__ void
reduce_kernel(T *g_idata, T *g_odata, unsigned int n)
// reduce_kernel(T *g_idata, T *g_odata)
{
    T *sdata = SharedMemory<T>();
    // extern __shared__ T sdata[];
    
    // COMPLETE THIS
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = g_idata[idx];
    __syncthreads();

 
    for (int s = 1; s < blockDim.x; s = s*2){
        if (tid % (2*s) == 0){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }


    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}


template <class T>
__global__ void
vec_mul_kernel(T c, T* x, T* y, T* z, int n)
{
    // COMPLETE THIS
    // unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n){
        z[idx] = c * x[idx] * y[idx];
    }
}


double ddot_gpu(int n, double* x, double *y, double* z1, double* z2)
{
    // Create temporary buffer
    int next_p2 = n;
    // printf("%d\n",n);
    if(!((n != 0) && ((n & (n - 1)) == 0))) {
       next_p2 = pow(2, (int) log2((double) n) + 1);
    } 
    checkCudaErrors(cudaMemset((void*) z1, 0, sizeof(double) * next_p2));


    unsigned int threads = 64;
    unsigned int blocks = next_p2 / threads;
    

    dim3 dimGrid(blocks,1,1);
    dim3 dimBlock(threads,1,1);

    vec_mul_kernel<double><<<dimGrid, dimBlock>>>(1.0,x,y,z1,n);
   

    while (next_p2 >= 64){
        unsigned int blocks = next_p2 / threads;
        // printf("The value of blocks %d\n",blocks);
        dim3 dimGrid(blocks,1,1);
        reduce_kernel<double><<<dimGrid,dimBlock,threads*sizeof(double)>>>(z1,z2,next_p2);
        next_p2 = next_p2 / 64;
        // checkCudaErrors(cudaMemset((void*) z2, 0, sizeof(double) * (next_p2)));
        z1 = z2;
    }
    double* result_z2 = (double*) malloc(sizeof(double) * 64);

    get_result_gpu(z2,result_z2,64);

    double sum_z2 = 0;
    for (int i=0; i < 64; i++){
        sum_z2 += result_z2[i];
    }
   
    double dot = 0.0;

    dot = sum_z2;
    // printf("dot value: %f\n",dot);
    return dot;
}


double dnrm2_gpu(const int n, double* x, double* z1, double* z2)
{
    double nrm = ddot_gpu(n, x, x, z1, z2);
    return sqrt(nrm);
}
// ----------------------------------------------------------------- 




int cg_gpu_matrix_free(double* idata,
  double* d_out_d2x, double* d_out_d2y, 
   double* d_out_bysy1, double* d_out_v2f_1, double* d_out_hyinv1,
  double* d_out_bysy2, double* d_out_v2f_2, double* d_out_hyinv2,
  double* d_out_bxsx3, double* d_out_v2f_3, double* d_out_hxinv3_a, double* d_out_hxinv3_b,
  double* d_out_bxsx4, double* d_out_v2f_4, double* d_out_hxinv4_a, double* d_out_hxinv4_b,
  double* d_out_all, double* d_out_hy, double* d_out_hx,
  int Nx, int Ny, int N, double h, int TILE_DIM_1, int TILE_DIM_2,
  double* z1, double* z2, int max_iter, double tol)
{
  // r0 = b - Ax
  // spmv_gpu(row_ptr, col_ind, vals, m, n, nnz, x, ap); // ap is the final out
  //   TO DO: replace spmv_gpu with matrix-free operator
    Matrix_free_GPU(idata,
      d_out_d2x, d_out_d2y, 
      d_out_bysy1, d_out_v2f_1, d_out_hyinv1,
      d_out_bysy2, d_out_v2f_2, d_out_hyinv2,
      d_out_bxsx3, d_out_v2f_3, d_out_hxinv3_a, d_out_hxinv3_b,
      d_out_bxsx4, d_out_v2f_4, d_out_hxinv4_a, d_out_hxinv4_b,
      d_out_all, d_out_hy, d_out_hx, 
      Nx, Ny, N, h, TILE_DIM_1,TILE_DIM_2); // d_out_hx is the final out
  //


  // vec_add_gpu(m, -1.0, ap, b, rk);
  vec_add_gpu(m, -1.0, d_out_hx, b, rk);

  // if r0 is sufficiently small, return x0 as the result
  double residual = dnrm2_gpu(n, rk, z1, z2);
  // printf("Getting result for dnrm2\n");
  if(residual < tol) {
      fprintf(stdout, "\tInput is the solution\n");
      return 0;
  } else {
      fprintf(stdout, "\n\tInitial residual is %f\n", residual);
  }

  // p0 = r0
  checkCudaErrors(cudaMemcpy(pk, rk, sizeof(double) * m, 
                  cudaMemcpyDeviceToDevice));

  int k = 0;
  double residual_new = 0.0;
  // repeat until convergence of max iterations has been reached
  for(int i = 0; i < max_iter; i++) {
  // A * p
  // spmv_gpu(row_ptr, col_ind, vals, m, n, nnz, pk, ap);
  //   TO DO: replace spmv_gpu with matrix-free operator
  Matrix_free_GPU(pk,
    d_out_d2x, d_out_d2y, 
    d_out_bysy1, d_out_v2f_1, d_out_hyinv1,
    d_out_bysy2, d_out_v2f_2, d_out_hyinv2,
    d_out_bxsx3, d_out_v2f_3, d_out_hxinv3_a, d_out_hxinv3_b,
    d_out_bxsx4, d_out_v2f_4, d_out_hxinv4_a, d_out_hxinv4_b,
    d_out_all, d_out_hy, d_out_hx, 
    Nx, Ny, N, h, TILE_DIM_1,TILE_DIM_2); // d_out_hx is the final out



    //
    // d = p^T * A * p
    // double dotprod = ddot_gpu(m, pk, ap, z1, z2);
    double dotprod = ddot_gpu(m, pk, d_out_hx, z1, z2);
    // printf("Getting result for ddot!\n");
    // alpha = (r^t * r) / d;
    double alpha = (residual * residual) / dotprod;

    // xk = xk + alpha * pk
    vec_add_gpu(m, alpha, pk, x, x);
    // rk = rk - alpha * A*p
    vec_add_gpu(m, (-1.0 * alpha), ap, rk, rk);

    // r^t * r
    residual_new = dnrm2_gpu(m, rk, z1, z2);
    // printf("Getting result for residual new!\n");
    if(residual_new < tol) {
    fprintf(stdout, "\tSolution calculated. Final residual: %f\n", 
        residual_new);
    break;
    } else {
    fprintf(stdout, "\tIt: %d\tresidual: %f\n", k, residual_new); 
    }

    // beta = (r^t * r) / residual
    double beta = (residual_new * residual_new) / (residual * residual);

    // p = r + beta * p 
    vec_add_gpu(m, beta, pk, rk, pk);

    residual = residual_new;
    k++;
    }
    return 0;
}



// function for Matrix_free_GPU, assembling all GPU kernels into one function


void Matrix_free_GPU(double* idata,
     double* d_out_d2x, double* d_out_d2y, 
      double* d_out_bysy1, double* d_out_v2f_1, double* d_out_hyinv1,
     double* d_out_bysy2, double* d_out_v2f_2, double* d_out_hyinv2,
     double* d_out_bxsx3, double* d_out_v2f_3, double* d_out_hxinv3_a, double* d_out_hxinv3_b,
     double* d_out_bxsx4, double* d_out_v2f_4, double* d_out_hxinv4_a, double* d_out_hxinv4_b,
     double* d_out_all, double* d_out_hy, double* d_out_hx,
     int Nx, int Ny, int N, double h, int TILE_DIM_1, int TILE_DIM_2)
{

  // Set variables for SBP-SAT

  double alpha1 =  -1.0;
  double alpha2 = -1.0;

  double alpha3 = - (double)13*double(Nx - 1);
  double alpha4 = - (double)13*double(Ny - 1);

  double beta = 1.0;

  printf("alpha1:%f\talpha2:%f\talpha3:%f\talpha4:%f\n",alpha1,alpha2,alpha3,alpha4);


  printf("Nx: %d, Ny: %d, N: %d, h: %f, TILE_DIM_1: %d, TILE_DIM_2:%d\n",Nx,Ny,N,h,TILE_DIM_1,TILE_DIM_2);
  dim3 dimBlock_2d_x(TILE_DIM_1, TILE_DIM_2, 1);
  dim3 dimGrid_2d_x(Nx/TILE_DIM_1 + 1, Ny/TILE_DIM_2 + 1,1);
  dim3 dimBlock_2d_y(TILE_DIM_2, TILE_DIM_1, 1);
  dim3 dimGrid_2d_y(Nx/TILE_DIM_2 + 1, Ny/TILE_DIM_1 + 1,1);

  dim3 dimBlock_1d(256,1,1);
  dim3 dimGrid_1d(N/256+1,1,1);

  D2x_shared_kernel<double><<<dimGrid_2d_x, dimBlock_2d_x>>>(idata,d_out_d2x,Nx,Ny,h);
  D2y_shared_kernel<double><<<dimGrid_2d_y, dimBlock_2d_y>>>(idata,d_out_d2y,Nx,Ny,h);

  // for face 1: North
  BySy_shared_kernel<double><<<dimGrid_2d_y, dimBlock_2d_y>>>(idata,d_out_bysy1,Nx,Ny,h);
  VOLtoFACE_shared_kernel<double><<<dimGrid_1d, dimBlock_1d>>>(d_out_bysy1,d_out_v2f_1,Nx,Ny,h,1);
  Hyinv_shared_kernel<double><<<dimGrid_2d_y, dimBlock_2d_y>>>(d_out_v2f_1,d_out_bysy1,Nx,Ny,h);

  // for face 2: South
  VOLtoFACE_shared_kernel<double><<<dimGrid_1d, dimBlock_1d>>>(d_out_bysy2,d_out_v2f_1,Nx,Ny,h,2);
  Hyinv_shared_kernel<double><<<dimGrid_2d_y, dimBlock_2d_y>>>(d_out_v2f_2,d_out_bysy2,Nx,Ny,h);

  // for face 3: West
  VOLtoFACE_shared_kernel<double><<<dimGrid_1d, dimBlock_1d>>>(idata,d_out_v2f_3,Nx,Ny,h,3);
  BxSx_tran_shared_kernel<double><<<dimGrid_2d_x, dimBlock_2d_x>>>(d_out_v2f_3,d_out_bxsx3,Nx,Ny,h);
  Hxinv_shared_kernel<double><<<dimGrid_2d_x, dimBlock_2d_x>>>(d_out_bxsx3,d_out_hxinv3_a,Nx,Ny,h);
  Hxinv_shared_kernel<double><<<dimGrid_2d_x, dimBlock_2d_x>>>(d_out_v2f_3,d_out_hxinv3_b,Nx,Ny,h);


  // for face 4: East
  VOLtoFACE_shared_kernel<double><<<dimGrid_1d, dimBlock_1d>>>(idata,d_out_v2f_4,Nx,Ny,h,4);
  BxSx_tran_shared_kernel<double><<<dimGrid_2d_x, dimBlock_2d_x>>>(d_out_v2f_4,d_out_bxsx4,Nx,Ny,h);
  Hxinv_shared_kernel<double><<<dimGrid_2d_x, dimBlock_2d_x>>>(d_out_bxsx4,d_out_hxinv4_a,Nx,Ny,h);
  Hxinv_shared_kernel<double><<<dimGrid_2d_x, dimBlock_2d_x>>>(d_out_v2f_4,d_out_hxinv4_b,Nx,Ny,h);




  vec_add_kernel<double><<<dimGrid_1d, dimBlock_1d>>>(1.0, d_out_d2x, d_out_d2y, d_out_all, N);
  vec_add_kernel<double><<<dimGrid_1d, dimBlock_1d>>>(alpha1, d_out_bysy1 , d_out_all, d_out_all, N);
  vec_add_kernel<double><<<dimGrid_1d, dimBlock_1d>>>(alpha2, d_out_bysy2 , d_out_all, d_out_all, N);
  vec_add_kernel<double><<<dimGrid_1d, dimBlock_1d>>>(beta, d_out_hxinv3_a , d_out_all, d_out_all, N);
  vec_add_kernel<double><<<dimGrid_1d, dimBlock_1d>>>(alpha3, d_out_hxinv3_b , d_out_all, d_out_all, N);
  vec_add_kernel<double><<<dimGrid_1d, dimBlock_1d>>>(beta, d_out_hxinv4_a , d_out_all, d_out_all, N);
  vec_add_kernel<double><<<dimGrid_1d, dimBlock_1d>>>(alpha4, d_out_hxinv4_b , d_out_all, d_out_all, N);
  vec_add_kernel<double><<<dimGrid_1d, dimBlock_1d>>>(-2.0, d_out_all, d_out_all, d_out_all,N);

  Hy_shared_kernel<double><<<dimGrid_2d_y, dimBlock_2d_y>>>(d_out_all,d_out_hy,Nx,Ny,h);
  Hx_shared_kernel<double><<<dimGrid_2d_x, dimBlock_2d_x>>>(d_out_hy,d_out_hx,Nx,Ny,h);
}



void free_gpu(double* d_out_hx)
{
  checkCudaErrors(cudaFree(d_out_hx))
}
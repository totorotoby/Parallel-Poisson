#include <iostream>
#include <stdio.h>
#include <assert.h>

#include <helper_cuda.h>
#include <cooperative_groups.h>


// Set fixed tile dimensions;
const int TILE_DIM_1 = 4;
const int TILE_DIM_2 = 4;



template <class T>
__global__ void 
D2x_kernel(T* idata, T* odata, int Nx, int Ny, double h)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N;
    N = Nx * Ny;

    if (idx < Ny) 
    {
        // odata[idx] = (idata[idx] - 2*idata[idx + Ny] + idata[idx + 2*Ny]) / h*h;
        odata[idx] = (idata[idx] - 2*idata[idx + Ny] + idata[idx + 2*Ny]) / (h*h);
    }

    if ((Ny <= idx) && (idx < N - Ny)) 
    {
        odata[idx] = (idata[idx - Ny] - 2*idata[idx] + idata[idx + Ny]) / (h*h);
    }

    if ((idx >= N - Ny) && (idx < N))
    {
        odata[idx] = (idata[idx - 2*Ny] - 2*idata[idx - Ny] + idata[idx]) / (h*h);
    }

    __syncthreads();
}


template <class T>
__global__ void
D2x_shared_kernel(T* idata, T* odata, int Nx, int Ny, double h)
{
    // __shared__ double s_f[my+8][lPencils];
    const int HALO_WIDTH = 2;
    const int TILE_DIM_1 = 4;
    const int TILE_DIM_2 = 4;
    __shared__ double tile[TILE_DIM_1][TILE_DIM_2 + 2 * HALO_WIDTH];


    unsigned tidx = threadIdx.x;
    unsigned tidy = threadIdx.y;

    unsigned i = blockIdx.x * blockDim.x + tidx;
    unsigned j = blockIdx.y * blockDim.y + tidy;

    unsigned global_index = i + j * Ny;

    // for tile indexing
    unsigned int k = tidx;
    unsigned int l = tidy;

    // Writing global memory to shared memory

    // for tile itself

    if ((k < TILE_DIM_1) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx))
    {
        tile[k][l+HALO_WIDTH] = idata[global_index];
    }

    // for left halo
    if ( (k < TILE_DIM_1) && (l < HALO_WIDTH) && (i < Ny) && (j >= HALO_WIDTH) && (j < HALO_WIDTH + Nx))
    {
        tile[k][l] = idata[global_index - HALO_WIDTH * Ny];
    }

    // for right halo
    if ((k < TILE_DIM_1) && (l >= TILE_DIM_2 - HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx - HALO_WIDTH))
    {
        tile[k][l+2*HALO_WIDTH] = idata[global_index + HALO_WIDTH*Ny];
    }

    __syncthreads();


    // starting calculation and writting data to output
    // Finite Difference Operations starts here

    // left boundary for second order
    if ((k < TILE_DIM_1) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH - 2) && (i < Ny) && (j == 0))
    {
        odata[global_index] = (tile[k][l + HALO_WIDTH] - 2*tile[k][l+HALO_WIDTH+1] + tile[k][l+HALO_WIDTH+2]) / (h*h);
        // odata[global_index] = 1.0;
    }

    // center
    if ((k < TILE_DIM_1) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH -1) && (i < Ny) && (j >= 1) && (j < Nx - 1))
    {
        odata[global_index] = (tile[k][l + HALO_WIDTH - 1] - 2*tile[k][l+HALO_WIDTH] + tile[k][l+HALO_WIDTH + 1]) / (h*h);
        // odata[global_index] = 2.0;
    }

    // right boundary for second order
    if ((k < TILE_DIM_1) && (2 <= l + HALO_WIDTH) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH) && (i < Ny) && (j == Nx-1))
    {
        odata[global_index] = (tile[k][l+HALO_WIDTH - 2] - 2*tile[k][l + HALO_WIDTH - 1] + tile[k][l+HALO_WIDTH])/ (h*h);
        // odata[global_index] = 3.0;
    }

    // __syncthreads();
}


template <class T>
__global__ void
Dx_shared_kernel(T* idata, T* odata, int Nx, int Ny, double h)
{
    const int HALO_WIDTH = 2;
    const int TILE_DIM_1 = 4;
    const int TILE_DIM_2 = 4;
    __shared__ double tile[TILE_DIM_1][TILE_DIM_2 + 2 * HALO_WIDTH];


    unsigned tidx = threadIdx.x;
    unsigned tidy = threadIdx.y;

    unsigned i = blockIdx.x * blockDim.x + tidx;
    unsigned j = blockIdx.y * blockDim.y + tidy;

    unsigned global_index = i + j * Ny;

    // for tile indexing
    unsigned int k = tidx;
    unsigned int l = tidy;

    // Writing global memory to shared memory

    // for tile itself

    if ((k < TILE_DIM_1) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx))
    {
        tile[k][l+HALO_WIDTH] = idata[global_index];
    }

    // for left halo
    if ( (k < TILE_DIM_1) && (l < HALO_WIDTH) && (i < Ny) && (j >= HALO_WIDTH) && (j < HALO_WIDTH + Nx))
    {
        tile[k][l] = idata[global_index - HALO_WIDTH * Ny];
    }

    // for right halo
    if ((k < TILE_DIM_1) && (l >= TILE_DIM_2 - HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx - HALO_WIDTH))
    {
        tile[k][l+2*HALO_WIDTH] = idata[global_index + HALO_WIDTH*Ny];
    }

    __syncthreads();

      // starting calculation and writting data to output
    // Finite Difference Operations starts here

    // left boundary for second order
    if ((k < TILE_DIM_1) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH - 2) && (i < Ny) && (j == 0))
    {
        odata[global_index] = (tile[k][l + HALO_WIDTH + 1] - tile[k][l+HALO_WIDTH]) / (h);
    }

    // center
    if ((k < TILE_DIM_1) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH -1) && (i < Ny) && (j >= 1) && (j < Nx - 1))
    {
        odata[global_index] = (tile[k][l + HALO_WIDTH - 1] - tile[k][l+HALO_WIDTH + 1]) / (2*h);
    }

    // right boundary for second order
    if ((k < TILE_DIM_1) && (2 <= l + HALO_WIDTH) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH) && (i < Ny) && (j == Nx-1))
    {
        odata[global_index] = (tile[k][l+HALO_WIDTH] - tile[k][l + HALO_WIDTH - 1])/ (h*h);
    }

}



template <class T>
__global__ void
Bx_shared_kernel(T* idata, T* odata, int Nx, int Ny, double h)
{
    const int HALO_WIDTH = 2;
    const int TILE_DIM_1 = 4;
    const int TILE_DIM_2 = 4;
    __shared__ double tile[TILE_DIM_1][TILE_DIM_2 + 2 * HALO_WIDTH];


    unsigned tidx = threadIdx.x;
    unsigned tidy = threadIdx.y;

    unsigned i = blockIdx.x * blockDim.x + tidx;
    unsigned j = blockIdx.y * blockDim.y + tidy;

    unsigned global_index = i + j * Ny;

    // for tile indexing
    unsigned int k = tidx;
    unsigned int l = tidy;

    // Writing global memory to shared memory

    // for tile itself

    if ((k < TILE_DIM_1) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx))
    {
        tile[k][l+HALO_WIDTH] = idata[global_index];
    }

    // for left halo
    if ( (k < TILE_DIM_1) && (l < HALO_WIDTH) && (i < Ny) && (j >= HALO_WIDTH) && (j < HALO_WIDTH + Nx))
    {
        tile[k][l] = idata[global_index - HALO_WIDTH * Ny];
    }

    // for right halo
    if ((k < TILE_DIM_1) && (l >= TILE_DIM_2 - HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx - HALO_WIDTH))
    {
        tile[k][l+2*HALO_WIDTH] = idata[global_index + HALO_WIDTH*Ny];
    }

    __syncthreads();

      // starting calculation and writting data to output
    // Finite Difference Operations starts here

    // left boundary for second order
    if ((k < TILE_DIM_1) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH - 2) && (i < Ny) && (j == 0))
    {
        odata[global_index] = -1.0;
    }

    // center
    if ((k < TILE_DIM_1) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH -1) && (i < Ny) && (j >= 1) && (j < Nx - 1))
    {
        odata[global_index] = 0;
    }

    // right boundary for second order
    if ((k < TILE_DIM_1) && (2 <= l + HALO_WIDTH) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH) && (i < Ny) && (j == Nx-1))
    {
        odata[global_index] = 1.0;
    }

}


template <class T>
__global__ void
BxSx_shared_kernel(T* idata, T* odata, int Nx, int Ny, double h)
{
    const int HALO_WIDTH = 2;
    const int TILE_DIM_1 = 4;
    const int TILE_DIM_2 = 4;
    __shared__ double tile[TILE_DIM_1][TILE_DIM_2 + 2 * HALO_WIDTH];


    unsigned tidx = threadIdx.x;
    unsigned tidy = threadIdx.y;

    unsigned i = blockIdx.x * blockDim.x + tidx;
    unsigned j = blockIdx.y * blockDim.y + tidy;

    unsigned global_index = i + j * Ny;

    // for tile indexing
    unsigned int k = tidx;
    unsigned int l = tidy;

    // Writing global memory to shared memory

    // for tile itself

    if ((k < TILE_DIM_1) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx))
    {
        tile[k][l+HALO_WIDTH] = idata[global_index];
    }

    // for left halo
    if ( (k < TILE_DIM_1) && (l < HALO_WIDTH) && (i < Ny) && (j >= HALO_WIDTH) && (j < HALO_WIDTH + Nx))
    {
        tile[k][l] = idata[global_index - HALO_WIDTH * Ny];
    }

    // for right halo
    if ((k < TILE_DIM_1) && (l >= TILE_DIM_2 - HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx - HALO_WIDTH))
    {
        tile[k][l+2*HALO_WIDTH] = idata[global_index + HALO_WIDTH*Ny];
    }

    __syncthreads();

      // starting calculation and writting data to output
    // Finite Difference Operations starts here

    // left boundary for second order
    if ((k < TILE_DIM_1) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH - 2) && (i < Ny) && (j == 0))
    {
        odata[global_index] = (1.5 * tile[k][l+HALO_WIDTH] - 2.0 * tile[k][l+HALO_WIDTH+1] + 0.5*tile[k][l+HALO_WIDTH+2]) / h;
    }

    // center
    if ((k < TILE_DIM_1) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH -1) && (i < Ny) && (j >= 1) && (j < Nx - 1))
    {
        odata[global_index] = 0;
    }

    // right boundary for second order
    if ((k < TILE_DIM_1) && (2 <= l + HALO_WIDTH) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH) && (i < Ny) && (j == Nx-1))
    {
        odata[global_index] =  (0.5 * tile[k][l+HALO_WIDTH-2] - 2.0 * tile[k][l+HALO_WIDTH-1] + 0.5*tile[k][l+HALO_WIDTH]) / h;
    }

}


template <class T>
__global__ void
BxSx_tran_shared_kernel(T* idata, T* odata, int Nx, int Ny, double h)
{
    const int HALO_WIDTH = 2;
    const int TILE_DIM_1 = 4;
    const int TILE_DIM_2 = 4;
    __shared__ double tile[TILE_DIM_1][TILE_DIM_2 + 2 * HALO_WIDTH];


    unsigned tidx = threadIdx.x;
    unsigned tidy = threadIdx.y;

    unsigned i = blockIdx.x * blockDim.x + tidx;
    unsigned j = blockIdx.y * blockDim.y + tidy;

    unsigned global_index = i + j * Ny;

    // for tile indexing
    unsigned int k = tidx;
    unsigned int l = tidy;

    // Writing global memory to shared memory

    // for tile itself

    if ((k < TILE_DIM_1) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx))
    {
        tile[k][l+HALO_WIDTH] = idata[global_index];
    }

    // for left halo
    if ( (k < TILE_DIM_1) && (l < HALO_WIDTH) && (i < Ny) && (j >= HALO_WIDTH) && (j < HALO_WIDTH + Nx))
    {
        tile[k][l] = idata[global_index - HALO_WIDTH * Ny];
    }

    // for right halo
    if ((k < TILE_DIM_1) && (l >= TILE_DIM_2 - HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx - HALO_WIDTH))
    {
        tile[k][l+2*HALO_WIDTH] = idata[global_index + HALO_WIDTH*Ny];
    }

    __syncthreads();

      // starting calculation and writting data to output
    // Finite Difference Operations starts here

    // left boundary for second order
    if ((k < TILE_DIM_1) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH - 2) && (i < Ny) && (j == 0))
    {
        odata[global_index] = (1.5 * tile[k][l+HALO_WIDTH]  ) / h;
        odata[global_index + Ny] = (- 2.0 * tile[k][l+HALO_WIDTH]) / h;
        odata[global_index + 2*Ny] = (0.5*tile[k][l+HALO_WIDTH]) / h;
    }
    __syncthreads();

    // center
    if ((k < TILE_DIM_1) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH -1) && (i < Ny) && (j >= 1) && (j < Nx - 1))
    {
        odata[global_index] = 0;
    }

    // right boundary for second order
    if ((k < TILE_DIM_1) && (2 <= l + HALO_WIDTH) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH) && (i < Ny) && (j == Nx-1))
    {
        odata[global_index] = (1.5 * tile[k][l+HALO_WIDTH]  ) / h;
        odata[global_index - Ny] = (- 2.0 * tile[k][l+HALO_WIDTH]) / h;
        odata[global_index - 2*Ny] = (0.5*tile[k][l+HALO_WIDTH]) / h;
    }

    __syncthreads();

}


template <class T>
__global__ void
Hxinv_shared_kernel(T* idata, T* odata, int Nx, int Ny, double h)
{
    const int HALO_WIDTH = 2;
    const int TILE_DIM_1 = 4;
    const int TILE_DIM_2 = 4;
    __shared__ double tile[TILE_DIM_1][TILE_DIM_2 + 2 * HALO_WIDTH];


    unsigned tidx = threadIdx.x;
    unsigned tidy = threadIdx.y;

    unsigned i = blockIdx.x * blockDim.x + tidx;
    unsigned j = blockIdx.y * blockDim.y + tidy;

    unsigned global_index = i + j * Ny;

    // for tile indexing
    unsigned int k = tidx;
    unsigned int l = tidy;

    // Writing global memory to shared memory

    // for tile itself

    if ((k < TILE_DIM_1) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx))
    {
        tile[k][l+HALO_WIDTH] = idata[global_index];
    }

    // for left halo
    if ( (k < TILE_DIM_1) && (l < HALO_WIDTH) && (i < Ny) && (j >= HALO_WIDTH) && (j < HALO_WIDTH + Nx))
    {
        tile[k][l] = idata[global_index - HALO_WIDTH * Ny];
    }

    // for right halo
    if ((k < TILE_DIM_1) && (l >= TILE_DIM_2 - HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx - HALO_WIDTH))
    {
        tile[k][l+2*HALO_WIDTH] = idata[global_index + HALO_WIDTH*Ny];
    }

    __syncthreads();

      // starting calculation and writting data to output
    // Finite Difference Operations starts here

    // left boundary for second order
    if ((k < TILE_DIM_1) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH - 2) && (i < Ny) && (j == 0))
    {
        odata[global_index] = (2*tile[k][l + HALO_WIDTH]) / (h);
    }

    // center
    if ((k < TILE_DIM_1) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH -1) && (i < Ny) && (j >= 1) && (j < Nx - 1))
    {
        odata[global_index] = (tile[k][l + HALO_WIDTH]) / (h);
    }

    // right boundary for second order
    if ((k < TILE_DIM_1) && (2 <= l + HALO_WIDTH) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH) && (i < Ny) && (j == Nx-1))
    {
        odata[global_index] = (2*tile[k][l+HALO_WIDTH])/ (h);
    }

}


template <class T>
__global__ void
Hx_shared_kernel(T* idata, T* odata, int Nx, int Ny, double h)
{
    const int HALO_WIDTH = 2;
    const int TILE_DIM_1 = 4;
    const int TILE_DIM_2 = 4;
    __shared__ double tile[TILE_DIM_1][TILE_DIM_2 + 2 * HALO_WIDTH];


    unsigned tidx = threadIdx.x;
    unsigned tidy = threadIdx.y;

    unsigned i = blockIdx.x * blockDim.x + tidx;
    unsigned j = blockIdx.y * blockDim.y + tidy;

    unsigned global_index = i + j * Ny;

    // for tile indexing
    unsigned int k = tidx;
    unsigned int l = tidy;

    // Writing global memory to shared memory

    // for tile itself

    if ((k < TILE_DIM_1) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx))
    {
        tile[k][l+HALO_WIDTH] = idata[global_index];
    }

    // for left halo
    if ( (k < TILE_DIM_1) && (l < HALO_WIDTH) && (i < Ny) && (j >= HALO_WIDTH) && (j < HALO_WIDTH + Nx))
    {
        tile[k][l] = idata[global_index - HALO_WIDTH * Ny];
    }

    // for right halo
    if ((k < TILE_DIM_1) && (l >= TILE_DIM_2 - HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx - HALO_WIDTH))
    {
        tile[k][l+2*HALO_WIDTH] = idata[global_index + HALO_WIDTH*Ny];
    }

    __syncthreads();

      // starting calculation and writting data to output
    // Finite Difference Operations starts here

    // left boundary for second order
    if ((k < TILE_DIM_1) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH - 2) && (i < Ny) && (j == 0))
    {
        odata[global_index] = (h*tile[k][l + HALO_WIDTH]) / (2);
    }

    // center
    if ((k < TILE_DIM_1) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH -1) && (i < Ny) && (j >= 1) && (j < Nx - 1))
    {
        odata[global_index] = h * (tile[k][l + HALO_WIDTH]) ;
    }

    // right boundary for second order
    if ((k < TILE_DIM_1) && (2 <= l + HALO_WIDTH) && (l + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH) && (i < Ny) && (j == Nx-1))
    {
        odata[global_index] = (h*tile[k][l+HALO_WIDTH])/ (2);
    }

}

template <class T>
__global__ void
D2y_shared_kernel(T* idata, T* odata, int Nx, int Ny, double h)
{
    // __shared__ double s_f[my+8][lPencils];
    const int HALO_WIDTH = 2;
    const int TILE_DIM_1 = 4;
    const int TILE_DIM_2 = 4;
    // __shared__ double tile[TILE_DIM_1][TILE_DIM_2 + 2 * HALO_WIDTH];
    __shared__ double tile[TILE_DIM_1 + 2*HALO_WIDTH][TILE_DIM_2];


    unsigned tidx = threadIdx.x;
    unsigned tidy = threadIdx.y;

    unsigned i = blockIdx.x * blockDim.x + tidx;
    unsigned j = blockIdx.y * blockDim.y + tidy;

    unsigned global_index = i + j * Ny;

    // for tile indexing
    unsigned int k = tidx;
    unsigned int l = tidy;

    // Writing global memory to shared memory

    // for tile itself

    if ((k < HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx))
    {
        tile[k+HALO_WIDTH][l] = idata[global_index];
    }

    // for upper halo
    if ( (k < HALO_WIDTH) && (l < TILE_DIM_2) && (i >=  HALO_WIDTH) && (j < Nx))
    {
        tile[k][l] = idata[global_index - HALO_WIDTH];
    }

    // for lower halo
    if ((k < TILE_DIM_1) && (k >= TILE_DIM_1 - HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny - HALO_WIDTH) && (j < Nx))
    {
        tile[k+2*HALO_WIDTH][l] = idata[global_index + HALO_WIDTH*Ny];
    }

    __syncthreads();


    // starting calculation and writting data to output
    // Finite Difference Operations starts here

    // left boundary for second order
    if ((l < TILE_DIM_2) && (k + HALO_WIDTH < TILE_DIM_1 + 2*HALO_WIDTH - 2) && (j < Ny) && (i == 0))
    {
        odata[global_index] = (tile[k+HALO_WIDTH][l] - 2*tile[k+HALO_WIDTH+1][l] + tile[k+HALO_WIDTH+2][l]) / (h*h);
    }

    // center
    if ((l < TILE_DIM_2) && (k + HALO_WIDTH < TILE_DIM_1 + 2*HALO_WIDTH -1) && (j < Ny) && (i >= 1) && (i < Nx - 1))
    {
        odata[global_index] = (tile[k+HALO_WIDTH-1][l] - 2*tile[k+HALO_WIDTH][l] + tile[k+HALO_WIDTH+1][l]) / (h*h);
    }

    // right boundary for second order
    if ((l < TILE_DIM_2) && (2 <= k + HALO_WIDTH) && (k + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH) && (j < Ny) && (i == Nx-1))
    {
        odata[global_index] = (tile[k+HALO_WIDTH-2][l] - 2*tile[k+HALO_WIDTH-1][l] + tile[k+HALO_WIDTH][l])/ (h*h);
    }

    // __syncthreads();
}

template <class T>
__global__ void
Dy_shared_kernel(T* idata, T* odata, int Nx, int Ny, double h)
{
    // __shared__ double s_f[my+8][lPencils];
    const int HALO_WIDTH = 2;
    const int TILE_DIM_1 = 4;
    const int TILE_DIM_2 = 4;
    // __shared__ double tile[TILE_DIM_1][TILE_DIM_2 + 2 * HALO_WIDTH];
    __shared__ double tile[TILE_DIM_1 + 2*HALO_WIDTH][TILE_DIM_2];


    unsigned tidx = threadIdx.x;
    unsigned tidy = threadIdx.y;

    unsigned i = blockIdx.x * blockDim.x + tidx;
    unsigned j = blockIdx.y * blockDim.y + tidy;

    unsigned global_index = i + j * Ny;

    // for tile indexing
    unsigned int k = tidx;
    unsigned int l = tidy;

    // Writing global memory to shared memory

    // for tile itself

    if ((k < HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx))
    {
        tile[k+HALO_WIDTH][l] = idata[global_index];
    }

    // for upper halo
    if ( (k < HALO_WIDTH) && (l < TILE_DIM_2) && (i >=  HALO_WIDTH) && (j < Nx))
    {
        tile[k][l] = idata[global_index - HALO_WIDTH];
    }

    // for lower halo
    if ((k < TILE_DIM_1) && (k >= TILE_DIM_1 - HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny - HALO_WIDTH) && (j < Nx))
    {
        tile[k+2*HALO_WIDTH][l] = idata[global_index + HALO_WIDTH*Ny];
    }

    __syncthreads();


    // starting calculation and writting data to output
    // Finite Difference Operations starts here

    // left boundary for second order
    if ((l < TILE_DIM_2) && (k + HALO_WIDTH < TILE_DIM_1 + 2*HALO_WIDTH - 2) && (j < Ny) && (i == 0))
    {
        odata[global_index] = (tile[k+HALO_WIDTH+1][l] - tile[k+HALO_WIDTH][l]) / (h);
    }

    // center
    if ((l < TILE_DIM_2) && (k + HALO_WIDTH < TILE_DIM_1 + 2*HALO_WIDTH -1) && (j < Ny) && (i >= 1) && (i < Nx - 1))
    {
        odata[global_index] = (tile[k+HALO_WIDTH+1][l] - tile[k+HALO_WIDTH-1][l]) / (2*h);
    }

    // right boundary for second order
    if ((l < TILE_DIM_2) && (2 <= k + HALO_WIDTH) && (k + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH) && (j < Ny) && (i == Nx-1))
    {
        odata[global_index] = (tile[k+HALO_WIDTH][l] - tile[k+HALO_WIDTH-1][l])/ (h*h);
    }

    // __syncthreads();
}



template <class T>
__global__ void
By_shared_kernel(T* idata, T* odata, int Nx, int Ny, double h)
{
    // __shared__ double s_f[my+8][lPencils];
    const int HALO_WIDTH = 2;
    const int TILE_DIM_1 = 4;
    const int TILE_DIM_2 = 4;
    // __shared__ double tile[TILE_DIM_1][TILE_DIM_2 + 2 * HALO_WIDTH];
    __shared__ double tile[TILE_DIM_1 + 2*HALO_WIDTH][TILE_DIM_2];


    unsigned tidx = threadIdx.x;
    unsigned tidy = threadIdx.y;

    unsigned i = blockIdx.x * blockDim.x + tidx;
    unsigned j = blockIdx.y * blockDim.y + tidy;

    unsigned global_index = i + j * Ny;

    // for tile indexing
    unsigned int k = tidx;
    unsigned int l = tidy;

    // Writing global memory to shared memory

    // for tile itself

    if ((k < HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx))
    {
        tile[k+HALO_WIDTH][l] = idata[global_index];
    }

    // for upper halo
    if ( (k < HALO_WIDTH) && (l < TILE_DIM_2) && (i >=  HALO_WIDTH) && (j < Nx))
    {
        tile[k][l] = idata[global_index - HALO_WIDTH];
    }

    // for lower halo
    if ((k < TILE_DIM_1) && (k >= TILE_DIM_1 - HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny - HALO_WIDTH) && (j < Nx))
    {
        tile[k+2*HALO_WIDTH][l] = idata[global_index + HALO_WIDTH*Ny];
    }

    __syncthreads();


    // starting calculation and writting data to output
    // Finite Difference Operations starts here

    // left boundary for second order
    if ((l < TILE_DIM_2) && (k + HALO_WIDTH < TILE_DIM_1 + 2*HALO_WIDTH - 2) && (j < Ny) && (i == 0))
    {
        odata[global_index] = - 1.0;
    }

    // center
    if ((l < TILE_DIM_2) && (k + HALO_WIDTH < TILE_DIM_1 + 2*HALO_WIDTH -1) && (j < Ny) && (i >= 1) && (i < Nx - 1))
    {
        odata[global_index] = 0;
    }

    // right boundary for second order
    if ((l < TILE_DIM_2) && (2 <= k + HALO_WIDTH) && (k + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH) && (j < Ny) && (i == Nx-1))
    {
        odata[global_index] = 1.0;
    }

    // __syncthreads();
}



template <class T>
__global__ void
BySy_shared_kernel(T* idata, T* odata, int Nx, int Ny, double h)
{
    // __shared__ double s_f[my+8][lPencils];
    const int HALO_WIDTH = 2;
    const int TILE_DIM_1 = 4;
    const int TILE_DIM_2 = 4;
    // __shared__ double tile[TILE_DIM_1][TILE_DIM_2 + 2 * HALO_WIDTH];
    __shared__ double tile[TILE_DIM_1 + 2*HALO_WIDTH][TILE_DIM_2];


    unsigned tidx = threadIdx.x;
    unsigned tidy = threadIdx.y;

    unsigned i = blockIdx.x * blockDim.x + tidx;
    unsigned j = blockIdx.y * blockDim.y + tidy;

    unsigned global_index = i + j * Ny;

    // for tile indexing
    unsigned int k = tidx;
    unsigned int l = tidy;

    // Writing global memory to shared memory

    // for tile itself

    if ((k < HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx))
    {
        tile[k+HALO_WIDTH][l] = idata[global_index];
    }

    // for upper halo
    if ( (k < HALO_WIDTH) && (l < TILE_DIM_2) && (i >=  HALO_WIDTH) && (j < Nx))
    {
        tile[k][l] = idata[global_index - HALO_WIDTH];
    }

    // for lower halo
    if ((k < TILE_DIM_1) && (k >= TILE_DIM_1 - HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny - HALO_WIDTH) && (j < Nx))
    {
        tile[k+2*HALO_WIDTH][l] = idata[global_index + HALO_WIDTH*Ny];
    }

    __syncthreads();


    // starting calculation and writting data to output
    // Finite Difference Operations starts here

    // left boundary for second order
    if ((l < TILE_DIM_2) && (k + HALO_WIDTH < TILE_DIM_1 + 2*HALO_WIDTH - 2) && (j < Ny) && (i == 0))
    {
        odata[global_index] = (1.5 * tile[k+HALO_WIDTH][l] - 2*tile[k+HALO_WIDTH+1][l] - 0.5*tile[k+HALO_WIDTH+2][l]) / h;
    }

    // center
    if ((l < TILE_DIM_2) && (k + HALO_WIDTH < TILE_DIM_1 + 2*HALO_WIDTH -1) && (j < Ny) && (i >= 1) && (i < Nx - 1))
    {
        odata[global_index] = 0;
    }

    // right boundary for second order
    if ((l < TILE_DIM_2) && (2 <= k + HALO_WIDTH) && (k + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH) && (j < Ny) && (i == Nx-1))
    {
        odata[global_index] = (1.5 * tile[k+HALO_WIDTH - 2][l] - 2*tile[k+HALO_WIDTH-1][l] - 0.5*tile[k+HALO_WIDTH][l]) / h;
    }

    // __syncthreads();
}



template <class T>
__global__ void
BySy_tran_shared_kernel(T* idata, T* odata, int Nx, int Ny, double h)
{
    // __shared__ double s_f[my+8][lPencils];
    const int HALO_WIDTH = 2;
    const int TILE_DIM_1 = 4;
    const int TILE_DIM_2 = 4;
    // __shared__ double tile[TILE_DIM_1][TILE_DIM_2 + 2 * HALO_WIDTH];
    __shared__ double tile[TILE_DIM_1 + 2*HALO_WIDTH][TILE_DIM_2];


    unsigned tidx = threadIdx.x;
    unsigned tidy = threadIdx.y;

    unsigned i = blockIdx.x * blockDim.x + tidx;
    unsigned j = blockIdx.y * blockDim.y + tidy;

    unsigned global_index = i + j * Ny;

    // for tile indexing
    unsigned int k = tidx;
    unsigned int l = tidy;

    // Writing global memory to shared memory

    // for tile itself

    if ((k < HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx))
    {
        tile[k+HALO_WIDTH][l] = idata[global_index];
    }

    // for upper halo
    if ( (k < HALO_WIDTH) && (l < TILE_DIM_2) && (i >=  HALO_WIDTH) && (j < Nx))
    {
        tile[k][l] = idata[global_index - HALO_WIDTH];
    }

    // for lower halo
    if ((k < TILE_DIM_1) && (k >= TILE_DIM_1 - HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny - HALO_WIDTH) && (j < Nx))
    {
        tile[k+2*HALO_WIDTH][l] = idata[global_index + HALO_WIDTH*Ny];
    }

    __syncthreads();


    // starting calculation and writting data to output
    // Finite Difference Operations starts here

    // left boundary for second order
    if ((l < TILE_DIM_2) && (k + HALO_WIDTH < TILE_DIM_1 + 2*HALO_WIDTH - 2) && (j < Ny) && (i == 0))
    {
       odata[global_index] = (1.5*tile[k+HALO_WIDTH][l]) / h;
       odata[global_index + 1] = (-2.0 * tile[k+HALO_WIDTH][l]) / h;
       odata[global_index + 2] = (0.5 * tile[k+HALO_WIDTH][l]) / h;
    }

    // center
    if ((l < TILE_DIM_2) && (k + HALO_WIDTH < TILE_DIM_1 + 2*HALO_WIDTH -1) && (j < Ny) && (i >= 1) && (i < Nx - 1))
    {
        odata[global_index] = 0;
    }

    // right boundary for second order
    if ((l < TILE_DIM_2) && (2 <= k + HALO_WIDTH) && (k + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH) && (j < Ny) && (i == Nx-1))
    {
        odata[global_index] = (1.5*tile[k+HALO_WIDTH][l]) / h;
        odata[global_index - 1] = (-2.0 * tile[k+HALO_WIDTH][l]) / h;
        odata[global_index - 2] = (0.5 * tile[k+HALO_WIDTH][l]) / h;
    }

    // __syncthreads();
}



template <class T>
__global__ void
Hyinv_shared_kernel(T* idata, T* odata, int Nx, int Ny, double h)
{
    // __shared__ double s_f[my+8][lPencils];
    const int HALO_WIDTH = 2;
    const int TILE_DIM_1 = 4;
    const int TILE_DIM_2 = 4;
    // __shared__ double tile[TILE_DIM_1][TILE_DIM_2 + 2 * HALO_WIDTH];
    __shared__ double tile[TILE_DIM_1 + 2*HALO_WIDTH][TILE_DIM_2];


    unsigned tidx = threadIdx.x;
    unsigned tidy = threadIdx.y;

    unsigned i = blockIdx.x * blockDim.x + tidx;
    unsigned j = blockIdx.y * blockDim.y + tidy;

    unsigned global_index = i + j * Ny;

    // for tile indexing
    unsigned int k = tidx;
    unsigned int l = tidy;

    // Writing global memory to shared memory

    // for tile itself

    if ((k < HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx))
    {
        tile[k+HALO_WIDTH][l] = idata[global_index];
    }

    // for upper halo
    if ( (k < HALO_WIDTH) && (l < TILE_DIM_2) && (i >=  HALO_WIDTH) && (j < Nx))
    {
        tile[k][l] = idata[global_index - HALO_WIDTH];
    }

    // for lower halo
    if ((k < TILE_DIM_1) && (k >= TILE_DIM_1 - HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny - HALO_WIDTH) && (j < Nx))
    {
        tile[k+2*HALO_WIDTH][l] = idata[global_index + HALO_WIDTH*Ny];
    }

    __syncthreads();


    // starting calculation and writting data to output
    // Finite Difference Operations starts here

    // left boundary for second order
    if ((l < TILE_DIM_2) && (k + HALO_WIDTH < TILE_DIM_1 + 2*HALO_WIDTH - 2) && (j < Ny) && (i == 0))
    {
        odata[global_index] = (2*tile[k+HALO_WIDTH][l]) / (h);
    }

    // center
    if ((l < TILE_DIM_2) && (k + HALO_WIDTH < TILE_DIM_1 + 2*HALO_WIDTH -1) && (j < Ny) && (i >= 1) && (i < Nx - 1))
    {
        odata[global_index] = (tile[k+HALO_WIDTH][l]) / (h);
    }

    // right boundary for second order
    if ((l < TILE_DIM_2) && (2 <= k + HALO_WIDTH) && (k + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH) && (j < Ny) && (i == Nx-1))
    {
        odata[global_index] = (2*tile[k+HALO_WIDTH][l])/ (h);
    }

    // __syncthreads();
}



template <class T>
__global__ void
Hy_shared_kernel(T* idata, T* odata, int Nx, int Ny, double h)
{
    // __shared__ double s_f[my+8][lPencils];
    const int HALO_WIDTH = 2;
    const int TILE_DIM_1 = 4;
    const int TILE_DIM_2 = 4;
    // __shared__ double tile[TILE_DIM_1][TILE_DIM_2 + 2 * HALO_WIDTH];
    __shared__ double tile[TILE_DIM_1 + 2*HALO_WIDTH][TILE_DIM_2];


    unsigned tidx = threadIdx.x;
    unsigned tidy = threadIdx.y;

    unsigned i = blockIdx.x * blockDim.x + tidx;
    unsigned j = blockIdx.y * blockDim.y + tidy;

    unsigned global_index = i + j * Ny;

    // for tile indexing
    unsigned int k = tidx;
    unsigned int l = tidy;

    // Writing global memory to shared memory

    // for tile itself

    if ((k < HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny) && (j < Nx))
    {
        tile[k+HALO_WIDTH][l] = idata[global_index];
    }

    // for upper halo
    if ( (k < HALO_WIDTH) && (l < TILE_DIM_2) && (i >=  HALO_WIDTH) && (j < Nx))
    {
        tile[k][l] = idata[global_index - HALO_WIDTH];
    }

    // for lower halo
    if ((k < TILE_DIM_1) && (k >= TILE_DIM_1 - HALO_WIDTH) && (l < TILE_DIM_2) && (i < Ny - HALO_WIDTH) && (j < Nx))
    {
        tile[k+2*HALO_WIDTH][l] = idata[global_index + HALO_WIDTH*Ny];
    }

    __syncthreads();


    // starting calculation and writting data to output
    // Finite Difference Operations starts here

    // left boundary for second order
    if ((l < TILE_DIM_2) && (k + HALO_WIDTH < TILE_DIM_1 + 2*HALO_WIDTH - 2) && (j < Ny) && (i == 0))
    {
        odata[global_index] = (h*tile[k+HALO_WIDTH][l]) / (2);
    }

    // center
    if ((l < TILE_DIM_2) && (k + HALO_WIDTH < TILE_DIM_1 + 2*HALO_WIDTH -1) && (j < Ny) && (i >= 1) && (i < Nx - 1))
    {
        odata[global_index] = (h*tile[k+HALO_WIDTH][l]);
    }

    // right boundary for second order
    if ((l < TILE_DIM_2) && (2 <= k + HALO_WIDTH) && (k + HALO_WIDTH < TILE_DIM_2 + 2*HALO_WIDTH) && (j < Ny) && (i == Nx-1))
    {
        odata[global_index] = (h*tile[k+HALO_WIDTH][l])/ (2);
    }

    // __syncthreads();
}

// template <class T>
// __global__ void
// test_shared(T* idata, T*odata)
// {
//     unsigned idx_x = blockDim.x*blockIdx.x + threadIdx.x;
//     unsigned idx_y = blockIdx.y*blockIdx.y + threadIdx.y;

//     __shared__ double sf[4][8];

//     unsigned int global_index = idx_x * 16 + idx_y;

//     unsigned int si = threadIdx.x + 2;
//     unsigned int sj = threadIdx.y;

//     // if (si < 8 && sj < 4)
//     if (global_index < 256)
//     {
//         sf[sj][si] = idata[global_index];
//     }
//     __syncthreads();
    
//     // if (si < 8 && sj < 4)
//     if (global_index < 256)
//     {
//         odata[global_index] = sf[sj][si] + 2.5;
//     }
//     __syncthreads();
// }


template <class T>
__global__ void
FACEtoVOL_shared_kernel(T* idata, T* odata, int Nx, int Ny, double h, int face)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N;
    N = Nx * Ny;

    if (idx < Nx)
    {
        odata[idx] = 0.0;
    }

    if ((face == 1) && (idx < Nx))
    {
        odata[idx*Ny] = idata[idx];
    }

    if ((face == 2) && (idx < Nx))
    {
        odata[(idx+1)*Ny-1] = idata[idx];
    }

    if ((face == 3) && (idx < Nx))
    {
        odata[idx] = idata[idx];
    }

    if ((face == 4) && (idx < Nx))
    {
        odata[idx + (Ny-1)*Nx] = idata[idx];
    }
    __syncthreads();
}

template <class T>
__global__ void
VOLtoFACE_shared_kernel(T* idata, T* odata, int Nx, int Ny, double h, int face)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N;
    N = Nx * Ny;

    if (idx < Nx)
    {
        odata[idx] = 0.0;
    }

    if ((face == 1) && (idx < Nx))
    {
        // odata[idx*Ny] = idata[idx];
        odata[idx] = idata[idx*Ny];
    }

    if ((face == 2) && (idx < Nx))
    {
        // odata[(idx+1)*Ny-1] = idata[idx];
        odata[idx] = idata[(idx+1)*Ny-1];

    }

    if ((face == 3) && (idx < Nx))
    {
        // odata[idx] = idata[idx];
        odata[idx] = idata[idx];
    }

    if ((face == 4) && (idx < Nx))
    {
        // odata[idx + (Ny-1)*Nx] = idata[idx];
        odata[idx] = idata[idx + (Ny-1)*Nx];
    }
    __syncthreads();
}


void Matrix_free()
{
      unsigned int Nx = 1025;
      unsigned int Ny = 1025;
      unsigned int TILE_DIM_1 = 4;
      unsigned int TILE_DIM_2 = 16;

      unsigned int GRID_DIM_X_X = Nx / TILE_DIM_1 + 1;
      unsigned int GRID_DIM_X_Y = Ny / TILE_DIM_2 + 1;
      unsigned int GRID_DIM_Y_X = Nx / TILE_DIM_2 + 1;
      unsigned int GRID_DIM_Y_Y = Ny / TILE_DIM_1 + 1;

      dim3 dimBlock_X(TILE_DIM_1, TILE_DIM_2);
      dim3 dimBlock_y(TILE_DIM_2, TILE_DIM_1);
      dim3 dimGrid_X(GRID_DIM_X_X, GRID_DIM_X_Y);
      dim3 dimGrid_y(GRID_DIM_Y_X, GRID_DIM_Y_Y);
}
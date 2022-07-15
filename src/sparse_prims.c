#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "cpu.h"
#include <string.h>

/*
  res = A + c*B, wheere A is COO B is COO and c is a scalar
*/
sparseCOO* addCOO(sparseCOO* A, sparseCOO* B, double c)
{
  // allocate memory for result
  sparseCOO* res = malloc(sizeof(sparseCOO));

  //count the number of nonzeros for the result
  int nnz = 0;
  int iter;
  sparseCOO* outside;
  sparseCOO* inside;
  if (A->nnz <= B->nnz)
    {
      nnz = B->nnz;
      inside = B;
      outside = A;
    }
  else
    {
      nnz = A->nnz;
      inside = A;
      outside = B;
    }
    
  for (int i = 0 ; i < outside->nnz; i++){
    int A_row = outside->row_index[i];
    int A_col = outside->col_index[i];
    int count = 0;    
    for (int j =  0 ; j < inside->nnz; j++)
      {	
	if (A_row != inside->row_index[j] || A_col != inside->col_index[j])
	  count++;
      }
    if (count == inside->nnz)
      nnz++;
  }

  // allocate memory to new matrix
  int* col_index = malloc(sizeof(int) * nnz);
  int* row_index = malloc(sizeof(int) * nnz);
  double* values = malloc(sizeof(double) * nnz);


  int index = 0;
  int a_ind = 0;
  int b_ind = 0;
  
  while(a_ind < A->nnz && b_ind < B->nnz)
    {
      
      if(A->row_index[a_ind] < B->row_index[b_ind] ||
	 (A->row_index[a_ind] == B->row_index[b_ind] &&
	  A->col_index[a_ind] < B->col_index[b_ind]))
	{

	  
	  
	  row_index[index] = A->row_index[a_ind];
	  col_index[index] = A->col_index[a_ind];
	  values[index] = A->values[a_ind];

	  a_ind++;
	  index++;
	}
	    
	    
	  else if(A->row_index[a_ind] > B->row_index[b_ind] ||
		  (A->row_index[a_ind] == B->row_index[b_ind] &&
		   A->col_index[a_ind] > B->col_index[b_ind]))
	    {
	      row_index[index] = B->row_index[b_ind];
	      col_index[index] = B->col_index[b_ind];
	      values[index] = c * B->values[b_ind];
	      
	      b_ind++;
	      index++;
	    }
	  
      //if(B->row_index[a_ind] == A->row_index[b_ind] &&
	    //B->col_index[a_ind] == A->row_index[b_ind])
	  else
	    {
	      
	      row_index[index] = A->row_index[a_ind];
	      col_index[index] = A->col_index[a_ind];
	      values[index] = c * B->values[b_ind] + A->values[a_ind] ;
	      
	      b_ind++;
	      a_ind++;
	      index++;
	    }	 
    }

  //get the rest in
  while(a_ind < A->nnz)
    {
      	  row_index[index] = A->row_index[a_ind];
	  col_index[index] = A->col_index[a_ind];
	  values[index] = A->values[a_ind];
	  a_ind++;
	  index++;
    }

  while(b_ind < B->nnz)
    {
      	      row_index[index] = B->row_index[b_ind];
	      col_index[index] = B->col_index[b_ind];
	      values[index] = B->values[b_ind];
	      
	      b_ind++;
	      index++;
    }
      

  res->row_index = row_index;
  res->col_index = col_index;
  res->values = values;
  res->nnz = nnz;
  
  return res;
  
}


sparseCOO* transpose(sparseCOO* A)
{

  int i;
  int nnz = A->nnz;
  sparseCOO* res = malloc(sizeof(sparseCOO));
  coopoint *to_sort = (coopoint*) malloc(sizeof(coopoint)*nnz);

  int* row_ind = malloc(sizeof(int) * nnz);
  int* col_ind = malloc(sizeof(int) * nnz);
  double* values = malloc(sizeof(double) * nnz);

  for (i = 0 ; i < nnz ; i++)
    {
      to_sort[i].row_ind = A -> col_index[i];
      to_sort[i].col_ind = A -> row_index[i];
      to_sort[i].val = A -> values[i];
    }

  qsort(to_sort, nnz, sizeof(coopoint), cmpfunc);

  for (i = 0 ; i < nnz ; i++)
    {
      row_ind[i] = to_sort[i].row_ind;
      col_ind[i] = to_sort[i].col_ind;
      values[i] = to_sort[i].val;
      
    }

  res -> row_index = row_ind;
  res -> col_index = col_ind;
  res -> values = values;

  free(to_sort);
  
  return res;

}

int cmpfunc(const void *a, const void *b)
{
  int row_a = ((coopoint*)a)->row_ind;
  int row_b = ((coopoint*)b)->row_ind;

  return (row_a - row_b);
}



void scaleCOO(double c, sparseCOO* mat)
{
  for (int i = 0 ; i < mat->nnz ; i++)
    {
      mat -> values[i] = c * (mat -> values[i]);
    }
}


double *addvecs(double* v1, double* v2, double scale, int length)
{

  double* res = malloc(sizeof(double) * length);

  int i;
  #pragma omp parallel for private(i)
  for (i = 0; i < length ; i++)
    {
      res[i] = v1[i] + scale * v2[i];
    }

  return res;
  
}


void spmv_coo(sparseCOO* mat, int m, int n, double* vec,
	      double *res)
{

  int i;
  int col;
  int row;

  omp_lock_t* writelock; 
  init_locks(&writelock, m);

  int* row_ind = mat->row_index;
  int* col_ind = mat->col_index;
  double* vals = mat->values;
  int nnz = mat->nnz;
  
  #pragma omp parallel for private(i)
  for (i = 0 ; i < m ; i++)
    res[i] = 0;

  #pragma omp parallel for private(i,col,row)
  for (i = 0 ; i < nnz ; i++)
    {
      col = col_ind[i];
      row = row_ind[i];
      omp_set_lock(&(writelock[row]));
      res[row] += vals[i] * vec[col];
      omp_unset_lock(&(writelock[row]));
    }
}


/* SpMV function for CSR stored sparse matrix
 */
void spmv(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
          double* csr_vals, int m, int n, int nnz, 
          double* vector_x, double *res)
{
  
  int i;
  int j;
  
  #pragma omp parallel for private(i,j)
  for (i = 0 ; i < m ; i++)
    {
      res[i] = 0.0;
      for (j = csr_row_ptr[i] ; j < csr_row_ptr[i+1] ; j++)
	{
	  res[i] += csr_vals[j] * vector_x[csr_col_ind[j]];
	}
    }
}



void vec_add(const int n, const double a, const double* x, double* y, double* z)
{
  int i;
  //#pragma omp parallel for private(i) schedule(static)
  for (i = 0 ; i < n ; i++)
    {
      z[i] = a * x[i] + y[i];
    }
	
}


double dnrm2(const int n, double* x, const int incx)
{
  int i;
  double nrm = sqrt(ddot(n, x, incx, x, incx));
  return nrm;
}

double ddot(const int n, double* x, const int incx, double* y, const int incy)
{
    double sum = 0.0;
    int i; 

    //just for testing cg_parallel
    //omp_set_num_threads(12);
    #pragma omp parallel for private(i) schedule(static) reduction(+:sum)
    for(i = 0 ; i<n ; i++)
      {
      sum += x[i * incx] * y[i * incy];
      }

    return sum;
}



void init_locks(omp_lock_t** locks, int m)
{
    omp_lock_t* _locks = (omp_lock_t*) malloc(sizeof(omp_lock_t) * m);
    for(int i = 0; i < m; i++) {
        omp_init_lock(&(_locks[i]));
    }
    *locks = _locks;
}

/* SpMV function for CSR stored sparse matrix
 

void spmv(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
          double* csr_vals, int m, int n, int nnz, 
          double* vector_x, double *res)
{

  int i;
  int j;
 
  //#pragma omp parallel for private(i,j)
  for (i = 0 ; i < m ; i++)
    {
      res[i] = 0.0;
      for (j = csr_row_ptr[i] ; j < csr_row_ptr[i+1] ; j++)
	{
	  res[i] += csr_vals[j] * vector_x[csr_col_ind[j]];
	}
    }
}

*/
void printCOO(sparseCOO* toprint)
{
  
  int n = toprint->nnz;

  printf("Number of non zeros: %d\n", n);
  printf("Sparse COO matrix\n");
  printf("\trow\tcol\tval\n");
  for(int i=0; i < n ; i++)
    printf("\t%d\t%d\t%f\n", toprint->row_index[i], toprint->col_index[i], toprint->values[i]);
  
}

void printCOOns(sparseCOO* toprint, int N)
{

  printf("%dx%d Sparse Matrix:\n", N, N);
  
  int index = 0;
  for (int i = 0 ; i < N ; i++)
    {
      printf("\n");
      for (int j = 0; j < N ; j++)
	{
	  if (toprint->row_index[index] == i &&
	      toprint-> col_index[index] == j)
	    {
	      printf("%7.2f", toprint->values[index]);
	      index ++;
	    }
	  else
	    {
	      printf("%7.2f", 0.0);
	    }
	}
    }
  printf("\n\n");
}

void printCSRns(sparseCSR* toprint, int N)
{

  int begin;
  int end;
  int diff;
  printf("\n");
  for (int i = 0 ; i < N ; i++)
    {
      begin = toprint->row_ptr[i];
      end = toprint->row_ptr[i+1];
      int count = 0;
      for (int j = begin ; j < end ; j++)
	{
	  diff = toprint->col_index[j+1] - toprint->col_index[j] - 1;
	  /*if (i == 2)
	    {
	      printf("%d, %d, %d\n", toprint->col_index[j],
		     toprint->col_index[j+1],
		     toprint->col_index[j+1] - toprint->col_index[j]-1);
	      
		     }*/

	  if (j == begin)
	    {
	    for (int k = 0 ; k <toprint->col_index[j] ; k++)
	      {
		printf("%7.1f", 0.0);
		count++;
	      }
	    }
	  printf("%7.1f", toprint-> values[j]);
	  count++;
	  for (int k = 0 ; k < diff ; k++)
	    {
	      printf("%7.1f", 0.0);
	      count++;
	    }
	}
      while (count < N)
	{
	  printf("%7.1f", 0.0);
	  count++;
	}
      printf("\n");
    }
}   



void freeCOO(sparseCOO* tofree)
{

  free(tofree->col_index);
  free(tofree->row_index);
  free(tofree->values);
  free(tofree);
  
}

void convert_coo_to_csr(sparseCOO* coo, int m, int n,
			      int** csr_row_ptr, int** csr_col_index,
			      double** csr_values)
{
  
  int nnz = coo->nnz;
  int i;

  int *csr_row_ptr_ =  malloc(sizeof(int)*(m+1));

  for (i = 0 ; i <= m ; i++)
    {
      csr_row_ptr_[i] = 0;
    }
  
  for (i = 0 ; i < nnz ; i++)
    {
      csr_row_ptr_[coo->row_index[i] + 1]++;
    }
  
  for (i = 0 ; i < m ; i++)
    {
      csr_row_ptr_[i+1] += csr_row_ptr_[i];
    }
 
 
  *csr_col_index = coo->col_index;
  *csr_values = coo->values;
  *csr_row_ptr = csr_row_ptr_;
  
}


//test for addCOO
/*
int main()
{

  int n = 4;

  int nnza = 9;
  int nnzb = 8;
  
  sparseCOO* A = malloc(sizeof(sparseCOO));
  sparseCOO* B = malloc(sizeof(sparseCOO));

  int* col_indexA = malloc(sizeof(int) * nnza);
  int* row_indexA = malloc(sizeof(int) * nnza);
  double* valuesA = malloc(sizeof(double) * nnza);
  
  int* col_indexB = malloc(sizeof(int) * nnzb);
  int* row_indexB = malloc(sizeof(int) * nnzb);
  double* valuesB = malloc(sizeof(double) * nnzb);

  double* b = malloc(sizeof(double) * n);
  b[0] = 1;
  b[1] = 2;
  b[2] = 3;
  b[3] = 4;
  
  

  col_indexA[0] = 0;
  row_indexA[0] = 0;
  valuesA[0] = 1;
  col_indexA[1] = 3;
  row_indexA[1] = 0;
  valuesA[1] = 2;
  col_indexA[2] = 0;
  row_indexA[2] = 1;
  valuesA[2] = 4;
  col_indexA[3] = 2;
  row_indexA[3] = 1;
  valuesA[3] = 3;
  col_indexA[4] = 1;
  row_indexA[4] = 2;
  valuesA[4] = 2;
  col_indexA[5] = 0;
  row_indexA[5] = 3;
  valuesA[5] = 1;
  col_indexA[6] = 1;
  row_indexA[6] = 3;
  valuesA[6] = 5;
  col_indexA[7] = 2;
  row_indexA[7] = 3;
  valuesA[7] = 5;
  col_indexA[8] = 3;
  row_indexA[8] = 3;
  valuesA[8] = 5;

  
  col_indexB[0] = 0;
  row_indexB[0] = 0;
  valuesB[0] = 2;
  col_indexB[1] = 0;
  row_indexB[1] = 1;
  valuesB[1] = 3;
  col_indexB[2] = 1;
  row_indexB[2] = 1;
  valuesB[2] = 2;
  col_indexB[3] = 3;
  row_indexB[3] = 1;
  valuesB[3] = 4;
  col_indexB[4] = 2;
  row_indexB[4] = 2;
  valuesB[4] = 2;
  col_indexB[5] = 3;
  row_indexB[5] = 2;
  valuesB[5] = 3;
  col_indexB[6] = 1;
  row_indexB[6] = 3;
  valuesB[6] = 2;
  col_indexB[7] = 3;
  row_indexB[7] = 3;
  valuesB[7] = 2;
  
  
  A->col_index = col_indexA;
  A->row_index = row_indexA;
  A->values = valuesA;
  A->nnz = nnza;
  
  B->col_index = col_indexB;
  B->row_index = row_indexB;
  B->values = valuesB;
  B->nnz = nnzb;

  //sparseCOO* res = addCOO(B, A, n);
  
  //testing spmv
  sparseCSR* Acsr = convert_coo_to_csr(A, n, n);
  //double *spmv_vec = malloc(sizeof(double)*n);
  //printCSRns(Acsr, n);
  sparseCOO* AT = transpose(A);
  //printCOOns(A, n);
  //printCOOns(B, n);
  //printCOOns(AT, n);
  
  
  //spmv(Acsr, n, n, b, spmv_vec);

  //for (int i = 0 ; i < n ; i++)
  //printf("%f\n", spmv_vec[i]);
  
  
}
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cpu.h"








/*
Constructs Ax matrix (matrix for interior x operators without boundaries)
  parameters:
  h        length between nodes in both spatial directions
  n        number of nodes in each physical dimension
  Ax       memory to fill
 */
void constructAx(double h, int n, sparseCOO*Ax)
{

  int nnz = (3*(n-2) + 4) * n;
  int N = n*n;

  
  int* row_ind = malloc(sizeof(int) * nnz);
  int* col_ind = malloc(sizeof(int) * nnz);
  double* values = malloc(sizeof(double) * nnz);

  int index = 0;
  for (int i = 0 ; i < N ; i++)
    {
      if (i < n)
	{
	  row_ind[index] = i;
	  col_ind[index] = i;
	  values[index] = 1.0/h;

	  index++;
	  
	  row_ind[index] = i;
	  col_ind[index] = i+n;
	  values[index] = -1.0/h;

	  index++;
	  
	}
      else if (i > (n*n) - (n+1))
	{

	  row_ind[index] = i;
	  col_ind[index] = i-n;
	  values[index] = -1.0/h;

	  index++;
	  
	  row_ind[index] = i;
	  col_ind[index] = i;
	  values[index] = 1.0/h;

	  index++;
	  
	}

      else
	{
	  
	  row_ind[index] = i;
	  col_ind[index] = i-n;
	  values[index] = -1.0/h;
	  
	  index++;
	  
	  row_ind[index] = i;
	  col_ind[index] = i;
	  values[index] = 2.0/h;

	  index++;
	  
	  row_ind[index] = i;
	  col_ind[index] = i+n;
	  values[index] = -1.0/h;

	  index++;
	  
	}	
    }

  Ax->row_index = row_ind;
  Ax->col_index = col_ind;
  Ax->values = values;
  Ax->nnz = nnz;

}
/*
Constructs Ay matrix (matrix for interior y operators without boundaries)
  parameters:
  h        length between nodes in both spatial directions
  n        number of nodes in each physical dimension
  Ay     memory to fill
 */
void constructAy(double h, int n, sparseCOO*Ay)
{

  int nnz = (3*(n-2) + 4) * n;
  int N = n*n;

  
  int* row_ind = malloc(sizeof(int) * nnz);
  int* col_ind = malloc(sizeof(int) * nnz);
  double* values = malloc(sizeof(double) * nnz);

  
  //TODO: Fix this loop.
  //everything scaled by 1/h
  int index = 0;
  for (int i = 0 ; i < N ; i++)
    {
      //Fill top boundary rows
      if (i % n == 0)
	{
	  col_ind[index] = i;
	  row_ind[index] = i;
	  values[index] = 1.0/h;

	  index++;
	  
	  col_ind[index] = i+1;
	  row_ind[index] = i;
	  values[index] = -1.0/h;
	  
	  index++;
	}	  
      // fill bottom boundary rows
      else if (i % n == n-1)
	{
	  col_ind[index] = i-1;
	  row_ind[index] = i;
	  values[index] = -1.0/h;

	  index++;
	  //printf("%d\n", i);
	  col_ind[index] = i;
	  row_ind[index] = i;
	  values[index] = 1.0/h;

	  index++;
	}
      // fill interior rows
      else
	{
	  col_ind[index] = i-1;
	  row_ind[index] = i;
	  values[index] = -1.0/h;

	  index++;

	  col_ind[index] = i;
	  row_ind[index] = i;
	  values[index] = 2.0/h;

	  index++;
	  
	  col_ind[index] = i+1;
	  row_ind[index] = i;
	  values[index] = -1.0/h;

	  index++;
      }
    }
  
  Ay->row_index = row_ind;
  Ay->col_index = col_ind;
  Ay->values = values;
  Ay->nnz = nnz;
  
}


/*
Constructs inverse of the quadtrature matrix H
  parameters:
  h        length between nodes in both spatial directions
  n        number of nodes in each physical dimension
  Hinv     memory to fill
*/
void constructHinv(double h, int n, sparseCOO* Hinv)
{
  int nnz = n*n;
  int N = n*n;

  int* row_ind = malloc(sizeof(int) * nnz);
  int* col_ind = malloc(sizeof(int) * nnz);
  double* values = malloc(sizeof(double) * nnz);

  int index = 0;
  for(int i = 0 ; i < N ; i++)
    {
      if (i % n == 0)
	{
	  col_ind[index] = i;
	  row_ind[index] = i;
	  values[index] = 2.0/h;
	  
	  index++;
	}
      else if ( i % n == (n-1))
	{
	  col_ind[index] = i;
	  row_ind[index] = i;
	  values[index] = 2.0/h;

	  index++;
	}
      else
	{
	  col_ind[index] = i;
	  row_ind[index] = i;
	  values[index] = 1.0/h;

	  index++;
	}
      
    }
  
  Hinv->row_index = row_ind;
  Hinv->col_index = col_ind;
  Hinv->values = values;
  Hinv->nnz = nnz;
}


/*
matrix free implementation of HinvY over entire volume
(only used for construction of A in matrix verisons of solving poissons)
*/
void HinvY(double h, int n, sparseCOO *mat)
{

  int row_ind;

  for (int index = 0 ; index < mat->nnz ; index++)
    {
      row_ind = mat->row_index[index];
      if (row_ind % n == 0 || row_ind % n == (n-1))
	  mat->values[index] *= 2.0/h;
      else
	mat->values[index] *= 1.0/h;
    }
}



/*
matrix free implementation of HinvX over entire volume
(only used for construction of A in matrix verisons of solving poissons)
*/
void HinvX(double h, int n, sparseCOO *mat)
{

  int row_ind;

  for (int index = 0 ; index < mat->nnz ; index++)
    {
      row_ind = mat->row_index[index];
      if (row_ind < n || row_ind > n*n - (n+1))
	  mat->values[index] *= 2.0/h;
      else
	mat->values[index] *= 1.0/h;
    }
}


/*
matrix free implementation of H over entire volume in both directions.
double integral Quadtrature over volume.
(only used for construction of A in matrix verisons of solving poissons)
*/
void HinteriorM(int n, sparseCOO *mat)
{

  int row_ind;

  for (int index = 0 ; index < mat->nnz ; index++)
    {
      row_ind = mat->row_index[index];

      // corners of domain
      if (row_ind  == 0 || row_ind == n*n - 1
	  || row_ind == n-1 || row_ind == n*n - n)
	{
	 mat->values[index] *= .25/((n-1)*(n-1));
	}
      // N S  boundaries
      else if (row_ind % n == 0 || row_ind % n == (n-1))
	{
	  mat->values[index] *= .5/((n-1)*(n-1));
	}
      // E W boundaries
      else if (row_ind < n-1 || row_ind > n*n - n - 1)
	{
	  mat->values[index] *= .5/((n-1)*(n-1));
	}
      else
	{
	  mat->values[index] *= 1.0/((n-1)*(n-1));
	}
    }
}

/* 
   matrix free takes boundary Hinv depedent on side of vector.
*/
void bHinv(double h, int n, double *vec, char side, double scale)
{
  
  int i;
  if (side == 'W')
    {
      for (i = 0 ; i < n ; i++)
	vec[i] *= (scale*2.0)/h;
    }
  if (side == 'E')
    {
      for (i = n*(n-1) ; i < n*n ; i++)
	vec[i] *= (scale*2.0)/h;
    }
  if (side == 'S')
    {
      for (i = 0 ; i < n*n ; i++)
	if (i % n == 0)
	  vec[i] *= (scale*2.0)/h;
    }
  if (side == 'N')
    {
      for (i = 0 ; i < n*n ; i++)
	if (i % n == (n-1))
	  vec[i] *= (scale*2.0)/h;
    }
}

  
void SATW (int n, double h, double alphaW, double beta, sparseCOO* SATW)
{

  int nnz = 3 * n;
  int* row_ind = malloc(sizeof(int) * nnz);
  int* col_ind = malloc(sizeof(int) * nnz);
  double* values = malloc(sizeof(double) * nnz);

  int i;
  int index = 0;
  
  for (i = 0 ; i < n ; i++)
    {

      row_ind[index] = i; 
      col_ind[index] = i;
      // first term quadrature second dirch penalty, third stability proof
      // term
      values[index] = ((2.0/h) * (alphaW)) + (beta * 3.0/(h*h));

      index++;

    }

  for (i = 0; i < n ; i++)
    {
      row_ind[index] = i+n; 
      col_ind[index] = i;
      values[index] = (beta * -2.0/(h*h));

      index++;
    }
      
  
  for (i = 0 ; i < n ; i++)
    {
      row_ind[index] = i+(2*n); 
      col_ind[index] = i;
      values[index] = (beta * .5/(h*h));
      
      index++;
    }
  
  SATW->row_index = row_ind;
  SATW->col_index = col_ind;
  SATW->values = values;
  SATW->nnz = nnz;

  
  
}


void SATE(int n, double h, double alphaE, double beta, sparseCOO* SATE)
{

  int nnz = 3 * n;
  int* row_ind = malloc(sizeof(int) * nnz);
  int* col_ind = malloc(sizeof(int) * nnz);
  double* values = malloc(sizeof(double) * nnz);

  int i;
  int index = 0;


  for (i = n * (n-1) ; i < n*n ; i++)
    {
      row_ind[index] = i-(2*n); 
      col_ind[index] = i;
      values[index] = (beta * .5/(h*h));
      
      index++;
    }
  
  for (i = n * (n-1) ; i < n*n ; i++)
    {
      row_ind[index] = i-n; 
      col_ind[index] = i;
      values[index] = (beta * -2.0/(h*h));

      index++;
    }
  
  for (i = n * (n-1) ; i < n*n ; i++)
    {

      row_ind[index] = i; 
      col_ind[index] = i;
      // first term quadrature second dirch penalty, third stability proof
      // term
      values[index] = ((2.0/h) * (alphaE)) + (beta * 3.0/(h*h));

      index++;
    }

      
  SATE->row_index = row_ind;
  SATE->col_index = col_ind;
  SATE->values = values;
  SATE->nnz = nnz;

}

/*
  Gets first derivative in y operator along the north or south face, for weak
  enforcement of neumann condition.
 */
void D1ybound(double h, int n, char face, sparseCOO* D1f)
{

  int nnz = 3*n;
  
  int* row_ind = malloc(sizeof(int) * nnz);
  int* col_ind = malloc(sizeof(int) * nnz);
  double* values = malloc(sizeof(double) * nnz);
  int index = 0 ;
  int i;
  
  if (face == 'N')
    {
      for (i = 0 ; i < n*n ; i = i + n)
	{
	  //printf("helllo\n");
	  row_ind[index] = i;
	  col_ind[index] = i;
	  values[index] = 1.5/h;
	  
	  index++;
	  
	  row_ind[index] = i;
	  col_ind[index] = i+1;
	  values[index] = -2.0/h;
	  
	  index++;

	  row_ind[index] = i;
	  col_ind[index] = i+2;
	  values[index] = .5/h;
	  
	  index++;
	}
    }
	  
  else if (face == 'S')
    {
      for (i = n-1 ; i < n*n ; i = i + n)
	{

	  row_ind[index] = i;
	  col_ind[index] = i-2;
	  values[index] = .5/h;
	  
	  index++;

	  row_ind[index] = i;
	  col_ind[index] = i-1;
	  values[index] = -2.0/h;
	  
	  index++;
	  
	  row_ind[index] = i;
	  col_ind[index] = i;
	  values[index] = 1.5/h;
	  
	  index++;
	  
	}	
    }

  D1f->row_index = row_ind;
  D1f->col_index = col_ind;
  D1f->values = values;
  D1f->nnz = nnz;
    
}


void constructS0x(double h, int n, sparseCOO* S0x)
{
    int nnz =  3*n;
    int N = n*n;
    
    int* row_ind = malloc(sizeof(int) * nnz);
    int* col_ind = malloc(sizeof(int) * nnz);
    double* values = malloc(sizeof(double) * nnz);

    int index = 0;
    for(int i = 0 ; i < n ; i++)
      {

	row_ind[index] = i;
	col_ind[index] = i;
	values[index] = -1.5/h;

	index++;

	row_ind[index] = i;
	col_ind[index] = i+n;
	values[index] = 2.0/h;

	index++;

	row_ind[index] = i;
	col_ind[index] = i+(2*n);
	values[index] = -0.5/h;

	index++;
	
      }
      
    S0x->row_index = row_ind;
    S0x->col_index = col_ind;
    S0x->values = values;
    S0x->nnz = nnz;
}




/*
Constructs inverse of the quadtrature matrix H
  parameters:
  h        length between nodes in both spatial directions
  n        number of nodes in each physical dimension
  S0y     memory to fill
*/
void constructS0y(double h, int n, sparseCOO* S0y)
{
    int nnz =  3*n;
    int N = n*n;
    
    int* row_ind = malloc(sizeof(int) * nnz);
    int* col_ind = malloc(sizeof(int) * nnz);
    double* values = malloc(sizeof(double) * nnz);

    int index = 0;
    for (int i = 0 ; i < N ; i = i + n)
      {
	
	//printf("%d\n", i);
	col_ind[index] = i;
	row_ind[index] = i;
	values[index] = -1.5/h;

	index++;
	
	col_ind[index] = i+1;
	row_ind[index] = i;
	values[index] = 2.0/h;

	index++;
	  
	col_ind[index] = i+2;
	row_ind[index] = i;
	values[index] = -0.5/h;

	index++;

      }
   
    
    S0y->row_index = row_ind;
    S0y->col_index = col_ind;
    S0y->values = values;
    S0y->nnz = nnz;
}


void constructSNx(double h, int n, sparseCOO* SNx)
{
  int nnz =  3*n;
  int N = n*n;
  
  int* row_ind = malloc(sizeof(int) * nnz);
  int* col_ind = malloc(sizeof(int) * nnz);
  double* values = malloc(sizeof(double) * nnz);

  int index = 0;
  for(int i = (n*n)-(n) ; i < N ; i++)
    {

      
      
      
      row_ind[index] = i;
      col_ind[index] = i - (2*n);
      values[index] = .5/h;

      index++;
	
      row_ind[index] = i;
      col_ind[index] = i-n;
      values[index] = -2.0/h;

      index++;

      row_ind[index] = i;
      col_ind[index] = i;
      values[index] = 1.5/h;

      index++;
      
    }


  SNx->row_index = row_ind;
  SNx->col_index = col_ind;
  SNx->values = values;
  SNx->nnz = nnz;

  
}


/*
Constructs inverse of the quadtrature matrix H
  parameters:
  h        length between nodes in both spatial directions
  n        number of nodes in each physical dimension
  Hinv     memory to fill
*/
void constructSNy(double h, int n, sparseCOO* SNy)
{
  int nnz =  3*n;
  int N = n*n;
  
  int* row_ind = malloc(sizeof(int) * nnz);
  int* col_ind = malloc(sizeof(int) * nnz);
  double* values = malloc(sizeof(double) * nnz);

  int index = 0;
  //#pragma omp parallel for 
  for (int i = n-1 ; i < N ; i = i + n)
    {
      
      
      col_ind[index] = i-2;
      row_ind[index] = i;
      values[index] = 0.5/h;
      
      index++;
      
      col_ind[index] = i-1;
      row_ind[index] = i;
      values[index] = -2.0/h;
      
      index++;
	
      col_ind[index] = i;
      row_ind[index] = i;
      values[index] = 1.5/h;

      index++;

    }

  

  SNy->row_index = row_ind;
  SNy->col_index = col_ind;
  SNy->values = values;
  SNy->nnz = nnz;

  
}



/* 
Constructs the D2y matrix which is D2y = H^-1(-Ay - S0y + Sny)
   parameters:
   h        length between nodes in both spatial directions
   n        number of nodes in both spatial directions
   nnz      number of non-zeros in D2y (should be 3 per row)
   b_22     coefficent from equation
   D2y      memory passed to fill with

*/
sparseCOO* constructD2y(double h, int n, int nnz, double b_22)
{

  sparseCOO* Ay = malloc(sizeof(sparseCOO));
  sparseCOO* S0y = malloc(sizeof(sparseCOO));
  sparseCOO* SNy = malloc(sizeof(sparseCOO));

  constructAy(h, n, Ay);
  constructS0y(h, n, S0y);
  constructSNy(h, n, SNy);
  
  sparseCOO* D2y = addCOO(addCOO(SNy, S0y, -1.0), Ay, -1.0);
  HinvY(h, n, D2y);
  //scaleCOO(b_22, D2y);

  
  freeCOO(Ay);
  freeCOO(S0y);
  freeCOO(SNy);

  return D2y;
  
}

/* 
Constructs the Dx matrix which is Dx = H^-1(-Ax - S0x + Snx)
   parameters:
   h        length between nodes in both spatial directions
   n        number of nodes in both spatial directions
   nnz      number of non-zeros in D2y (should be 3 per row)
   b_22     coefficent from equation
   D2y      memory passed to fill with

*/
sparseCOO* constructD2x(double h, int n, int nnz, double b_11)
{

  sparseCOO* Ax = malloc(sizeof(sparseCOO));
  sparseCOO* S0x = malloc(sizeof(sparseCOO));
  sparseCOO* SNx = malloc(sizeof(sparseCOO));
    
  constructAx(h, n, Ax);
  constructS0x(h, n, S0x);
  constructSNx(h, n, SNx);

  sparseCOO* D2x = addCOO(addCOO(SNx, S0x, -1.0), Ax, -1.0);
  HinvX(h, n, D2x);
  //scaleCOO(b_11, D2x);

  freeCOO(Ax);
  freeCOO(S0x);
  freeCOO(SNx);
  
  return D2x;
   
}

 
sparseCOO* constructA(double h, int n, int nnzD2, double b_11,
		      double b_22, double alphaE, double alphaW,
		      double alphaN, double alphaS, double beta)
{

  sparseCOO* D2y = constructD2y(h, n, nnzD2, b_22);
  sparseCOO* D2x = constructD2x(h, n, nnzD2, b_11);

  sparseCOO* A = addCOO(D2x, D2y, 1.0);

  freeCOO(D2x);
  freeCOO(D2y);
  
  //printf("hello\n");
  sparseCOO* ESAT = malloc(sizeof(sparseCOO));
  SATE(n, h, alphaE, beta, ESAT);
  //printCOOns(ESAT, n*n);
  A = addCOO(A, ESAT, 1.0);
  freeCOO(ESAT);
  //printf("hello\n");

  sparseCOO* WSAT = malloc(sizeof(sparseCOO));
  SATW(n, h, alphaW, beta, WSAT);
  A = addCOO(A, WSAT, 1.0);
  //printCOOns(WSAT, n*n);
  freeCOO(WSAT);
  sparseCOO* NSAT = malloc(sizeof(sparseCOO));
  D1ybound(h, n, 'N', NSAT);
  // do boundary piece of Hinv
  scaleCOO(alphaN*(2.0/h), NSAT);
  A = addCOO(A, NSAT, 1.0);
  freeCOO(NSAT);
  
  
  sparseCOO* SSAT = malloc(sizeof(sparseCOO));
  D1ybound(h, n,'S', SSAT);
  // do boundary piece of Hinv
  scaleCOO(alphaS*2.0/h, SSAT);
  A = addCOO(A, SSAT, 1.0);
  freeCOO(SSAT);
  HinteriorM(n, A);

  return A;
}


void constructbVec(int n, double h, double b_11, double b_22,
		      double* gE, double* gW, double* gN, double* gS,
		      double alphaE, double alphaW, double alphaN,
		      double alphaS, double beta, double *b)
{


  sparseCOO* ESATb = malloc(sizeof(sparseCOO));
  SATE(n, h, alphaE, beta, ESATb);
  sparseCOO* WSATb = malloc(sizeof(sparseCOO));
  SATW(n, h, alphaW, beta, WSATb);

  //printCOOns(ESATb, n*n);
  
  double* bE = malloc(sizeof(double) * n * n);
  double* bW = malloc(sizeof(double) * n * n);

  spmv_coo(ESATb, n*n, n*n, gE, bE);
  spmv_coo(WSATb, n*n, n*n, gW, bW);
  
  double* bS = malloc(sizeof(double) * n * n);
  memcpy(bS, gS, sizeof(double) * n * n);
  bHinv(h, n, bS, 'S', alphaS);

  double*bN = malloc(sizeof(double) * n * n);
  memcpy(bN, gN, sizeof(double) * n * n);
  bHinv(h, n, bN, 'N', alphaN);

  vec_add(n*n, 1.0, bE, bW, b);
  vec_add(n*n, 1.0, bS, b, b);
  vec_add(n*n, 1.0, bN, b, b);

  freeCOO(ESATb);
  freeCOO(WSATb);
  free(bS);
  free(bN);
  free(bE);
  free(bW);
  
}





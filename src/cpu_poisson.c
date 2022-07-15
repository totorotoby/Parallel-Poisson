#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cpu.h"


/*
  fill each boudnary vector g with boundary condition for that face in global way.
 */
void get_boundaries(int n, double h, double* gE, double* gW,
		    double* gN, double* gS)
{

  int sc = 0;
  int nc = 0;
  int es = 0;
  for (int i = 0 ; i < n*n ; i++)
    {
      gW[i] = 0;
      gS[i] = 0;
      gN[i] = 0;
      gE[i] = 0;
      if (i < n)
	gW[i] = sin(pi*(h*i));
      if (i % n == 0)
	{
	  gS[i] = -pi*cos(pi*(h*sc));
	  sc++;
	}
      if (i % n == (n-1))
	{	  
	  gN[i] = pi*cos(pi*(h*nc) + pi);
	  nc++;
	}
      if (i >= n*n - n)
	{
	  gE[i] = -sin(pi*(h*es));
	  es++;
	}
    }
}


double* ue_vec(int n, double h)
{

  double* ue = malloc(sizeof(double) * n * n);
  
  int count = 0;
  int i;
  int j;
  for (i = 0; i<n ; i++)
    {
      for (j = 0; j<n ; j++)
	{
	  ue[count] = sin(pi*(i*h) + pi*(j*h));
	  count++;
	}
    }

  return ue;
  
}


/*
  ue(x,y) = sin(pi*x + pi*y)
  when plugging in get out that the source is
  f(x,y) = 2pi^2*ue
 */
void add_source(int n, double h, double* vec)
{

  int count = 0;
  int i;
  int j;
  for (i = 0; i<n ; i++)
    {
      for (j = 0; j<n ; j++)
	{
	  vec[count] += -2*pi*pi*(sin(pi*(i*h) + pi*(j*h)));
	  count++;
	}
    }
}

void printVec(double *x, int n)
{
  printf("\n%dx1 vector:\n", n);
  for (int i = 0 ; i < n ; i++)
    printf("%f\n", x[i]);

  printf("\n");
}

int cg(unsigned int* csr_row_ptr, unsigned int* csr_col_ind,
       double* csr_vals, int m, int n, int nnz,
       double* x, double* b, int max_iter, double tol)
{
    // COMPLETE THIS FUNCTION

    // set up the workspace
    double* rk = (double*) malloc(sizeof(double) * m);
    double* pk = (double*) malloc(sizeof(double) * m);
    double* Ap = (double*) malloc(sizeof(double) * m);

    double residual = 0.0;
    double alpha;
    

    // r0 = b - Ax
    spmv(csr_row_ptr, csr_col_ind, csr_vals, m, n, nnz, x, Ap);
    vec_add(m, -1.0, Ap, b, rk);
    
    // if r0 is sufficiently small, return x0 as the result
    residual = dnrm2(m, rk, 1);
    //print_vec(rk, 10);
    
    if(residual < tol) {
        fprintf(stdout, "\tInput is the solution\n");
        return 0;
    } else {
      //fprintf(stdout, "\n\tInitial residual is %f\n", residual);
    }

    // p0 = r0
    memcpy(pk, rk, sizeof(double) * m);

    int k = 0;
    double residual_new = 0.0;
    // repeat until convergence of max iterations has been reached
    for(int i = 0; i < max_iter; i++) {
      
     // A * p
      spmv(csr_row_ptr, csr_col_ind, csr_vals, m, n, nnz, pk, Ap);

      // d (stored in alpha) = p^T * A * p

      alpha = ddot(m, pk, 1, Ap, 1);

      // alpha = (r^t * r) / d;
      alpha = ddot(m, rk, 1, rk, 1) / alpha;
      
      
      
      // xk = xk + alpha * pk
      vec_add(m, alpha, pk, x, x);

      //residual(not actually residual, but for storage) =  rk^t * rk
      residual = ddot(m, rk, 1, rk, 1);

      // rk+1 = rk - alpha * A*p
      vec_add(m, -alpha, Ap, rk, rk);
      
      // d (stored in alpha) = rk+1^t * rk+1
      alpha = ddot(m, rk, 1, rk, 1);
      // could call dnrm2 to get this, but we need to save the intermediate dot product for later.
      residual_new = sqrt(alpha);
        if(residual_new < tol) {
	  //fprintf(stdout, "\tSolution calculated. Final residual: %f\n", 
	  //residual_new);
            break;
        } else {
	  //fprintf(stdout, "\tIt: %d\tresidual: %f\n", k, residual_new);
        }

	// beta (stored in alpha)  = (d/residual) rk+1^t rk+1 / rk^t rk
	alpha = alpha/residual;

        // p = r + beta * p 
	vec_add(m, alpha, pk, rk, pk);
        k++;
    }

    free(rk);
    free(pk);
    free(Ap);
}


int cgmatfree(int m, int n, double h, double b_11, double b_22,
	      double alphaE, double alphaW, double alphaN,
	      double alphaS, double beta, double* x, double* b,
	      int max_iter, double tol)
{
    // COMPLETE THIS FUNCTION

    // set up the workspace
    double* rk = (double*) malloc(sizeof(double) * m);
    double* pk = (double*) malloc(sizeof(double) * m);
    double* Ap = zeros(m);

    double residual = 0.0;
    double alpha;
    

    // r0 = b - Ax
    Amatfree(h, n, b_11, b_22, alphaE, alphaW,
	     alphaN, alphaS, beta, x, Ap);
    vec_add(m, -1.0, Ap, b, rk);
    
    // if r0 is sufficiently small, return x0 as the result
    residual = dnrm2(m, rk, 1);
    //print_vec(rk, 10);
    
    if(residual < tol) {
        fprintf(stdout, "\tInput is the solution\n");
        return 0;
    } else {
      //fprintf(stdout, "\n\tInitial residual is %f\n", residual);
    }

    // p0 = r0
    memcpy(pk, rk, sizeof(double) * m);

    int k = 0;
    double residual_new = 0.0;
    // repeat until convergence of max iterations has been reached
    for(int i = 0; i < max_iter; i++) {
      
     // A * p
      Amatfree(h, n, b_11, b_22, alphaE, alphaW,
	       alphaN, alphaS, beta, pk, Ap);
      
      // d (stored in alpha) = p^T * A * p
      alpha = ddot(m, pk, 1, Ap, 1);

      // alpha = (r^t * r) / d;
      alpha = ddot(m, rk, 1, rk, 1) / alpha;
      
      // xk = xk + alpha * pk
      vec_add(m, alpha, pk, x, x);

      //residual(not actually residual, but for storage) =  rk^t * rk
      residual = ddot(m, rk, 1, rk, 1);

      // rk+1 = rk - alpha * A*p
      vec_add(m, -alpha, Ap, rk, rk);
      
      // d (stored in alpha) = rk+1^t * rk+1
      alpha = ddot(m, rk, 1, rk, 1);
      // could call dnrm2 to get this, but we need to save the intermediate dot product for later.
      residual_new = sqrt(alpha);
        if(residual_new < tol) {
	  //fprintf(stdout, "\tSolution calculated. Final residual: %f\n", 
	  //residual_new);
            break;
        } else {
	  //fprintf(stdout, "\tIt: %d\tresidual: %f\n", k, residual_new);
        }

	// beta (stored in alpha)  = (d/residual) rk+1^t rk+1 / rk^t rk
	alpha = alpha/residual;

        // p = r + beta * p 
	vec_add(m, alpha, pk, rk, pk);
        k++;
    }

    free(rk);
    free(pk);
    free(Ap);
}


double* zeros(int n)
{
  double* zeros = malloc(sizeof(double)*n);
  #pragma omp parallel for
  for (int i = 0 ; i < n ; i++)
    zeros[i] = 0.0;

  return zeros;
}

void zerout(double* vec, int n)
{
  #pragma omp parrallel for
  for (int i = 0 ; i < n ; i++)
    vec[i] = 0.0;

}
  


void testMatFree(double h, int n,  double b_11,
		 double b_22, double alphaE, double alphaW,
		 double alphaN, double alphaS, double beta,
		 double* gE, double* gW, double* gN, double* gS)
{
  
  // b vector construction for mat free ops
  double* b = malloc(sizeof(double) * n*n);
  constructbVec(n, h, b_11, b_22, gE, gW, gN, gS,
		alphaE, alphaW, alphaN, alphaS, beta, b);
  // b vector construction for mat ops
  double* b_mat = malloc(sizeof(double) * n*n);
  constructbVec(n, h, b_11, b_22, gE, gW, gN, gS,
		alphaE, alphaW, alphaN, alphaS, beta, b_mat);
	  
  
  sparseCOO* Axt = malloc(sizeof(sparseCOO));
  sparseCOO* S0xt = malloc(sizeof(sparseCOO));
  sparseCOO* SNxt = malloc(sizeof(sparseCOO));
  constructAx(h, n, Axt);
  constructS0x(h, n, S0xt);
  constructSNx(h, n, SNxt);
  sparseCOO* Ayt = malloc(sizeof(sparseCOO));
  sparseCOO* S0yt = malloc(sizeof(sparseCOO));
  sparseCOO* SNyt = malloc(sizeof(sparseCOO));
  constructAy(h, n, Ayt);
  constructS0y(h, n, S0yt);
  constructSNy(h, n, SNyt);

  sparseCOO* ESATt = malloc(sizeof(sparseCOO));
  SATE(n, h, alphaE, beta, ESATt);
  sparseCOO* WSATt = malloc(sizeof(sparseCOO));
  SATW(n, h, alphaW, beta, WSATt);
  sparseCOO* NSATt = malloc(sizeof(sparseCOO));
  D1ybound(h, n, 'N', NSATt);
  scaleCOO(alphaN*(2.0/h), NSATt);
  sparseCOO* SSATt = malloc(sizeof(sparseCOO));
  D1ybound(h, n,'S', SSATt);
  scaleCOO(alphaS*2.0/h, SSATt);

  sparseCOO* Acoo = constructA(h, n, 3*n*n, b_11, b_22,
			       alphaE, alphaW, alphaN,
			       alphaS, beta);
  

  double *res_mat = malloc(sizeof(double)*n*n);
  double *res = zeros(n*n);
  double *diff = malloc(sizeof(double)*n*n);
  double error;

  spmv_coo(Axt, n*n, n*n, b_mat, res_mat);
  Ax(h, n, b, res);
  vec_add(n*n, -1, res, res_mat, diff);
  error = dnrm2(n*n, diff, 1);
  printf("Checking errors between matrix and matrix-free operators:\n");
  printf("Ax: %f\n", error);
  zerout(res, n*n);

  spmv_coo(S0xt, n*n, n*n, b_mat, res_mat);
  S0x(h, n, b, res);
  vec_add(n*n, -1, res, res_mat, diff);
  error = dnrm2(n*n, diff, 1);
  printf("Checking errors between matrix and matrix-free operators:\n");
  printf("S0x: %f\n", error);
  zerout(res, n*n);

  spmv_coo(SNxt, n*n, n*n, b_mat, res_mat);
  SNx(h, n, b, res);
  vec_add(n*n, -1, res, res_mat, diff);
  error = dnrm2(n*n, diff, 1);
  printf("Checking errors between matrix and matrix-free operators:\n");
  printf("SNx: %f\n", error);
  zerout(res, n*n);

  
  spmv_coo(Ayt, n*n, n*n, b_mat, res_mat);
  Ay(h, n, b, res);
  vec_add(n*n, -1, res, res_mat, diff);
  error = dnrm2(n*n, diff, 1);
  printf("Checking errors between matrix and matrix-free operators:\n");
  printf("Ay: %f\n", error);
  zerout(res, n*n);

  spmv_coo(S0yt, n*n, n*n, b_mat, res_mat);
  S0y(h, n, b, res);
  vec_add(n*n, -1, res, res_mat, diff);
  error = dnrm2(n*n, diff, 1);
  printf("Checking errors between matrix and matrix-free operators:\n");
  printf("S0y: %f\n", error);
  zerout(res, n*n);

  spmv_coo(SNyt, n*n, n*n, b_mat, res_mat);
  SNy(h, n, b, res);
  vec_add(n*n, -1, res, res_mat, diff);
  error = dnrm2(n*n, diff, 1);
  printf("Checking errors between matrix and matrix-free operators:\n");
  printf("SNy: %f\n", error);
  zerout(res, n*n);

  spmv_coo(ESATt, n*n, n*n, b_mat, res_mat);
  SATEmfree(h, n, alphaE, beta, b, res);
  vec_add(n*n, -1, res, res_mat, diff);
  error = dnrm2(n*n, diff, 1);
  printf("Checking errors between matrix and matrix-free operators:\n");
  printf("SATE: %f\n", error);
  zerout(res, n*n);
  
  spmv_coo(WSATt, n*n, n*n, b_mat, res_mat);
  SATWmfree(h, n, alphaW, beta, b, res);
  vec_add(n*n, -1, res, res_mat, diff);
  error = dnrm2(n*n, diff, 1);
  printf("Checking errors between matrix and matrix-free operators:\n");
  printf("SATW: %f\n", error);
  zerout(res, n*n);
  
  spmv_coo(NSATt, n*n, n*n, b_mat, res_mat);
  SATNmfree(h, n, alphaN, b, res);
  vec_add(n*n, -1, res, res_mat, diff);
  error = dnrm2(n*n, diff, 1);
  printf("Checking errors between matrix and matrix-free operators:\n");
  printf("SATN: %f\n", error);
  zerout(res, n*n);
  
  spmv_coo(SSATt, n*n, n*n, b_mat, res_mat);
  SATSmfree(h, n, alphaS, b, res);
  vec_add(n*n, -1, res, res_mat, diff);
  error = dnrm2(n*n, diff, 1);
  printf("Checking errors between matrix and matrix-free operators:\n");
  printf("SATS: %f\n", error);
  zerout(res, n*n);

  spmv_coo(Acoo, n*n, n*n, b_mat, res_mat);
  Amatfree(h, n, b_11, b_22, alphaE, alphaW,
	   alphaN, alphaS, beta, b, res);
  vec_add(n*n, -1, res, res_mat, diff);
  error = dnrm2(n*n, diff, 1);
  printf("Checking errors between matrix and matrix-free operators:\n");
  printf("A: %f\n", error);
  zerout(res, n*n);
  
}


/*
Solves Poissons equation on the unit square on a cpu using order 2
SBP operators, with problem specifications in paper.

usage:
./cpu_poisson <# of nodes in each dimension> <b_11> <b_22> <matrix-free>
 */
int main(int argc, char** argv)
{

  // print usage if wrong number of arguments
  if (argc != 3)
    {
    printf("./cpu_poisson <# of nodes in each dimension> <matrix-free>\n");
    exit(0);
    }
  
  
  // Initialize timess
  double timer[2];
  uint64_t t0;
  for(unsigned int i = 0; i < 2; i++) {
    timer[i] = 0.0;
  }
  
  InitTSC();

  int mat_free_opt = atoi(argv[2]);

  
  int n = atoi(argv[1]);
  printf("number of nodes per dimensions %d\n", n);
  
  double h = 1.0/(n-1);
  printf("h is %f\n", h);
  // material parameters
  double b_11 = 1;
  double b_22 = 1;
  
  // number of solution points (length of solution vector)
  int N = n*n;
     
  // boundary conditions
  double* gE = malloc(sizeof(double) * n * n);
  double* gW = malloc(sizeof(double) * n * n);
  double* gN = malloc(sizeof(double) * n * n);
  double* gS = malloc(sizeof(double) * n * n);
  get_boundaries(n, h, gE, gW, gN, gS);

    
  
  // penalty parameters
  double alphaN = -1.0;
  double alphaS = -1.0;
  double alphaE = -13/h;
  double alphaW = -13/h;
  double beta = 1.0;

  
  if (!mat_free_opt)
    {
      t0 = ReadTSC();
      // number of non_zeros in single 2d D2 matrix (not both added)
      int nnzD2 = 3 * n * n;
      
      //generates A matrix in Ax = b system that must be solved
      sparseCOO* Acoo = constructA(h, n, nnzD2, b_11, b_22,
				   alphaE, alphaW, alphaN, alphaS, beta);

      int* csr_row_ptr = NULL; 
      int* csr_col_index = NULL;  
      double* csr_values = NULL;
	  
      convert_coo_to_csr(Acoo, N, N, &csr_row_ptr,
			 &csr_col_index, &csr_values);

      // fill b vector
      double* b = malloc(sizeof(double) * N);
      constructbVec(n, h, b_11, b_22, gE, gW, gN, gS,
		    alphaE, alphaW, alphaN, alphaS, beta, b);
      
      add_source(n, h, b);
      HinteriorV(n, b, b);
      timer[0] += ElapsedTime(ReadTSC() - t0);
      
      //solution vector
      double *u = zeros(N);

      t0 = ReadTSC();
      cg(csr_row_ptr, csr_col_index, csr_values, N, N, Acoo->nnz, u, b, n*n, .00000001);
      timer[1] += ElapsedTime(ReadTSC() - t0);
      
      // mms solution
      double* ue = ue_vec(n , h);
      double* error1 = malloc(sizeof(double)*N);
      vec_add(n*n, -1.0, u, ue, error1);
      double* error2 = malloc(sizeof(double)*N);
      vec_add(n*n, -1.0, u, ue, error2);
      HinteriorV(n, error1, error1);
      double error = ddot(n*n, error1, 1, error2, 1);
      error = sqrt(error);
      printf("error: %e\n", error);

      free(u);
      free(b);
      free(error1);
      free(error2);
      freeCOO(Acoo);

      printf("Time to construct operators: %f\n", timer[0]);
      printf("Time to solve system: %f\n", timer[1]);
      
    }
    

  else
    {
	  
      //solution vector
      double *u = zeros(N);

      // b vector construction
      double* b = malloc(sizeof(double) * N);
      constructbVec(n, h, b_11, b_22, gE, gW, gN, gS,
		    alphaE, alphaW, alphaN, alphaS, beta, b);
      add_source(n, h, b);
      HinteriorV(n, b, b);
	  

      //testMatFree(h, n,  b_11, b_22, alphaE, alphaW,
      //alphaN, alphaS, beta, gE, gW, gN, gS);

      t0 = ReadTSC();
      cgmatfree(n*n, n, h, b_11, b_22, alphaE, alphaW, alphaN,
		alphaS, beta, u, b, n*n, .000001);
      timer[0] += ElapsedTime(ReadTSC() - t0);
	  
      double* ue = ue_vec(n , h);
      double* error1 = malloc(sizeof(double)*N);
      vec_add(n*n, -1.0, u, ue, error1);
      double* error2 = malloc(sizeof(double)*N);
      vec_add(n*n, -1.0, u, ue, error2);
      HinteriorV(n, error1, error1);
      double error = ddot(n*n, error1, 1, error2, 1);
      error = sqrt(error);
      printf("error: %e\n", error);
      
      free(b);
      free(u);
      free(ue);
      free(error1);
      free(error2);
      free(gE);
      free(gW);
      free(gN);
      free(gS);
      

      printf("Time to solve system matrix-free: %f\n", timer[0]);
	  
    }


      
}


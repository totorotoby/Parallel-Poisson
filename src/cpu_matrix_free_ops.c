#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cpu.h"

#define n2_vec zeros(n*n)
#define allocate malloc(sizeof(double)*n*n)

void Amatfree(double h, int n,  double b_11,
	      double b_22, double alphaE, double alphaW,
	      double alphaN, double alphaS, double beta,
	      double* uin, double* uout)
{
  
  //D2y
  double* uS0y = n2_vec;
  S0y(h, n, uin, uS0y);
  double* uSNy = n2_vec;
  SNy(h, n, uin, uSNy);
  
  double* uSy = n2_vec;
  // uSy = uSNy - uS0y 
  vec_add(n*n, -1.0, uS0y, uSNy, uSy);
  free(uS0y);
  free(uSNy);
  
  //get uAy
  Ay(h, n, uin, uout);
  // HD2y = -uAy + uSy
  vec_add(n*n, -1.0, uout, uSy, uout);
  free(uSy);
    // uD2y = Hinv(-uAy + uSy)
  HinvYmfree(h, n, uout, uout);

  
  //D2x
  double* uS0x = n2_vec;
  S0x(h, n, uin, uS0x);
  double* uSNx = n2_vec;
  SNx(h, n, uin, uSNx);
  double* uSx = n2_vec;
  // uSx = uSNx - uS0x
  vec_add(n*n, -1.0, uS0x, uSNx, uSx);
  free(uS0x);
  free(uSNx);
  
  double* uAx = n2_vec;
  //get uAx
  Ax(h, n, uin, uAx);

  // HD2x = -UAx + uSx
  vec_add(n*n, -1.0, uAx, uSx, uAx);
  free(uSx);
  // D2x = Hinv(-UAx + uSx)
  HinvXmfree(h, n, uAx, uAx);

  //D2 = D2x + D2y
  vec_add(n*n, 1.0, uAx, uout, uout);
  free(uAx);
  
  // D2 += uESAT
  double* uESAT = n2_vec;
  SATEmfree(h, n, alphaE, beta, uin, uESAT);
  vec_add(n*n, 1.0, uESAT, uout, uout);
  free(uESAT);

  // D2 += uWSAT
  double* uWSAT = n2_vec;
  SATWmfree(h, n, alphaW, beta, uin, uWSAT);
  vec_add(n*n, 1.0, uWSAT, uout, uout);
  free(uWSAT);

  // D2 += uNSAT
  double* uNSAT = n2_vec;
  SATNmfree(h, n, alphaN, uin, uNSAT);
  vec_add(n*n, 1.0, uNSAT, uout, uout);
  free(uNSAT);

  // D2 += uSSAT
  double* uSSAT = n2_vec;
  SATSmfree(h, n, alphaS, uin, uSSAT);
  vec_add(n*n, 1.0, uSSAT, uout, uout);
  free(uSSAT);
  
  HinteriorV(n, uout, uout);
}

void S0y(double h, int n, double* uin, double* uout)
{
 
  #pragma omp parallel for
  for (int i = 0 ; i < n*n ; i = i+n)
      uout[i] = (-1.5*uin[i] + 2.0*uin[i+1] - .5*uin[i+2])/h;
}

void SNy(double h, int n, double* uin, double* uout)
{
 
  #pragma omp parallel for
  for (int i = n-1 ; i < n*n ; i = i+n)
    uout[i] = (1.5*uin[i] - 2.0*uin[i-1] + .5*uin[i-2])/h;
   
}

void S0x(double h, int n, double* uin, double* uout)
{
   #pragma omp parallel for
  for (int i = 0 ; i < n ; i++)
    uout[i] = (-1.5*uin[i] + 2.0*uin[i+n] - .5*uin[i+2*n])/h;
    
}

void SNx(double h, int n, double* uin, double* uout)
{
   #pragma omp parallel for
  for (int i = n*n - (n) ; i < n*n ; i++)
    {
      uout[i] = (1.5*uin[i] - 2.0*uin[i-n] + .5*uin[i-2*n])/h;
    }
}

void HinvYmfree(double h, int n, double* uin, double* uout)
{

#pragma omp parallel for
  for (int i = 0 ; i < n*n ; i++)
    {
      if (i % n == 0 || i % n == (n-1))
	uout[i] = 2.0*uin[i]/h;
      else
	uout[i] = 1.0*uin[i]/h;
    }

}

void HinvXmfree(double h, int n, double* uin, double* uout)
{
  
  #pragma omp parallel for
  for (int i = 0 ; i < n*n ; i++)
    {
      if (i < n || i >= n*n - n)
	uout[i] = 2.0*uin[i]/h;
      else
	uout[i] = 1.0*uin[i]/h;
    }
}

void Ay(double h, int n, double* uin, double* uout)
{
  
  #pragma omp parallel for
  for (int i = 0 ; i < n*n ; i++)
    {
      if (i % n == 0)	  
	  uout[i] = (uin[i] - uin[i+1])/h;
      else if (i % n == (n-1))
	uout[i] = (uin[i] - uin[i-1])/h;
      else
	uout[i] = (-uin[i-1] + 2*uin[i] - uin[i+1])/h;
    }
}



void Ax(double h, int n, double* uin, double* uout)
{
  
  #pragma omp parallel for
  for (int i = 0 ; i < n*n ; i++)
    {
      if (i < n)
	uout[i] = (uin[i] - uin[i+n])/h;
      else if (i > n*n - (n+1))
	uout[i] = (uin[i] - uin[i-n])/h;
      else
	uout[i] = (-uin[i-n] + 2*uin[i] - uin[i+n])/h;
    }
}

void SATEmfree(double h, int n, double alphaE, double beta,
	  double* uin, double* uout)
{
  //#pragma omp parallel for
  for (int i = 0 ; i < n ; i++)
    {
      uout[(n*n - n*3) + i] = (beta *.5 *uin[n*n - n + i])/(h*h);
      uout[(n*n - n*2) + i] = (-2.0 * beta * uin[n*n - n + i])/(h*h);
      uout[(n*n - n) + i] = 2.0/h * alphaE * uin[n*n - n + i] + beta * 3.0 * uin[n*n - n + i]/(h*h);
    }

}

void SATWmfree(double h, int n, double alphaW, double beta,
	  double* uin, double* uout)
{
  #pragma omp parallel for
  for (int i = 0 ; i < 3*n ; i++)
    {
      if (i < n)
	uout[i] = 2.0/h * alphaW * uin[i] + beta * 3.0 * uin[i]/(h*h);
      else if (i < 2*n)
	uout[i] = (-2.0 * beta * uin[i - n])/(h*h);
      else
	uout[i] = (beta *.5 *uin[i - 2*n])/(h*h);
    }
}

void SATNmfree(double h, int n, double alphaN, double* uin, double* uout)
{
  #pragma omp parallel for
  for (int i = 0 ; i <= n*n - n ; i = i + n)
    {
      uout[i] = (alphaN * 2.0/h) * (1.5 * uin[i] - 2.0 * uin[i+1] + .5 * uin[i+2]) / h;
    }
}


void SATSmfree(double h, int n, double alphaS, double* uin, double* uout)
{
  #pragma omp parallel for
  for (int i = n-1 ; i < n*n ; i = i + n)
    {
      uout[i] = (alphaS * 2.0/h) * (1.5 * uin[i] - 2.0 * uin[i-1] + .5 * uin[i-2]) / h;
    }
}


/*
matrix free implementation of H over entire volume in both directions.
double integral Quadtrature over volume.
(only used for construction of A in matrix verisons of solving poissons)
*/
void HinteriorV(int n, double* uin, double* uout)
{

  for (int i = 0; i < n*n  ; i++)
    {
      // corners of domain
      if (i  == 0 || i == n*n - 1
	  || i == n-1 || i == n*n - n)
	{
	 uout[i] = .25/((n-1)*(n-1)) * uin[i];
	}
      // N S  boundaries
      else if (i % n == 0 || i % n == (n-1))
	{
	  uout[i] = .5/((n-1)*(n-1)) * uin[i];
	}
      // E W boundaries
      else if (i < n-1 || i > n*n - n - 1)
	{
	  uout[i] = .5/((n-1)*(n-1)) * uin[i];
	}
      else
	{
	  uout[i] =  1.0/((n-1)*(n-1)) * uin[i];
	}
    }  
}

#ifndef CPU
#define CPU

#include<omp.h>
#include <stdint.h>
#include <stdlib.h>

#define pi 3.1415928

typedef struct COO
{
  int* col_index;
  int* row_index;
  double* values;
  int nnz;
} sparseCOO;


typedef struct CSR
{
  int* col_index;
  int* row_ptr;
  double* values;
  int nnz;
} sparseCSR;

typedef struct COOpoint{
  int row_ind;
  int col_ind;
  double val;
} coopoint;


void get_boundaries(int N, double h, double* g1, double* g2, double* g3, double* g4);
void add_source(int n, double h, double* vec);
void printVec(double *x, int n);
double* zeros(int n);
void zerout(double* vec, int n);
double* ue_vec(int n, double h);
void SATW(int n, double h, double alphaW, double beta, sparseCOO* SATW);
void SATE(int n, double h, double alphaE, double beta, sparseCOO* SATE);
sparseCOO* addCOO(sparseCOO* A, sparseCOO* B, double c);
sparseCOO* transpose(sparseCOO* A);
int cmpfunc(const void *a, const void *b);
void freeCOO(sparseCOO* tofree);
double *addvecs(double* v1, double* v2, double scale, int length);
void vec_add(const int n, const double a, const double* x, 
             double* y, double* z);
double dnrm2(const int n, double* x, const int incx);
double ddot(const int n, double* x, const int incx, double* y, const int incy);
void convert_coo_to_csr(sparseCOO* coo, int m, int n,
			int** csr_row_ptr, int** csr_col_index,
			double** csr_values);
int cg(unsigned int* csr_row_ptr, unsigned int* csr_col_ind,
       double* csr_vals, int m, int n, int nnz,
       double* x, double* b, int max_iter, double tol);
void printCSRns(sparseCSR* toprint, int N);
sparseCOO* constructA(double h, int n, int nnzD2, double b_11,
		      double b_22, double alphaE, double alphaW,
		      double alphaN, double alphaS, double beta);
void constructbVec(int n, double h, double b_11, double b_22,
		      double* gE, double* gW, double* gN, double* gS,
		      double alphaE, double alphaW, double alphaN,
		   double alphaS, double beta, double* b);
void scaleCOO(double c, sparseCOO* mat);
sparseCOO* constructD2y(double h, int n, int nnz, double b_22);
sparseCOO* constructD2x(double h, int n, int nnz, double b_11);
void D1ybound(double h, int n, char face, sparseCOO* D1f);
void constructSNy(double h, int n, sparseCOO* SNx);
void constructSNx(double h, int n, sparseCOO* SNy);
void constructS0y(double h, int n, sparseCOO* S0y);
void constructS0x(double h, int n, sparseCOO* S0x);
void constructHinv(double h, int n, sparseCOO* Hinv);
void HinteriorM(int n, sparseCOO* mat);
void HinvY(double h, int n, sparseCOO *mat);
void HinvX(double h, int n, sparseCOO *mat);
void constructAy(double h, int n, sparseCOO*Ay);
void constructAx(double h, int n, sparseCOO*Ax);
void printCOO(sparseCOO* toprint);
void printCOOns(sparseCOO* toprint, int N);
void spmv_coo(sparseCOO* mat, int m, int n, double* vec, double *res);
void init_locks(omp_lock_t** locks, int m);
void spmv(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
          double* csr_vals, int m, int n, int nnz, 
          double* vector_x, double *res);
void testMatFree(double h, int n,  double b_11,
		 double b_22, double alphaE, double alphaW,
		 double alphaN, double alphaS, double beta,
		 double* gE, double* gW, double* gN, double* gS);



/* Matrix Freeee */

void Ay(double h, int n, double* uin, double* uout);
void Ax(double h, int n, double* uin, double* uout);

void S0y(double h, int n, double* uin, double* uout);
void SNy(double h, int n, double* uin, double* uout);
void S0x(double h, int n, double* uin, double* uout);
void SNx(double h, int n, double* uin, double* uout);

void HinvYmfree(double h, int n, double* uin, double* uout);
void HinvXmfree(double h, int n, double* uin, double* uout);
  
void SATEmfree(double h, int n, double alphaE, double beta,
	  double* uin, double* uout);
void SATWmfree(double h, int n, double alphaW, double beta,
	       double* uin, double* uout);
void SATNmfree(double h, int n, double alphaN, double* uin, double* uout);
void SATSmfree(double h, int n, double alphaS, double* uin, double* uout);

void Amatfree(double h, int n,  double b_11,
	      double b_22, double alphaE, double alphaW,
	      double alphaN, double alphaS, double beta,
	      double* uin, double* uout);
void HinteriorV(int n, double* uin, double* uout);




static inline uint64_t ReadTSC(void)
{
#if defined(__i386__)

    uint64_t x;
    __asm__ __volatile__(".byte 0x0f, 0x31":"=A"(x));
    return x;

#elif defined(__x86_64__)

    uint32_t hi, lo;
    __asm__ __volatile__("rdtsc":"=a"(lo), "=d"(hi));
    return ((uint64_t) lo) | (((uint64_t) hi) << 32);

#elif defined(__powerpc__)

    uint64_t result = 0;
    uint64_t upper, lower, tmp;
    __asm__ __volatile__("0:                  \n"
                         "\tmftbu   %0           \n"
                         "\tmftb    %1           \n"
                         "\tmftbu   %2           \n"
                         "\tcmpw    %2,%0        \n"
                         "\tbne     0b         \n":"=r"(upper), "=r"(lower),
                         "=r"(tmp)
        );
    result = upper;
    result = result << 32;
    result = result | lower;
    return result;

#endif // defined(__i386__)
}

void InitTSC(void);

double ElapsedTime(uint64_t ticks);




#endif

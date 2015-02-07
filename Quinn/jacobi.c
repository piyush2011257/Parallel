/*
Jacobi method to solve AX = b matrix system of linear equations.

       Input                : Real Symmetric Positive definite Matrix
                              and the real vector - Input_A and Vector_B
                              Read files (mdatjac.inp) for Matrix A
                              and (vdatjac.inp) for Vector b

       Description          : Input matrix is stored in n by n format.
                              Diagonal preconditioning matrix is used.
                              Rowwise block striped partitioning matrix 
                              is used.Maximum iterations is given by
                              MAX_ITERATIONS.Tolerance value is given by
                              EPSILON.

       Output               : The solution of  Ax=b on process with Rank 0
                              and the number of iterations for convergence
                              of the method.

       Necessary conditions : Number of Processes should be less than or
                              equal to 8 Input Matrix Should be Square Matrix.
                              Matrix size for Matrix A and vector size for
                              vector b should be equally striped, that is
                              Matrix size and Vector Size should be dividible
                              by the number of processes used.

*/

#include <stdio.h>
#include <assert.h>
#include <mpi.h>
#define  MAX_ITERATIONS 100

double Distance(double *X_Old, double *X_New, int n_size)
{	int  index;
	double Sum;
	Sum = 0.0;
	for(index=0; index<n_size; index++)
		Sum += (X_New[index] - X_Old[index])*(X_New[index]-X_Old[index]);
	return(Sum);
}

int main(int argc, char** argv)
{	MPI_Status status;     
	int n_size, NoofRows_Bloc, NoofRows, NoofCols;
	int Numprocs, MyRank, Root=0;
	int irow, jrow, icol, index, Iteration, GlobalRowNo;
	double **Matrix_A, *Input_A, *Input_B, *ARecv, *BRecv;
	double *X_New, *X_Old, *Bloc_X, tmp;
	FILE *fp;
	
	MPI_Init(&argc, &argv); 
	MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
	MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);
	if(MyRank == Root)
	{	if ((fp = fopen ("./marix-data-jacobi.inp", "r")) == NULL)
		{	printf("Can't open input matrix file");
			exit(-1);
		}
		fscanf(fp, "%d %d", &NoofRows,&NoofCols);
		n_size=NoofRows;
		/* ...Allocate memory and read data .....*/
		Matrix_A = (double **) malloc(n_size*sizeof(double *));
		for(irow = 0; irow < n_size; irow++)
		{	Matrix_A[irow] = (double *) malloc(n_size * sizeof(double));
			for(icol = 0; icol < n_size; icol++)
				fscanf(fp, "%lf", &Matrix_A[irow][icol]);
		}
		fclose(fp);
		if ((fp = fopen ("./vector-data-jacobi.inp", "r")) == NULL)
		{	printf("Can't open input vector file");
			exit(-1);
		}
		fscanf(fp, "%d", &NoofRows);
		n_size=NoofRows;
		Input_B  = (double *)malloc(n_size*sizeof(double));
		for (irow = 0; irow<n_size; irow++)
			fscanf(fp, "%lf",&Input_B[irow]);
		fclose(fp); 
		/* ...Convert Matrix_A into 1-D array Input_A ......*/
		Input_A  = (double *)malloc(n_size*n_size*sizeof(double));
		index    = 0;
		for(irow=0; irow<n_size; irow++)
			for(icol=0; icol<n_size; icol++)
				Input_A[index++] = Matrix_A[irow][icol];
	}
	MPI_Bcast(&NoofRows, 1, MPI_INT, Root, MPI_COMM_WORLD); 
	MPI_Bcast(&NoofCols, 1, MPI_INT, Root, MPI_COMM_WORLD); 
	if(NoofRows != NoofCols)
	{	MPI_Finalize();
		if(MyRank == 0)
		{	printf("Input Matrix Should Be Square Matrix ..... \n");
		}
		exit(-1);
	}  	
	/* .... Broad cast the size of the matrix to all ....*/
	MPI_Bcast(&n_size, 1, MPI_INT, Root, MPI_COMM_WORLD); 
	if(n_size % Numprocs != 0) 
	{	MPI_Finalize();
		if(MyRank == 0)
		{	printf("Matrix Can not be Striped Evenly ..... \n");
		}
		exit(-1);
	}
	NoofRows_Bloc = n_size/Numprocs;
	/*......Memory of input matrix and vector on each process .....*/
	ARecv = (double *) malloc (NoofRows_Bloc * n_size* sizeof(double));
	BRecv = (double *) malloc (NoofRows_Bloc * sizeof(double));
	/*......Scatter the Input Data to all process ......*/
	MPI_Scatter (Input_A, NoofRows_Bloc * n_size, MPI_DOUBLE, ARecv, NoofRows_Bloc * n_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter (Input_B, NoofRows_Bloc, MPI_DOUBLE, BRecv, NoofRows_Bloc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	X_New  = (double *) malloc (n_size * sizeof(double));
	X_Old  = (double *) malloc (n_size * sizeof(double));
	Bloc_X = (double *) malloc (NoofRows_Bloc * sizeof(double));
	/* Initailize X[i] = B[i] */
	for(irow=0; irow<NoofRows_Bloc; irow++)
		Bloc_X[irow] = BRecv[irow];
	MPI_Allgather(Bloc_X, NoofRows_Bloc, MPI_DOUBLE, X_New, NoofRows_Bloc, MPI_DOUBLE, MPI_COMM_WORLD);
	Iteration = 0;
	do
	{	for(irow=0; irow<n_size; irow++)
			X_Old[irow] = X_New[irow];
		for(irow=0; irow<NoofRows_Bloc; irow++)
		{	GlobalRowNo = (MyRank * NoofRows_Bloc) + irow;
			Bloc_X[irow] = BRecv[irow];
			index = irow * n_size;
			for(icol=0; icol<GlobalRowNo; icol++)
			{	Bloc_X[irow] -= X_Old[icol] * ARecv[index + icol];
			}
			for(icol=GlobalRowNo+1; icol<n_size; icol++)
			{	Bloc_X[irow] -= X_Old[icol] * ARecv[index + icol];
			}
			Bloc_X[irow] = Bloc_X[irow] / ARecv[irow*n_size + GlobalRowNo];
		}
		MPI_Allgather(Bloc_X, NoofRows_Bloc, MPI_DOUBLE, X_New, NoofRows_Bloc, MPI_DOUBLE, MPI_COMM_WORLD);
		Iteration++;
	}while( (Iteration < MAX_ITERATIONS) && (Distance(X_Old, X_New, n_size) >= 1.0E-24)); 
	/* .......Output vector .....*/
	if (MyRank == 0)
	{	printf ("\n");
		printf("Results of Jacobi Method on processor %d: \n", MyRank);
		printf ("\n");
		printf("Matrix Input_A \n");
		printf ("\n");
		for (irow = 0; irow < n_size; irow++)
		{	for (icol = 0; icol < n_size; icol++)
				printf("%.3lf  ", Matrix_A[irow][icol]);
			printf("\n");
		}
		printf ("\n");
		printf("Matrix Input_B \n");
		printf("\n");
		for (irow = 0; irow < n_size; irow++)
		{	printf("%.3lf\n", Input_B[irow]);
		}
		printf ("\n");
		printf("Solution vector \n");
		printf("Number of iterations = %d\n",Iteration);
		printf ("\n");
		for(irow = 0; irow < n_size; irow++)
			printf("%.12lf\n",X_New[irow]);
	}
	MPI_Finalize(); 
}

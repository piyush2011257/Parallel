

/*
************************************************************************
                        Example 6.3 (conjugate-gradient-mpi-code-clang.c)

       Objective            : Conjugate Gradient Method to solve AX = b
                              matrix system of linear equations.

       Input                : Real Symmetric Positive definite Matrix
                              and the real vector - Input_A and Vector_B
                              Read files (mdatcg.inp) for Matrix A
                              and (vdatcg.inp) for Vector b

       Description          : Input matrix is stored in n by n format.
                              Diagonal preconditioning matrix is used.
                              Rowwise block striped partitioning matrix
                              is used.Maximum iterations is given by 
                              MAX_ITERATIONS.Tolerance value is given 
                              by EPSILON
                              Header file used  : cg_constants.h

       Output               : The solution of  Ax=b on process with 
                              Rank 0 and the number of iterations 
                              for convergence of the method.

       Necessary conditions : Number of Processes should be less than
                              or equal to 8 Input Matrix Should be 
                              Square Matrix. Matrix size for Matrix A
                              and vector size for vector b should be 
                              equally striped, that is Matrix size and
                              Vector Size should be dividible by the 
                              number of processes used.

***************************************************************
*/



#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>

#define EPSILON 		  1.0E-20
#define MAX_ITERATIONS 10000

/******************************************************************************/
void 
GetPreconditionMatrix(double **Bloc_Precond_Matrix, int NoofRows_Bloc, 
		      int NoofCols)
{

	/*... Preconditional Martix is identity matrix .......*/
	int 	Bloc_MatrixSize;
	int 	irow, icol, index;
	double *Precond_Matrix;

	Bloc_MatrixSize = NoofRows_Bloc*NoofCols;

	Precond_Matrix = (double *) malloc(Bloc_MatrixSize * sizeof(double));

   index = 0;
	for(irow=0; irow<NoofRows_Bloc; irow++){
		for(icol=0; icol<NoofCols; icol++){
			Precond_Matrix[index++] = 1.0;
		}
	}
	*Bloc_Precond_Matrix = Precond_Matrix;
}
/******************************************************************************/
double 
ComputeVectorDotProduct(double *Vector1, double *Vector2, int VectorSize)
{
	int 	index;
	double Product;

	Product = 0.0;
	for(index=0; index<VectorSize; index++)
		Product += Vector1[index]*Vector2[index];

	return(Product);
}
/******************************************************************************/
void 
CalculateResidueVector(double *Bloc_Residue_Vector, double *Bloc_Matrix_A, 
							  double *Input_B, double *Vector_X, int NoofRows_Bloc, 
							  int VectorSize, int MyRank)
{
	/*... Computes residue = AX - b .......*/
	int   irow, index, GlobalVectorIndex;
	double value;

	GlobalVectorIndex = MyRank * NoofRows_Bloc;
	for(irow=0; irow<NoofRows_Bloc; irow++){
		index = irow * VectorSize;
		value = ComputeVectorDotProduct(&Bloc_Matrix_A[index], Vector_X, 
												  VectorSize);
		Bloc_Residue_Vector[irow] = value - Input_B[GlobalVectorIndex++];
	}
}
/******************************************************************************/
void
SolvePrecondMatrix(double *Bloc_Precond_Matrix, double *HVector, 
						 double *Bloc_Residue_Vector, int Bloc_VectorSize)
{
	/*...HVector = Bloc_Precond_Matrix inverse * Bloc_Residue_Vector.......*/
	int	index;

	for(index=0; index<Bloc_VectorSize; index++){
		HVector[index] = Bloc_Residue_Vector[index]/1.0;
	}
}
/******************************************************************************/
main(int argc, char *argv[])
{
	int        NumProcs, MyRank, Root=0;
	int        NoofRows, NoofCols, VectorSize;
	int        n_size, NoofRows_Bloc, Bloc_MatrixSize, Bloc_VectorSize;
	int	     Iteration = 0, irow, icol, index, CorrectResult;
	double	   **Matrix_A, *Input_A, *Input_B, *Vector_X, *Bloc_Vector_X; 
	double      *Bloc_Matrix_A, *Bloc_Precond_Matrix, *Buffer;
	double 	  *Bloc_Residue_Vector ,*Bloc_HVector, *Bloc_Gradient_Vector;
	double		  *Direction_Vector, *Bloc_Direction_Vector;
	double      Delta0, Delta1, Bloc_Delta0, Bloc_Delta1;
	double		  Tau, val, temp, Beta;

	double      *AMatDirect_local, *XVector_local;
	double      *ResidueVector_local, *DirectionVector_local;
	double     StartTime, EndTime;
	MPI_Status status;
	FILE  	  *fp;

  	/*...Initialising MPI .......*/
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&NumProcs);
	MPI_Comm_rank(MPI_COMM_WORLD,&MyRank);


  /* .......Read the Input file ......*/
  if(MyRank == Root) {

     if ((fp = fopen ("./matrix-data-cg.inp", "r")) == NULL) {
       printf("Can't open input matrix file");
       exit(-1);
     }
     fscanf(fp, "%d %d", &NoofRows,&NoofCols);     
     n_size=NoofRows;

     /* ...Allocate memory and read data .....*/
	  Matrix_A = (double **) malloc(n_size*sizeof(double *));

     for(irow = 0; irow < n_size; irow++){
		  Matrix_A[irow] = (double *) malloc(n_size * sizeof(double));
        for(icol = 0; icol < n_size; icol++)
	        fscanf(fp, "%lf", &Matrix_A[irow][icol]);
	  }
     fclose(fp);

     if ((fp = fopen ("./vector-data-cg.inp", "r")) == NULL){
        printf("Can't open input vector file");
        exit(-1);
     }

     fscanf(fp, "%d", &VectorSize);     
     n_size=VectorSize;
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

	MPI_Barrier(MPI_COMM_WORLD);
	StartTime = MPI_Wtime();

	/*...Broadcast Matrix and Vector size and perform input validation tests...*/
	MPI_Bcast(&NoofRows, 1, MPI_INT, Root, MPI_COMM_WORLD);
	MPI_Bcast(&NoofCols, 1, MPI_INT, Root, MPI_COMM_WORLD);
	MPI_Bcast(&VectorSize, 1, MPI_INT, Root, MPI_COMM_WORLD);


   if(NoofRows != NoofCols){
		MPI_Finalize();
		if(MyRank == Root)
			printf("Error : Coefficient Matrix Should be square matrix");
		exit(-1);
	}

   if(NoofRows != VectorSize){
		MPI_Finalize();
		if(MyRank == Root)
			printf("Error : Matrix Size should be equal to VectorSize");
		exit(-1);
	}

	if(NoofRows % NumProcs != 0){
		MPI_Finalize();
		if(MyRank == Root)
			printf("Error : Matrix cannot be evenly striped among processes");
		exit(-1);
	}

   /*...Allocate memory for Input_B and BroadCast Input_B.......*/
   if(MyRank != Root)
		Input_B = (double *) malloc(VectorSize*sizeof(double));
	MPI_Bcast(Input_B, VectorSize, MPI_DOUBLE, Root, MPI_COMM_WORLD);

   /*...Allocate memory for Block Matrix A and Scatter Input_A .......*/
   NoofRows_Bloc   = NoofRows / NumProcs;
	Bloc_VectorSize = NoofRows_Bloc;
	Bloc_MatrixSize = NoofRows_Bloc * NoofCols;
	Bloc_Matrix_A = (double *) malloc (Bloc_MatrixSize*sizeof(double));
	MPI_Scatter(Input_A, Bloc_MatrixSize, MPI_DOUBLE,
				   Bloc_Matrix_A, Bloc_MatrixSize, MPI_DOUBLE, Root, MPI_COMM_WORLD);


	/*... Allocates memory for solution vector and intialise it to zero.......*/
	Vector_X = (double *) malloc(VectorSize*sizeof(double));
	for(index=0; index<VectorSize; index++)
		Vector_X[index] = 0.0;

	/*...Calculate RESIDUE = AX - b .......*/
   Bloc_Residue_Vector = (double *) malloc(NoofRows_Bloc*sizeof(double));
	CalculateResidueVector(Bloc_Residue_Vector,Bloc_Matrix_A, Input_B, Vector_X,
								  NoofRows_Bloc, VectorSize, MyRank);

	/*... Precondtion Matrix is identity matrix ......*/
	GetPreconditionMatrix( &Bloc_Precond_Matrix, NoofRows_Bloc, NoofCols);

	/*...Bloc_HVector = Bloc_Precond_Matrix inverse * Bloc_Residue_Vector......*/
	Bloc_HVector = (double *) malloc(Bloc_VectorSize*sizeof(double));
	SolvePrecondMatrix(Bloc_Precond_Matrix, Bloc_HVector, Bloc_Residue_Vector, 
							 Bloc_VectorSize); 


   /*...Initailise Bloc Direction Vector = -(Bloc_HVector).......*/	
	Bloc_Direction_Vector = (double *) malloc(Bloc_VectorSize*sizeof(double));
	for(index=0; index<Bloc_VectorSize; index++)
		Bloc_Direction_Vector[index] = 0 - Bloc_HVector[index];

	/*...Calculate Delta0 and check for convergence .......*/
	Bloc_Delta0 = ComputeVectorDotProduct(Bloc_Residue_Vector,
										Bloc_HVector, Bloc_VectorSize);
	MPI_Allreduce(&Bloc_Delta0, &Delta0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	if(Delta0 < EPSILON){
		MPI_Finalize();
		exit(0);
	}

	/*...Allocate memory for Direction Vector.......*/
	Direction_Vector = (double *) malloc(VectorSize*sizeof(double));

	/*...Allocate temporary buffer to store Bloc_Matrix_A*Direction_Vector...*/
	Buffer = (double *) malloc(Bloc_VectorSize*sizeof(double));

	/*...Allocate memory for Bloc_Vector_X .......*/
	Bloc_Vector_X = (double *)malloc(Bloc_VectorSize*sizeof(double));

	Iteration = 0;
	do{

		Iteration++;
		/*
		if(MyRank == Root)
			printf("Iteration : %d\n",Iteration);
		*/

		/*...Gather Direction Vector on all processes.......*/
		MPI_Allgather(Bloc_Direction_Vector, Bloc_VectorSize, MPI_DOUBLE, 
		            Direction_Vector, Bloc_VectorSize, MPI_DOUBLE, MPI_COMM_WORLD);

		/*...Compute Tau = Delta0 / (DirVector Transpose*Matrix_A*DirVector)...*/
		for(index=0; index<NoofRows_Bloc; index++){
			Buffer[index] = ComputeVectorDotProduct(&Bloc_Matrix_A[index*NoofCols],
													  Direction_Vector, VectorSize);
		}
		temp = ComputeVectorDotProduct(Bloc_Direction_Vector, Buffer, 
												 Bloc_VectorSize);

		MPI_Allreduce(&temp, &val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		Tau = Delta0 / val;

		/*Compute new vector Xnew = Xold + Tau*Direction........................*/
		/*Compute BlocResidueVec  = BlocResidueVect + Tau*Bloc_MatA*DirVector...*/
		for(index = 0; index<Bloc_VectorSize; index++){
			Bloc_Vector_X[index]       = Vector_X[MyRank*Bloc_VectorSize + index] +
										  		  Tau*Bloc_Direction_Vector[index];
			Bloc_Residue_Vector[index] = Bloc_Residue_Vector[index] + 
												  Tau*Buffer[index];
		}
		/*...Gather New Vector X at all  processes......*/
		MPI_Allgather(Bloc_Vector_X, Bloc_VectorSize, MPI_DOUBLE, Vector_X, 
						  Bloc_VectorSize, MPI_DOUBLE, MPI_COMM_WORLD);

		SolvePrecondMatrix(Bloc_Precond_Matrix, Bloc_HVector, Bloc_Residue_Vector,
								 Bloc_VectorSize); 
		Bloc_Delta1 = ComputeVectorDotProduct(Bloc_Residue_Vector, Bloc_HVector, 
														  Bloc_VectorSize);
		
		MPI_Allreduce(&Bloc_Delta1, &Delta1, 1, MPI_DOUBLE, MPI_SUM, 
						  MPI_COMM_WORLD);

		if(Delta1 < EPSILON)
			break;

		Beta   = Delta1 / Delta0;
		Delta0 = Delta1;
		for(index=0; index<Bloc_VectorSize; index++){
			Bloc_Direction_Vector[index] = -Bloc_HVector[index] + 
													 Beta*Bloc_Direction_Vector[index];
		}
	}while(Delta0 > EPSILON && Iteration < MAX_ITERATIONS);

	MPI_Barrier(MPI_COMM_WORLD);
	EndTime = MPI_Wtime();

  if (MyRank == 0) {

     printf ("\n");
     printf(" ------------------------------------------- \n");
     printf("Results of Jacobi Method on processor %d: \n", MyRank);
     printf ("\n");

     printf("Matrix Input_A \n");
     printf ("\n");
     for (irow = 0; irow < n_size; irow++) {
        for (icol = 0; icol < n_size; icol++)
	        printf("%.3lf  ", Matrix_A[irow][icol]);
        printf("\n");
     }
     printf ("\n");
     printf("Matrix Input_B \n");
     printf("\n");
     for (irow = 0; irow < n_size; irow++) {
         printf("%.3lf\n", Input_B[irow]);
     }
     printf ("\n");
     printf("Solution vector \n");
	  printf("Number of iterations = %d\n",Iteration);
     printf ("\n");
     for(irow = 0; irow < n_size; irow++)
        printf("%.12lf\n",Vector_X[irow]);
     printf(" --------------------------------------------------- \n");
  }
  


	MPI_Finalize();
}





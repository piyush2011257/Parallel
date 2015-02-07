#include <stdio.h>
#include<stdlib.h>
#include "mpi.h"
#define MAX_ORDER 100
/* 
Exercise 5.12.1, see Chap 5, p. 87 & ff in PPMPI.
by John Weathewax
Here we input/create a matrix distributed by block rows.
*/ 

typedef float LOCAL_MATRIX_T[MAX_ORDER][MAX_ORDER];

void Gen_matrix( LOCAL_MATRIX_T local_A, int local_m, int n )
{	int i, j;
	for (i = 0; i < local_m; i++)
		for (j = 0; j < n; j++)
			local_A[i][j] = 1.0;
}

void Gen_vector( float  local_x[], int local_m )
{	int i;
	for (i = 0; i < local_m; i++)
		local_x[i] = 1.0;
}

void Print_matrix( char *title, LOCAL_MATRIX_T local_A, int local_m, int m, int n, int my_rank )
{	int i, j;
	float temp[MAX_ORDER][MAX_ORDER];
	/* Gather all local_A's into the array temp at the root: */
	MPI_Gather(local_A, local_m*MAX_ORDER, MPI_FLOAT, temp, local_m*MAX_ORDER, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if (my_rank == 0)
	{	printf("%s\n", title);
		for (i = 0; i < m; i++)
		{	for (j = 0; j < n; j++)
				printf("%4.1f ", temp[i][j]);
			printf("\n");
		}
	}
}

void Print_vector( char *title, float local_x[], int local_m, int m, int my_rank )
{	int i;
	float temp[MAX_ORDER];
	/* Gather all local_x's into the array temp at the root: */
	MPI_Gather(local_x, local_m, MPI_FLOAT, temp, local_m, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if (my_rank == 0)
	{	printf("%s\n", title);
		for (i = 0; i < m; i++)
			printf("%4.1f ", temp[i]);
		printf("\n");
	} 
}

int main( int argc, char* argv[] )
{	int my_rank, p;
	LOCAL_MATRIX_T local_A;
	float local_x[MAX_ORDER];
	int m, n, local_m;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	if( my_rank==0 )
	{	printf("Enter the size of the matrix (m x n):\n");
		scanf("%d %d",&m,&n);
		if( m > MAX_ORDER || n > MAX_ORDER )
		{	printf("m or n is too large...exiting\n");
			exit(1);
		}
	}
	MPI_Bcast(&m,1,MPI_INT,0,MPI_COMM_WORLD); 
	MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD); 
	local_m = m/p;
	/* Fill the matrix/vector in each process: */ 
	Gen_matrix(local_A,local_m,n);
	Gen_vector(local_x,local_m); 
	/* Print the matrix/vector from each process: */
	Print_matrix("The matrix A is: ",local_A,local_m,m,n,my_rank);
	Print_vector("The vector x is: ",local_x,local_m,m,  my_rank);
	MPI_Finalize();
}

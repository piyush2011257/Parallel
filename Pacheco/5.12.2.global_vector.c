#include <stdio.h>
#include<stdlib.h>
#include "mpi.h"
#define MAX_ORDER 100
/* 
Exercise 5.12.1, see Chap 5, p. 87 & ff in PPMPI.
by John Weathewax
Here we input/create a matrix distributed by block rows.
*/ 

// scatter, gather, all_gather, reduce, all_reduce
void global_vector_sum ( float *val, int n, int my_rank, int p )
{	float global_val[p][n], tmp_val[n];
	// extra buffer tmp_val needed else it will lead to aliasing of val variable, which is an in/out variable, which is not allowed !!
	MPI_Reduce(val, tmp_val, n, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Gather(val, n, MPI_FLOAT, global_val, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if ( my_rank == 0 )
	{	int i,j;
		for ( i=0; i<p; i++)
		{	printf("process: %d\n", i );
			for ( j=0; j<n; j++)
				printf("%f  ", global_val[i][j] );
			printf("\n");
		}
		for ( j=0; j<n; j++)
			printf("%f  ", tmp_val[j] );
			//*val= tmp_val;
		//printf("%f\n", *val);
	}
}

int main( int argc, char* argv[] )
{	int my_rank, p;
	float val[10];
	int n=10,i;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	//MPI_Bcast(&m,1,MPI_INT,0,MPI_COMM_WORLD); 
	for ( i=0; i<10; i++ )
		val[i]= rand()%100;
	//printf("my_rank: %d val: %f\n", my_rank, val);
	/* Fill the matrix/vector in each process: */ 
	global_vector_sum(val, n, my_rank, p);
	MPI_Finalize();
}

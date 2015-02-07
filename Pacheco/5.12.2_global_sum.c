#include <stdio.h>
#include<stdlib.h>
#include "mpi.h"
#define MAX_ORDER 100
/* 
Exercise 5.12.1, see Chap 5, p. 87 & ff in PPMPI.
by John Weathewax
Here we input/create a matrix distributed by block rows.
*/ 

void global_sum ( float *val, int my_rank, int p )
{	float global_val[MAX_ORDER], tmp_val;
	// extra buffer tmp_val needed else it will lead to aliasing of val variable, which is an in/out variable, which is not allowed !!
	MPI_Reduce(val, &tmp_val, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Gather(val, 1, MPI_FLOAT, global_val, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if ( my_rank == 0 )
	{	int i;
		for ( i=0; i<p; i++)
			printf("process: %d val: %f\n", i, global_val[i] );
		printf("\n");
		*val= tmp_val;
		printf("%f\n", *val);
	}
}

int main( int argc, char* argv[] )
{	int my_rank, p;
	float val;
	int m, n, local_m;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	//MPI_Bcast(&m,1,MPI_INT,0,MPI_COMM_WORLD); 
	val= rand()%100;
	printf("my_rank: %d val: %f\n", my_rank, val);
	/* Fill the matrix/vector in each process: */ 
	global_sum(&val, my_rank, p);
	MPI_Finalize();
}

#include <mpi.h>
#include <stdio.h>

void times_table(int n)
{	int i, i_times_n, rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	for (i=1; i<=n; ++i)
	{	i_times_n = i * n;
		printf("Process %d says %d times %d equals %d.\n", rank, i, n, i_times_n );
		//sleep(1);
	}
}

int main(int argc, char **argv)
// all the processes run simultaneously. Obsereve the output with -np 3
{	int rank;
	MPI_Init(&argc, &argv);		// initialize MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);		// get the rank of the current process
	// using SPMD (Single Program Multiple Data )
	if (rank == 0)
	{	printf("This is the main process.\n");
		times_table(12);
	}
	else if (rank == 1)
	{	printf("This is the process %d.\n", rank);
		times_table(11);
	}
	else if (rank == 2)
	{	printf("This is the process %d.\n", rank);
		times_table(10);
	}
	else
	{	printf("This is the process %d.\n", rank);
		printf("I am not needed...\n");
		//sleep(1);
	}
	MPI_Finalize();
	return 0;
}

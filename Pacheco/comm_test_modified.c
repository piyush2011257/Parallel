// mpirun -np 16 ./a.out
#include <stdio.h>
#include "mpi.h"
#include <math.h>
#include <stdlib.h>

// Understand how new communicator group and old communicator groups are related!
int main(int argc, char* argv[])
{	int p, my_rank, n_bar = 2, proc, i, my_rank_in_first_row, q;
	// q= sqrt(p), n_bar= n/q. each grid is of size n/q * n/q and q grids in each row and column. let n=4, p=4. n_bar has been fixed at 2
	MPI_Group group_world, first_row_group;
	MPI_Comm first_row_comm;
	int *process_ranks;
	float *A_00;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	q = (int) sqrt((double) p);
	process_ranks = (int*) malloc(q*sizeof(int));
	int t=0;
	// Here we store q processes that will form a new communicator gruop. The reanks of these processes in MPI_COMM_WORLD is stored in process_rank[] array. Here we take all even umbered processes and those < q. No redundant elements in processor_rank and all valid process numbers / ranks only!!	
	for (proc = 0; proc < q; t+=2, proc++)
		process_ranks[proc] = t;	
	/* Get the group underlying MPI_COMM_WORLD */
	MPI_Comm_group(MPI_COMM_WORLD, &group_world);
	/* Create the new group containing process of rank in the process_rank[] array */
	MPI_Group_incl(group_world, q, process_ranks, &first_row_group);
	/* Create the new communicator. This command in collective communication type and hence it has to be called by every process in MPI_COMM_WORLD */
	MPI_Comm_create(MPI_COMM_WORLD, first_row_group, &first_row_comm);
	/* Now broadcast across the first row */
	if (my_rank % 2 == 0 && my_rank < q*2) 			// broadcast among these processes only. my_rank in MPI_COMM_WORLD !!
	{	// error if called for any process not i first_row_comm
		MPI_Comm_rank(first_row_comm, &my_rank_in_first_row);
		// see the difference in rank of the same process in MPI_COMM_WORLD and first_row_comm. The ordering changes!
		printf("my_rank: %d my_rank_in_first_row: %d\n", my_rank, my_rank_in_first_row);
		A_00 = (float*) malloc (n_bar*n_bar*sizeof(float));
		if (my_rank_in_first_row == 0)
		{	for (i = 0; i < n_bar*n_bar; i++)
				A_00[i] = (float) i;
		}
		MPI_Bcast(A_00, n_bar*n_bar, MPI_FLOAT, 0, first_row_comm);
		printf("Process %d > ", my_rank);
		for (i = 0; i < n_bar*n_bar; i++)
			printf("%4.1f ", A_00[i]);
		printf("\n");
	}
	MPI_Finalize();
}

/*
o/p:
piyush@piyush-TravelMate-4740:~/Parallel Computing/Programs$ mpirun -np 16 ./a.out 
my_rank: 0 my_rank_in_first_row: 0
my_rank: 2 my_rank_in_first_row: 1
Process 0 > my_rank: 4 my_rank_in_first_row: 2
Process 4 >  0.0  1.0  2.0  3.0 
 0.0 Process 2 >  0.0  1.0  2.0  3.0 
 1.0  2.0 my_rank: 6 my_rank_in_first_row: 3
 3.0 
Process 6 >  0.0  1.0  2.0  3.0 

this is correct. printing also done parallely here!

another better o/p form:
piyush@piyush-TravelMate-4740:~/Parallel Computing/Programs$ mpirun -np 16 ./a.out 
my_rank: 0 my_rank_in_first_row: 0
Process 0 >  0.0  1.0  2.0  3.0 
my_rank: 2 my_rank_in_first_row: 1
Process 2 >  0.0  1.0  2.0  3.0 
my_rank: 4 my_rank_in_first_row: 2
Process 4 >  0.0  1.0  2.0  3.0 
my_rank: 6 my_rank_in_first_row: 3
Process 6 >  0.0  1.0  2.0  3.0 
*/

/* comm_test.c -- creates a communicator from the first q processes
 *     in a communicator containing p = q^2 processes.  Broadcasts
 *     an array to the members of the newly created communicator.
 *
 * Input: none
 * Output: Contents of array broadcast to each process in the newly
 *     created communicator
 *
 * Note:  MPI_COMM_WORLD should contain p = q^2 processes.
 *
 * See Chap 7., pp. 117 & ff. in PPMPI
 */
#include <stdio.h>
#include "mpi.h"
#include <math.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{	int p, my_rank, n_bar = 2, proc, i, my_rank_in_first_row, q;
	// q= sqrt(p), n_bar= n/q	let n=4, p=4
	MPI_Group group_world, first_row_group;
	MPI_Comm first_row_comm;
	int *process_ranks;
	float *A_00;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	q = (int) sqrt((double) p);
	/* Make a list of the processes in the new communicator */
	process_ranks = (int*) malloc(q*sizeof(int));
	for (proc = 0; proc < q; proc++)
		process_ranks[proc] = proc;		// this creates an array of rank of processors to be grouped ( the rank is of the MPI_COMM_WORLD )
	/* Get the group underlying MPI_COMM_WORLD */
	MPI_Comm_group(MPI_COMM_WORLD, &group_world);
	/* Create the new group containing process of rank in the process_rank[] array */
	MPI_Group_incl(group_world, q, process_ranks, &first_row_group);
	/* Create the new communicator. This command in collective communication type and hence it has to be called by every process in MPI_COMM_WORLD */
	MPI_Comm_create(MPI_COMM_WORLD, first_row_group, &first_row_comm);
	/* Now broadcast across the first row */
	if (my_rank < q) 				// broadcast among these processes only !!
	{	MPI_Comm_rank(first_row_comm, &my_rank_in_first_row);
		/* Allocate space for A_00 */
		printf("my_rank: %d my_rank_in_first_row: %d\n", my_rank, my_rank_in_first_row);
		A_00 = (float*) malloc (n_bar*n_bar*sizeof(float));
		if (my_rank_in_first_row == 0)
		{	/* Initialize A_00 */
			for (i = 0; i < n_bar*n_bar; i++)
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

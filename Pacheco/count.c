/* count.c -- send a subvector from process 0 to process 1
 *
 * Input: none
 * Output: contents of vector received by process 1
 *
 * Note: Program should only be run with 2 processes.
 *
 * See Chap 6, pp. 89 & ff. in PPMPI
 */
#include <stdio.h>
#include "mpi.h"

int main(int argc, char* argv[])
{	float vector[100];
	MPI_Status status;
	int p, my_rank, i;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	/*  Initialize vector and send */
	if (my_rank == 0)
	{	for (i = 0; i < 50; i++)
			vector[i] = 0.0;
		for (i = 50; i < 100; i++)
			vector[i] = 1.0;
		MPI_Send(vector+50, 50, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
	}
	else
	{	/* my_rank == 1 */
		MPI_Recv(vector+50, 50, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
		for (i = 50; i < 100; i++)
			printf("%3.1f ",vector[i]);
		printf("\n");
	}
	MPI_Finalize();
}

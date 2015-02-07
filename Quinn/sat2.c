/*
 *   Circuit Satisfiability, Version 2
 *
 *   This enhanced version of the program prints the
 *   total number of solutions.
 */

#include "mpi.h"
#include <stdio.h>

/* Return 1 if 'i'th bit of 'n' is 1; 0 otherwise */
#define EXTRACT_BIT(n,i) ((n&(1<<i))?1:0)

int check_circuit (int id, int z)
{	int v[16];        /* Each element is a bit of z */
	int i;
	for (i = 0; i < 16; i++)
		v[i] = EXTRACT_BIT(z,i);
	if ((v[0] || v[1]) && (!v[1] || !v[3]) && (v[2] || v[3]) && (!v[3] || !v[4]) && (v[4] || !v[5]) && (v[5] || !v[6]) && (v[5] || v[6]) && (v[6] || !v[15]) && (v[7] || !v[8]) && (!v[7] || !v[13]) && (v[8] || v[9]) && (v[8] || !v[9]) && (!v[9] || !v[10]) && (v[9] || v[11]) && (v[10] || v[11]) && (v[12] || v[13]) && (v[13] || !v[14]) && (v[14] || v[15]))
	{	printf ("%d) %d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d\n", id,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15]);
		fflush (stdout);
		return 1;
	}
	else
		return 0;
}

int main (int argc, char *argv[])
{	int count, global_count, i, my_rank, p;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	count = 0;
	for (i = my_rank; i < 65536; i += p)
		count += check_circuit (my_rank, i);
	MPI_Reduce (&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	printf ("Process %d is done\n", my_rank);
	fflush (stdout);
	MPI_Finalize();
	if ( my_rank == 0 )
		printf ("There are %d different solutions\n", global_count);
	return 0;
}

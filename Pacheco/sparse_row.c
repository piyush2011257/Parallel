/* sparse_row.c -- pack a row of a sparse matrix and send from process 0
 *     to process 1.  Process 1 allocates required storage after partially
 *     unpacking.
 *
 * Input: none
 * Output: the row received by process 1.
 *
 * Notes:  
 *     1. This program should only be run with 2 processes.  
 *     2. Only the row of the matrix is created on both processes.
 *
 * See Chap. 6, pp. 104 & ff in PPMPI
 */
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#define HUGE 100

int main(int argc, char* argv[])
{	int p, my_rank, nonzeroes, position, row_number, i;
	float *entries;
	int *column_subscripts;
	char buffer[HUGE]; 	/* HUGE is a predefined constant */
	MPI_Status  status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	if (my_rank == 0)
	{	/* Get the number of nonzeros in the row. Allocate storage for the row. Initialize entries and column_subscripts */
		nonzeroes = 10;
		entries = (float*) malloc(nonzeroes*sizeof(float));
		column_subscripts = (int*) malloc(nonzeroes*sizeof(int));
		for (i = 0; i < nonzeroes; i++)
		{	entries[i] = (float) 2*i;		// instead of sending full row send only non-zero entries of a sparse row
			column_subscripts[i] = 3*i;		// for 1 fixed row
		}
		/* Now pack the data and send */
		position = 0;
		MPI_Pack(&nonzeroes, 1, MPI_INT, buffer, HUGE, &position, MPI_COMM_WORLD);
		MPI_Pack(&row_number, 1, MPI_INT, buffer, HUGE, &position, MPI_COMM_WORLD);
		MPI_Pack(entries, nonzeroes, MPI_FLOAT, buffer, HUGE, &position, MPI_COMM_WORLD);
		MPI_Pack(column_subscripts, nonzeroes, MPI_INT, buffer, HUGE, &position, MPI_COMM_WORLD);
		// note the parameters in MPI_Send()
		MPI_Send(buffer, position, MPI_PACKED, 1, 0, MPI_COMM_WORLD);		// data type is MPI_Packed()
	}
	else
	{	/* my_rank == 1 */
		MPI_Recv(buffer, HUGE, MPI_PACKED, 0, 0, MPI_COMM_WORLD, &status);
		position = 0;
		MPI_Unpack(buffer, HUGE, &position, &nonzeroes, 1, MPI_INT, MPI_COMM_WORLD);
		MPI_Unpack(buffer, HUGE, &position, &row_number, 1, MPI_INT, MPI_COMM_WORLD);
		/* Allocate storage for entries and column_subscripts */
		entries = (float *) malloc(nonzeroes*sizeof(float));
		column_subscripts = (int *) malloc(nonzeroes*sizeof(int));
		MPI_Unpack(buffer,HUGE, &position, entries, nonzeroes, MPI_FLOAT, MPI_COMM_WORLD);
		MPI_Unpack(buffer, HUGE, &position, column_subscripts, nonzeroes, MPI_INT, MPI_COMM_WORLD);
		for (i = 0; i < nonzeroes; i++)
			printf("%4.1f %2d\n", entries[i], column_subscripts[i]);
	}
	MPI_Finalize();
}

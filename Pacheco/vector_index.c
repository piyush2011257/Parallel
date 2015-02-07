// use of vector index-> send column to row
// sending lower triangle to upper triangle
// sending 2 columns of a matrix to upper triangle  / linear array
#include <stdio.h>
#include "mpi.h"

int main(int argc, char* argv[])
{	int p, my_rank;
	float A[4][4];
	MPI_Status status;
	MPI_Datatype column_mpi_t;
	int i, j;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Type_vector(4, 2, 4, MPI_FLOAT, &column_mpi_t);
	MPI_Type_commit(&column_mpi_t);
	int displacements[4], block_lengths[4];
	MPI_Datatype  index_mpi_t;
	block_lengths[0] = block_lengths[1] = 3;
	block_lengths[2] = block_lengths[3] = 1;
	displacements[0] = 0;
	displacements[1] = 4;
	displacements[2] = 10;
	displacements[3] = 14;
	/*
	Hence we get a signature:
	(n*MPI_Float,0), ((n-1)*MPI_Float, (n+1)*1*sizeof(MPI_Float)), ...... (n+1)*(n-1)*MPI_Float, n*i*sizeof(MPI_Float))-> hence our signature. (n-i)*MPI_Floatv as no.of elements in each block = n-i for each copy
	*/
	MPI_Type_indexed(4, block_lengths, displacements, MPI_FLOAT, &index_mpi_t);
	MPI_Type_commit(&index_mpi_t);
	
	if (my_rank == 0)
	{	for (i = 0; i < 4; i++)
			for (j = 0; j < 4; j++)
				A[i][j] = (float) i;
		MPI_Send(&(A[0][0]), 1, column_mpi_t, 1, 0, MPI_COMM_WORLD);		// sends first 2 columns. as column_mpi_t means the elements at first 2 columns! Note that MPI_Vector() creates a data type !! on how elements stored and send and received! 
	}
	else
	{	/* my_rank = 1 */
		for (i = 0; i < 4; i++)
			for (j = 0; j < 4; j++)
				A[i][j] = -1.0;
		// makes the same signature irrespective of displacement!!
		// sending validity is checked on the basis of type signature
		// signature of send is (float)(float)(float)... 10 times and so is the receiving signature and hence no error!! displacement portion of signature not considered!
		MPI_Recv(&(A[0][1]), 1, index_mpi_t, 0, 0, MPI_COMM_WORLD, &status);
		for (i = 0; i < 4; i++)
		{	for (j = 0; j < 4; j++)
				printf("%3.1f\t", A[i][j]);
			printf("\n");
		}		
		// all elements in the first row of the array
		printf("\n");
	}
	MPI_Finalize();
}

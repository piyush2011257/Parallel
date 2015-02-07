// no use of poiter to a function. Menu choice based program
// storing a and b as a continuous elements of an array! Hence blocks = 2 and block_lengths[0]=2!
	
#include <stdio.h>
#include "mpi.h"

void Build_derived_type( float *ab_ptr, int *n_ptr, MPI_Datatype *mesg_mpi_t_ptr )
{	// storing a and b as a continuous elements of an array!
	int block_lengths[2];
	MPI_Aint displacements[2];
	MPI_Datatype typelist[2];  
	MPI_Aint start_address;
	MPI_Aint address;
	block_lengths[0] = 2;
	block_lengths[1] = ab_ptr*(ab_ptr+1);
	typelist[0] = MPI_FLOAT;
	typelist[1] = MPI_INT;
	displacements[0] = 0;
	MPI_Address(ab_ptr, &start_address);
	MPI_Address(n_ptr, &address);
	displacements[1] = address - start_address;
	MPI_Type_struct(2, block_lengths, displacements, typelist, mesg_mpi_t_ptr);
	MPI_Type_commit(mesg_mpi_t_ptr);
}

void Get_data( float *ab_ptr, int *n_ptr, int my_rank )
{	// storing a and b as a continuous elements of an array!
	MPI_Datatype mesg_mpi_t;	/* MPI type corresponding to 3 floats and an int */
	if (my_rank == 0)
	{	printf("Enter a, b\n");
		scanf("%f %f", ab_ptr, ab_ptr+1);
		int i,j;
		for ( i=0; i<*ab_ptr; i++)
		{	for ( j=0; j<*(ab_ptr+1); j++)
				n_ptr[i][j] = rand()%100;
		}
	}
	Build_derived_type(ab_ptr, n_ptr, &mesg_mpi_t);
	MPI_Bcast(ab_ptr, 1, mesg_mpi_t, 0, MPI_COMM_WORLD);
}

int main(int argc, char** argv)
{	int my_rank, p, n[100][100];
	float ab[2];				// ab[2]-> ab[0]=a, ab[1]=b
	MPI_Status  status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	Get_data(&ab[0], &n[0][0], my_rank);			// Pointers so that we have call by reference!!
	printf("my_rank: %d %f %f %d\n", my_rank, ab[0], ab[1]);
	for ( int i=0; i<ab[0]; i++)
	{	for ( int j=0; j<ab[1]; j++ )
			printf("%d  ", n[i][j]);
		printf("\n");
	}
	MPI_Finalize();
}

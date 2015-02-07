// no use of poiter to a function. Menu choice based program
// storing a and b as a continuous elements of an array! Hence blocks = 2 and block_lengths[0]=2!
	
#include <stdio.h>
#include "mpi.h"

float f(float x)
{	float return_val;
	return_val = x*x;
	return return_val;
}

float Trap( float local_a, float local_b, int local_n, float h )
{	float integral, x;
	int i;
	integral = (f(local_a) + f(local_b))/2.0;
	x = local_a;
	for (i = 1; i <= local_n-1; i++)
	{	x = x + h;
		integral = integral + f(x);
	}
	integral = integral*h;
	return integral;
}

void Build_derived_type( float *ab_ptr, int *n_ptr, MPI_Datatype *mesg_mpi_t_ptr )
{	// storing a and b as a continuous elements of an array!
	int block_lengths[2];
	MPI_Aint displacements[2];
	MPI_Datatype typelist[2];  
	MPI_Aint start_address;
	MPI_Aint address;
	block_lengths[0] = 2;
	block_lengths[1] = 1;
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
	{	printf("Enter a, b, and n\n");
		scanf("%f %f %d", ab_ptr, ab_ptr+1, n_ptr);
	}
	// called by all!!
	Build_derived_type(ab_ptr, n_ptr, &mesg_mpi_t);
	MPI_Bcast(ab_ptr, 1, mesg_mpi_t, 0, MPI_COMM_WORLD);
}

int main(int argc, char** argv)
{	int my_rank, p, source, dest = 0, tag = 0, n, local_n, ch;
	float ab[2], h, local_a, local_b, integral, total;		// ab[2]-> ab[0]=a, ab[1]=b
	MPI_Status  status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	Get_data(&ab[0], &n, my_rank);			// Pointers so that we have call by reference!!
	h = (ab[1]-ab[0])/n;
	local_n = n/p;					// no. of trapezoids handled by 1 process
	local_a = ab[0] + my_rank*local_n*h;
	local_b = local_a + local_n*h;
	integral = Trap(local_a, local_b, local_n, h);
	printf("my_rank: %d %f %f %d\n", my_rank, ab[0], ab[1], n);
	printf("integral:%f\n", integral);
	MPI_Reduce(&integral, &total, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	if (my_rank == 0)
		printf("With n = %d trapezoids, our estimate of the integral from %f to %f = %f\n",n, ab[0], ab[1], total);
	MPI_Finalize();
}

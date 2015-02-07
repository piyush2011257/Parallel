// no use of poiter to a function. Menu choice based program
#include <stdio.h>
#include "mpi.h"

float f2(float x)
{	float return_val;
	return_val = x*x;
	return return_val;
}

float f1(float x)
{	return x;
}

float f3(float x)
{	float return_val;
	return_val = x*x*x;
	return return_val;
}

float Trap( float  local_a, float  local_b, int local_n, float h, int ch )
{	float integral, x;
	int i;
	switch(ch)
	{	case 1:{	integral = (f1(local_a) + f1(local_b))/2.0;
				x = local_a;
				for (i = 1; i <= local_n-1; i++)
				{	x = x + h;
					integral = integral + f1(x);
				}
				integral = integral*h;
				printf("integral: %f\n", integral);
				break;
			}
		case 2:{	integral = (f2(local_a) + f2(local_b))/2.0;
				x = local_a;
				for (i = 1; i <= local_n-1; i++)
				{	x = x + h;
					integral = integral + f2(x);
				}
				integral = integral*h;
				printf("integral: %f\n", integral);
				break;
			}
		case 3:{	integral = (f3(local_a) + f3(local_b))/2.0;
				x = local_a;
				for (i = 1; i <= local_n-1; i++)
				{	x = x + h;
					integral = integral + f3(x);
				}
				integral = integral*h;
				printf("integral: %f\n", integral);
				break;
			}
	}
	return integral;
}

void Build_derived_type( float *a_ptr, float *b_ptr, int *n_ptr, int *ch, MPI_Datatype *mesg_mpi_t_ptr )
{	int block_lengths[4];
	MPI_Aint displacements[4];
	MPI_Datatype typelist[4];  
	MPI_Aint start_address;
	MPI_Aint address;
	block_lengths[0] = block_lengths[1] = block_lengths[2] = block_lengths[3] = 1;
	typelist[0] = typelist[1] = MPI_FLOAT;
	typelist[2] = typelist[3] = MPI_INT;
	displacements[0] = 0;
	MPI_Address(a_ptr, &start_address);
	MPI_Address(b_ptr, &address);
	displacements[1] = address - start_address;
	MPI_Address(n_ptr, &address);
	displacements[2] = address - start_address;
	MPI_Address(ch, &address);
	displacements[3] = address - start_address;
	MPI_Type_struct(4, block_lengths, displacements, typelist, mesg_mpi_t_ptr);
	MPI_Type_commit(mesg_mpi_t_ptr);
}

void Get_data( float *a_ptr, float *b_ptr, int *n_ptr, int *ch, int my_rank )
{	MPI_Datatype mesg_mpi_t;	/* MPI type corresponding to 3 floats and an int */
	if (my_rank == 0)
	{	printf("Enter a, b, and n\n");
		scanf("%f %f %d", a_ptr, b_ptr, n_ptr);
		printf("Enter the function to be used\n1- f(x)=x\n2- f(x) = x^2\n3- f(x) = x^3\n");
		scanf("%d",ch);
	}
	Build_derived_type(a_ptr, b_ptr, n_ptr, ch, &mesg_mpi_t);
	MPI_Bcast(a_ptr, 1, mesg_mpi_t, 0, MPI_COMM_WORLD);
}

int main(int argc, char** argv)
{	int my_rank, p, source, dest = 0, tag = 0, n, local_n, ch;
	float a, b, h, local_a, local_b, integral, total;
	MPI_Status  status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	Get_data(&a, &b, &n, &ch, my_rank);		// Pointers so that we have call by reference!!
	h = (b-a)/n;
	local_n = n/p;					// no. of trapezoids handled by 1 process
	local_a = a + my_rank*local_n*h;
	local_b = local_a + local_n*h;
	integral = Trap(local_a, local_b, local_n, h, ch);
	MPI_Reduce(&integral, &total, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	if (my_rank == 0)
		printf("With n = %d trapezoids, our estimate of the integral from %f to %f = %f\n",n, a, b, total);
	MPI_Finalize();
}

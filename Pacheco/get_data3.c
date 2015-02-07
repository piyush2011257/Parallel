/* get_data3.c -- Parallel Trapezoidal Rule.  Builds a derived type
 *     for use with the distribution of the input data.
 *
 * Input: 
 *    a, b: limits of integration.
 *    n: number of trapezoids.
 * Output:  Estimate of the integral from a to b of f(x) 
 *    using the trapezoidal rule and n trapezoids.
 *
 * Notes:  
 *    1.  f(x) is hardwired.
 *    2.  the number of processes (p) should evenly divide
 *        the number of trapezoids (n).
 *
 * See Chap 6, pp. 90 & ff in PPMPI
 */
#include <stdio.h>
#include "mpi.h"

void Build_derived_type( float *a_ptr, float *b_ptr, int *n_ptr, MPI_Datatype *mesg_mpi_t_ptr )
{	/* The number of elements in each "block" of the new type.  For us, 1 each. */
	int block_lengths[3];
	/* Displacement of each element from start of new type. MPI_Aint ("address int") is an MPI defined C type. Usually an int.*/
	MPI_Aint displacements[3];
	/* MPI types of the elements.  The "t_i's."        */
	MPI_Datatype typelist[3];  
	/* Use for calculating displacements               */
	MPI_Aint start_address;
	MPI_Aint address;
	// here 1 block = each individual elemnt of the derived data type. 3 blocks-> a, b, n and each is a single element enetity  ot an array!
	block_lengths[0] = block_lengths[1] = block_lengths[2] = 1;
	/* Build a derived datatype consisting of two floats and an int                   */
 	typelist[0] = MPI_FLOAT;
	typelist[1] = MPI_FLOAT;
	typelist[2] = MPI_INT;
	/* First element, a, is at displacement 0      */
	displacements[0] = 0;
	/* Calculate other displacements relative to a */
	MPI_Address(a_ptr, &start_address);
	/* Find address of b and displacement from a   */
	MPI_Address(b_ptr, &address);
	displacements[1] = address - start_address;
	/* Find address of n and displacement from a   */
	MPI_Address(n_ptr, &address);
	displacements[2] = address - start_address;
	/* Build the derived datatype */
	MPI_Type_struct(3, block_lengths, displacements, typelist, mesg_mpi_t_ptr);
	/* Commit it -- tell system we'll be using it for communication. Mandatory step to allow more efficient communication and allow system to make necessary interal change for message passing */
	MPI_Type_commit(mesg_mpi_t_ptr);
}

void Get_data3( float *a_ptr, float *b_ptr, int *n_ptr, int my_rank )
{	MPI_Datatype mesg_mpi_t;	/* MPI type corresponding to 3 floats and an int */
	if (my_rank == 0)
	{	printf("Enter a, b, and n\n");
		scanf("%f %f %d", a_ptr, b_ptr, n_ptr);
	}
	// for all processes Biuld_derived_type()
	Build_derived_type(a_ptr, b_ptr, n_ptr, &mesg_mpi_t);
	// note that broadcasting is an expensive operation. Hence we reduce the total calls to MPI_Bcast() from 3 to 1 by using a structure / derived type for the 3 variables and treating them as a single entity
	// note a_ptr is the variable that is sent and IT IS THE BASE ADDRESS for displacement and the data type is mesg_mpi_type
	// 1 single MPI_Bcast using derived data type is better than using multiple bcast of simple type! There is always a latency associated with transmission of messgae
	MPI_Bcast(a_ptr, 1, mesg_mpi_t, 0, MPI_COMM_WORLD);		// and value is bcasted to b_ptr and n_ptr as well
	// variable passed a_ptr ( as displacement = 0 for this (its our starting address ) and type is mesg_mpi_type. we use count = 1 so only 1 element of size location = sizeof(mesg_mpi_type)
	// using the below we send only the values of b_ptr and _pte to other process!! as the base address starts from b and not a so it skips a.
	// MPI_Bcast(b_ptr, 1, mesg_mpi_t, 0, MPI_COMM_WORLD);
	// MPI_Bcast(n_ptr, 1, mesg_mpi_t, 0, MPI_COMM_WORLD);
	
}

float f(float x)
{	float return_val;
	/* Calculate f(x). Store calculation in return_val. */
	return_val = x*x;
	return return_val;
}

float Trap( float local_a, float local_b, int local_n, float h )
{	float integral;   /* Store result in integral  */
	float x;
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

int main(int argc, char** argv)
{	int my_rank, p, source, dest = 0, tag = 0;
	float a = 0.0, b = 1.0;
	int n = 1024;
	float h;
	float local_a;  /* Left endpoint my process  */
	float local_b;  /* Right endpoint my process */
	int local_n;    /* Number of trapezoids for  */
			/* my calculation            */
	float integral;  /* Integral over my interval */
	float total;     /* Total integral            */
	MPI_Status  status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	// passes value of a, b and n in each process
	Get_data3(&a, &b, &n, my_rank);
	h = (b-a)/n;    /* h is the same for all processes */
	printf("a: %lf b: %lf n: %d\n", a,b,n);
	local_n = n/p;  /* So is the number of trapezoids */
	/* Length of each process' interval of integration = local_n*h. So my interval starts at: */
	local_a = a + my_rank*local_n*h;
	local_b = local_a + local_n*h;
	integral = Trap(local_a, local_b, local_n, h);
	printf("my_rank: %d %f %f %d\n", my_rank, a, b, n);
	printf("integral:%f\n", integral);
	/* Add up the integrals calculated by each process */
	MPI_Reduce(&integral, &total, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	if (my_rank == 0)
	{	printf("With n = %d trapezoids, our estimate\n", n);
		printf("of the integral from %f to %f = %f\n", a, b, total);
	}
	MPI_Finalize();
}

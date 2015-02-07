/* get_data.c -- Parallel Simpsons' Rule, uses basic Get_data function for 
 *     input.
 *
 * by John Weatherwax
 *
 * Input: 
 *    a, b: limits of integration.
 *    n: number of trapezoids.
 * Output:  Estimate of the integral from a to b of f(x) 
 *    using the trapezoidal rule and n trapezoids.
 *
 * Notes:  
 *    1.  f(x) is hardwired.
 *    2.  Assumes number of processes (p) evenly divides
 *        number of trapezoids (n).
 *
 * See Chap. 4, pp. 60 & ff in PPMPI.
 */
#include <stdio.h>

/* We'll be using MPI routines, definitions, etc. */
#include "mpi.h"

float f(float x)
{	float return_val; 
	return_val = (float)4.0 / ( (float)1.0 + x*x);
	return return_val; 
}

// breaking integration limits
float Simp( float  local_a, float  local_b, int local_n, float h )
{	float integral, x;
	int i;
	integral = f(local_a) + f(local_b);		// + f(local_b) ( see below how )
	x = local_a;
	for (i = 1; i <= local_n-1; i++)		// till n-1 only not n
	{	x = x + h;
		/* METHOD #1: */
		if( (i % 2) == 1 )
			integral = integral + 4*f(x); 		/* every ODD  term gets a multiplication by 4 */
		else
			integral = integral + 2*f(x);		/* every EVEN term gets a multiplication by 2 */
	}
	/* VS METHOD #2. ... this should be faster ... add a second f(x) if needed */ 
	/* integral = integral + 2*f(x); 
	   if( (i % 2) == 1 ) integral = integral + 2*f(x); */ 
    integral = (float)(integral*h)/3.0; 
    return integral;
}

int main(int argc, char** argv)
{	int my_rank, p, n, local_n, source, dest=0, tag=0;
	float a, b, h, local_a, local_b, integral, total;
	MPI_Status  status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	//Get_data(&a, &b, &n, my_rank, p);
	a=0, b=1, n=12;
	h = (b-a)/n;
	local_n = n/p;
	/* Length of each process' interval of integration = local_n*h.  So my interval starts at: */
	local_a = a + my_rank*local_n*h;
	local_b = local_a + local_n*h;
	integral = Simp(local_a, local_b, local_n, h);
	/* Add up the integrals calculated by each process */
	if (my_rank == 0)
	{	total = integral;
		for (source = 1; source < p; source++)
		{	MPI_Recv(&integral, 1, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);
			total = total + integral;
		}
	}
	else
		MPI_Send(&integral, 1, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
	if (my_rank == 0)
	{	printf("With n = %d trapezoids, our estimate\n", n);
		printf("of the integral from %f to %f = %f\n", a, b, total); 
	}
	MPI_Finalize();
}

/* get_data4.c -- Parallel Trapezoidal Rule.  Uses MPI_Pack/Unpack in
 *     distribution of input data.
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
 * See Chap 6., pp. 100 & ff in PPMPI
 */
#include <stdio.h>
#include "mpi.h"

void Get_data4( float *a_ptr, float *b_ptr, int *n_ptr, int my_rank )
{	char buffer[100];  /* Store data in buffer        */
	int position;     /* Keep track of where data is in the buffer           */
	if (my_rank == 0)
	{	printf("Enter a, b, and n\n");
		scanf("%f %f %d", a_ptr, b_ptr, n_ptr);
		/* Now pack the data into buffer.  Position = 0 says start at beginning of buffer.           */
		position = 0;
		/* Position is in/out */
		MPI_Pack(a_ptr, 1, MPI_FLOAT, buffer, 100, &position, MPI_COMM_WORLD);
		/* Position has been incremented: it now references the first free location in buffer.     */
		MPI_Pack(b_ptr, 1, MPI_FLOAT, buffer, 100, &position, MPI_COMM_WORLD);
		/* Position has been incremented again. */
		MPI_Pack(n_ptr, 1, MPI_INT, buffer, 100, &position, MPI_COMM_WORLD);
		/* Position has been incremented again. */
		/* Now broadcast contents of buffer */
		// Pack / Unpack only create a derived data type but do not send the data to other process
		// note the parameters in MPI_Bcast()
		MPI_Bcast(buffer, 100, MPI_PACKED, 0, MPI_COMM_WORLD);
	}
	else
	{	MPI_Bcast(buffer, 100, MPI_PACKED, 0, MPI_COMM_WORLD);
		/* Now unpack the contents of buffer */
		position = 0;
		// in the same order of packing ( according to value of position ) !!!
		MPI_Unpack(buffer, 100, &position, a_ptr, 1, MPI_FLOAT, MPI_COMM_WORLD);
		/* Once again position has been incremented: it now references the beginning of b.     */
		MPI_Unpack(buffer, 100, &position, b_ptr, 1, MPI_FLOAT, MPI_COMM_WORLD);
		MPI_Unpack(buffer, 100, &position, n_ptr, 1, MPI_INT, MPI_COMM_WORLD);
	}
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
	float a, b, h;
	int n;
	float local_a, local_b, integral, total; 
	int local_n;
	MPI_Status  status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	// passes value of a, b and n in each process
	Get_data4(&a, &b, &n, my_rank);
	h = (b-a)/n;    /* h is the same for all processes */
	local_n = n/p;  /* So is the number of trapezoids */
	/* Length of each process' interval of integration = local_n*h. So my interval starts at: */
	local_a = a + my_rank*local_n*h;
	local_b = local_a + local_n*h;
	integral = Trap(local_a, local_b, local_n, h);
	/* Add up the integrals calculated by each process */
	MPI_Reduce(&integral, &total, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	if (my_rank == 0)
	{	printf("With n = %d trapezoids, our estimate\n", n);
		printf("of the integral from %f to %f = %f\n", a, b, total);
	}
	MPI_Finalize();
}

/* trap.c -- Parallel Trapezoidal Rule, first version
 *
 * Input: None.
 * Output:  Estimate of the integral from a to b of f(x)
 *    using the trapezoidal rule and n trapezoids.
 *
 * Algorithm:
 *    1.  Each process calculates "its" interval of
 *        integration.
 *    2.  Each process estimates the integral of f(x)
 *        over its interval using the trapezoidal rule.
 *    3a. Each process != 0 sends its integral to 0.
 *    3b. Process 0 sums the calculations received from
 *        the individual processes and prints the result.
 *
 * Notes:  
 *    1.  f(x), a, b, and n are all hardwired.
 *    2.  The number of processes (p) should evenly divide
 *        the number of trapezoids (n = 1024)
 *
 * See Chap. 4, pp. 56 & ff. in PPMPI.
 */
#include <stdio.h>
#include "mpi.h"

float f(float x)
{	float return_val;
	/* Calculate f(x). Store calculation in return_val. */
	return_val = x*x;
	return return_val;
}

float Trap( float  local_a /* in */, float  local_b /* in */, int local_n /* in */, float h /* in */)
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
	h = (b-a)/n;    /* h is the same for all processes */
	local_n = n/p;  /* So is the number of trapezoids */
	/* Length of each process' interval of integration = local_n*h. So my interval starts at: */
	local_a = a + my_rank*local_n*h;
	local_b = local_a + local_n*h;
	integral = Trap(local_a, local_b, local_n, h);
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
		printf("With n = %d trapezoids, our estimate of the integral from %f to %f = %f\n",n, a, b, total);
	MPI_Finalize();
}

/* Explanation
Total n trapezoids formed so size of each trapezoid- (b-a)/n = h
Now we have p processors, assuming n is divisible by p, we can assign n/p trapezoids to each processor
n/p = local_n for each processor denoting the number of trapeiods for a processor.
Now by symmetry we observer that for a processor p:
local_a = a + my_rank*local_n*h, and,
local_b = local_a + local_n*h
hence we compute the integral sequentially on each process using the formula on Pg-55 and return it back.
ad hence do a global sum to get the final result
*/


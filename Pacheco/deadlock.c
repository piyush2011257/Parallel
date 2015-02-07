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

int main(int argc, char** argv)
{	int my_rank,p;
	MPI_Status  status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	int integral = my_rank, tag=0	;
	/* This would cause a deadlock
	if (my_rank == 0)
	{	MPI_Recv(&integral, 1, MPI_FLOAT, 1, tag, MPI_COMM_WORLD, &status);
		MPI_Send(&integral, 1, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
		printf("my_rank %d %d\n", my_rank, integral);
	}
	else
	{	MPI_Recv(&integral, 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
		MPI_Send(&integral, 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
		printf("my_rank %d %d\n", my_rank, integral);
	}
	/* This also would cause a deadlock if synchronous communiation i.e. A will not send until B's buffer is ready to receive and B will not receive until A's message buffer is ready -> deadlock
	if (my_rank == 0)
	{	MPI_Send(&integral, 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
		MPI_Send(&integral, 1, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
		printf("my_rank %d %d\n", my_rank, integral);
	}
	else
	{	MPI_Recv(&integral, 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
		MPI_Recv(&integral, 1, MPI_FLOAT, 1, tag, MPI_COMM_WORLD, &status);
		printf("my_rank %d %d\n", my_rank, integral);
	}
	*/
	if (my_rank == 0)
	{	MPI_Send(&integral, 1, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
		MPI_Recv(&integral, 1, MPI_FLOAT, 1, tag, MPI_COMM_WORLD, &status);
		printf("my_rank %d %d\n", my_rank, integral);
	}
	else
	{	MPI_Recv(&integral, 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
		MPI_Send(&integral, 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
/*
Inverting the routine also works fine. Remember send is asynchronous and receive is synchronous. In this case both send the that simulatneously and have corresponding receives to retrieve the data! Try it!
		MPI_Send(&integral, 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
		MPI_Recv(&integral, 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
*/
		/*
		Also if we use this extra receive then deadlock as no send routine
		MPI_Recv(&integral, 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
		*/
		printf("my_rank %d %d\n", my_rank, integral);
	}
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

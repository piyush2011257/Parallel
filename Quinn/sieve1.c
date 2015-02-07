#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include<stdlib.h>
#define MIN(a,b)  ((a)<(b)?(a):(b))
//char m[100000000];

int main (int argc, char *argv[])
{	int count, n, p, proc0_size, prime, size, first, global_count, high_value, i, my_rank, index, low_value;
	double elapsed_time;
	char *marked;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	MPI_Barrier(MPI_COMM_WORLD);
	elapsed_time = -MPI_Wtime();
	n = atoi(argv[1]);//12345678;
	/* Figure out this process's share of the array, as well as the integers represented by the first and last array elements */
	low_value = my_rank*(n-1)/p;
	high_value = (my_rank+1)*(n-1)/p -1;
	size = high_value - low_value + 1;
	/* Bail out if all the primes used for sieving are not all held by process 0 */
	proc0_size = (n-1)/p;
	if ( proc0_size < (int) sqrt((double) n))		// we want n/p <= sqrt(n)
	{	if (!my_rank)
			printf ("Too many processes\n");
		MPI_Finalize();
		exit (1);
	}
	/* Allocate this process's share of the array. */
	marked = /*m;*/(char *) malloc (size);				// 1 byte per element
	/*if (marked == NULL)
	{	printf ("Cannot allocate enough memory\n");
		MPI_Finalize();
		exit (1);
	}*/
	for (i = 0; i < size; i++)
		marked[i] = 0;			// initialize
	if (!my_rank)
	{	index = 2;
		marked[0]=marked[1]=1;
	}
	prime = 2;				// common to all as a starting point. No need of bcats() here
	do
	{	if (prime * prime > low_value)			// start from prime^2 -> n
			first = prime * prime - low_value;
		else						// locate first occurence of a multiple
		{	if (!(low_value % prime))
				first = 0;
			else
				first = prime - (low_value % prime);
		}
		for (i = first; i < size; i += prime)
			marked[i] = 1;
		if (!my_rank)
		{	while (marked[++index] == 1);
			prime = index;
		}
		if (p > 1)
			MPI_Bcast (&prime,  1, MPI_INT, 0, MPI_COMM_WORLD);
	} while (prime * prime <= n);
	count = 0;
	for (i = 0; i < size; i++)
		if (!marked[i])
			count++;
	printf("rank %d count %d\n", my_rank, count);
	if (p > 1)
		MPI_Reduce (&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	/* Stop the timer */
	elapsed_time += MPI_Wtime();
	/* Print the results */
	if (!my_rank)
	{	printf ("There are %d primes less than or equal to %d\n",global_count, n);
		printf ("SIEVE (%d) %10.6f\n", p, elapsed_time);
	}
	MPI_Finalize ();
	return 0;
}

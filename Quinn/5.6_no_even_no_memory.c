#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include<stdlib.h>
#define MIN(a,b)  ((a)<(b)?(a):(b))
//char m[100000000];

int main (int argc, char *argv[])
{	int count, n, n0, p, proc0_size, prime, size, first, global_count, high_value, i, my_rank, index, low_value;
	double elapsed_time;
	char *marked;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	MPI_Barrier(MPI_COMM_WORLD);
	elapsed_time = -MPI_Wtime();
	n0= atoi(argv[1]);//12345;
	n0+= n0 % 2 != 0;			// for odd no. and if the last no. itself is prime
	n=n0>>1;
	low_value = (long long int)((my_rank*n/p)<<1)+1;
	high_value = (long long int)(((my_rank+1)*n/p)<<1)-1;
	size = ((high_value - low_value)>>1) + 1;
	//printf("my_rank %d low_value %d high value %d size %d\n", my_rank, low_value, high_value, size);
	proc0_size = n/p;
	if ( proc0_size < (int) sqrt((double) n))		// we want n/p <= sqrt(n)
	{	if (!my_rank)
			printf ("Too many processes\n");
		MPI_Finalize();
		exit (1);
	}
	marked = /*m;*/(char *) malloc (size);				// 1 byte per element
	/*if (marked == NULL)
	{	printf ("Cannot allocate enough memory\n");
		MPI_Finalize();
		exit (1);
	}*/
	for (i = 0; i < size; i++)
		marked[i] = 0;			// initialize
	if (!my_rank)
	{	index = 1;
		marked[0]=1;				// 1 is not a prime
	}
	prime = 3;
	int prime_sq=prime * prime, lval_mod_prime= low_value % prime;
	do
	{	if (prime_sq > low_value)			// start from prime^2 -> n
			first = (prime_sq - low_value)>>1;
		else						// locate first occurence of a multiple
		{	if (!(lval_mod_prime))
				first = 0;
			else
			{	first = prime - lval_mod_prime;
				if ( (low_value+first) % 2 == 0 )
				{	first+=prime;
					//printf("yo\n");
				}
				first >>= 1;
			}
		}
		//printf("first: %d prime %d\n", (first<<1)+low_value, prime);
		// change this full logic
		for (i = first; i < size; i += prime)
		{	marked[i] = 1;
			//printf("marked %d\n", (i<<1)+low_value);
		}
		if (!my_rank)
		{	while (marked[++index] == 1);
			prime = low_value+(index<<1);
		}
		if (p > 1)
			MPI_Bcast (&prime,  1, MPI_INT,0, MPI_COMM_WORLD);
		//if ( my_rank == 0 )
		//	printf("prime: %d\n", prime);
		prime_sq=prime * prime, lval_mod_prime= low_value % prime;
	} while (prime_sq <= n0);
	count = 0;
	i=0;
	//printf("first: %d my_rank: %d\n", low_value, my_rank);
	MPI_Status status;
	for ( ; i < size; i++)
		if (!marked[i])
		{	count++;
			//printf("%d, ", low_value+i*2);
		}
	//printf("rank %d count %d\n", my_rank, count);
	if (p > 1)
		MPI_Reduce (&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	else
		global_count=count;
	global_count++;
	/* Stop the timer */
	elapsed_time += MPI_Wtime();
	/* Print the results */
	if (!my_rank)
	{	printf ("There are %d primes less than or equal to %d\n", global_count, n0);
		printf ("SIEVE (%d) %10.6f\n", p, elapsed_time);
		//printf("%d\n", global_count);
	}
	MPI_Finalize ();
	return 0;
}

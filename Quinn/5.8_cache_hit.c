#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include<stdlib.h>
#define MIN(a,b)  ((a)<(b)?(a):(b))

#define rP(n) (sieve[n>>6]|=(1<<((n>>1)&31)))
#define gP(n) sieve[n>>6]&(1<<((n>>1)&31))

// for n>10^9 change data type and memory allocation!!
int main (int argc, char *argv[])
{	int count, n, n0, p, proc0_size, prime, size, first, global_count, high_value, i, my_rank, index, low_value, i1;
	double elapsed_time;
	char *marked;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	MPI_Barrier(MPI_COMM_WORLD);
	elapsed_time = -MPI_Wtime();
	n0= atoi(argv[1]);
	n0+= n0 % 2 != 0;			// for odd no. and if the last no. itself is prime
	n=n0>>1;
	int mx= (int) sqrt((double)n0)+1;
	int S= (int)sqrt((double)mx);
	unsigned sieve[700]={0};				// these values wrt to 10^9 !!
	int j, k, l=1, primes[8000]={0};			// these values wrt 10^9 !!
	primes[0]=2;
	for(i=3; i<=S; i+=2)
	{	if(!(gP(i)))
	        {	k=(i<<1);
	        	primes[l++]=i;
			for(j=i*i;j<mx;j+=k) 
				rP(j);
		}
	}
	for(; i<mx; i+=2)					// i+2 as we consider only odd!!
	{	if(!(gP(i)))
			 primes[l++]=i;
	}
	low_value = (long long int)((my_rank*n/p)<<1)+1;
	high_value = (long long int)(((my_rank+1)*n/p)<<1)-1;
	//printf("%d %d\n", low_value, high_value);
	size = ((high_value - low_value)>>1) + 1;
	proc0_size = n/p;
	if ( proc0_size < (int) sqrt((double) n))		// we want n/p <= sqrt(n)
	{	if (!my_rank)
			printf ("Too many processes\n");
		MPI_Finalize();
		exit (1);
	}
	marked = (char *) malloc (size);
	if (marked == NULL)
	{	printf ("Cannot allocate enough memory\n");
		MPI_Finalize();
		exit (1);
	}
	int partition_size = 1000000, o, s;				// take partition size to be even!
	s= i= 0;
// i denotes the starting and ending indices of the sub-array of the process not the partition being sieved (0-19), (20-39).... 0-> 2*0+1= 1, 19= 19*2+1= 39 so == (0-19) == (1-39), (20-39)== (41,69 ) ...
	// i is index and low/high value are values of those indices
	// sequentially seiveing a partition of size partition_size- higher cache hits!
	while ( i + partition_size -1 < size )		// if size = 20 i=0 then 0+20-19 < 20 indices in subarray -[i, i+size-1]
	{	//printf("my_rank: %d i= %d\n", my_rank, i);
		for (o=i ; o < i+partition_size; o++ )		// i means i*2 + 1th in number line
			marked[o] = 0;				// initialize
		if (my_rank == 0 && s == 0 )
			marked[0]=1;				// 1 is not prime (0*2+1 = 1)
		k=1;
		prime = primes[k];
		int n_low_value= low_value + (i<<1), n_high_value =  n_low_value+((partition_size-1)<<1);	// values not pos. partition in the sub array of process is: [n_low_value, n_high_value] 	e.g: [1->19], [21-39], [41-59] ...
		int prime_sq=prime * prime, lval_mod_prime= n_low_value % prime;				// values not pos
		while ( prime_sq <= n_high_value && k < l )
		{	//printf("low: %d high: %d prime_sq: %d rank: %d\n", n_low_value, n_high_value, prime_sq, my_rank);
			if (prime_sq > n_low_value)			// start from prime^2 -> n
				first = (prime_sq - n_low_value)>>1;	// pos in the partition
			else						// locate first occurence of a multiple
			{	if (!(lval_mod_prime))
					first = 0;
				else
				{	first = prime - lval_mod_prime;
					if ( (n_low_value+first) % 2 == 0 )
						first+=prime;
					first >>= 1;			// pos in the partition
				}
			}
			//printf("first: %d\n", first);
			if ( first < partition_size)
			{	for (i1 = first; i1 < partition_size; i1 += prime)
				{	marked[i+i1] = 1;			// i1-> pos in the partition. position in the subarray= i+i1
				//printf("marked: %d\n",  ((i+ i1)<<1)+1);
				}
			}
			prime=primes[++k];
			if ( k < l )
				prime_sq=prime * prime, lval_mod_prime= n_low_value % prime;
		}
		s++;							// go to next partition
		i= s*partition_size;					// starting of next partition
	}
	partition_size= size % partition_size;
	if ( partition_size != 0 )					// if size % partition_size != 0
	{	for (o=i ; o < i+partition_size; o++ )
			marked[o] = 0;			// initialize
		if (my_rank == 0 && s == 0 )
			marked[0]=1;
		k=1;
		prime = primes[k];
		int n_low_value= low_value + (i<<1), n_high_value =  n_low_value+((partition_size-1)<<1);		// value not pos
		int prime_sq=prime * prime, lval_mod_prime= n_low_value % prime;				// value not pos
		while ( prime_sq <= n_high_value && k < l )
		{	//printf("low: %d high: %d prime_sq: %d rank: %d\n", n_low_value, n_high_value, prime_sq, my_rank);
			if (prime_sq > n_low_value)			// start from prime^2 -> n
				first = (prime_sq - n_low_value)>>1;	// pos
			else						// locate first occurence of a multiple
			{	if (!(lval_mod_prime))
					first = 0;
				else
				{	first = prime - lval_mod_prime;
					if ( (n_low_value+first) % 2 == 0 )
						first+=prime;
					first >>= 1;			// pos
				}
			}
			//printf("first %d\n", first);
			if ( first < partition_size )
			{	for (i1 = first; i1 < partition_size; i1 += prime)
					marked[i+i1] = 1;
			}
			prime=primes[++k];
			if ( k < l )
				prime_sq=prime * prime, lval_mod_prime= n_low_value % prime;
		}
	}
	//printf("done %d\n", my_rank);
	count = 0;
	i=0;
	MPI_Status status;
	for ( ; i < size; i++)
		if (!marked[i])
		{	count++;
			if ( count % 500 == 1 )
			printf("%d, ", (i<<1) +1);
		}
	if (p > 1)
		MPI_Reduce (&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	else
		global_count=count;
	global_count++;				// account for 2
	elapsed_time += MPI_Wtime();
	if (!my_rank)
	{	printf ("There are %d primes less than or equal to %d\n", global_count, n0);
		printf ("SIEVE (%d) %10.6f\n", p, elapsed_time);
		//printf("%d\n", global_count);
	}
	MPI_Finalize();
	return 0;
}
/*
Explanation:
first thing this is a no even number case that uses no extra memory
let n=30
p=2
1 3 5 7 9 11 13 15 17 19 21 23 25 27 29
==n/2 = 15 numbers (no even case so n = n/2 during sieve)

subarray for each process= n/2 = 15/2 = 7
rank0: low_value=1	high_value=13	(size=7)
rank1: low_value=15	high_value=29 	(size=8)
*/

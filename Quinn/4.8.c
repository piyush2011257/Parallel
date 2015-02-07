#include<stdio.h>
#include<mpi.h>

int eval ( int a[], int size )
{	int tmp=0, i;
	for ( i=0; i<size-1; i++)
	{	if ( a[i]+2 == a[i+1] )
			tmp++;
	}
	return tmp;
}

// till 1000 not 1000000
int main( int argc, char **argv)
{	// assume that we have a list of prime already
	int prime[169]={2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 961, 967, 971, 977, 983, 991, 997};
	int my_rank, p;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	int local_n= 169/p,i,j;
	int local_a[100];
	for ( i= my_rank*local_n, j=0; i < local_n; i++, j++)
		local_a[j]=prime[i];
	if ( my_rank != p-1 )
		local_a[j]=prime[i+1];
	int count= eval(local_a, j+1);
	printf("my_rank: %d count: %d\n", my_rank, count);
	MPI_Finalize();
	return 0;
}

/* algorithm:
Let us suppose we have prime_size no. of primes till 10^3 and these are available to all processors ( hardwired )
now to get parallelism:
we divide the set of prime_size into p sets! (NOTE not disjoint!! Reason later ) of size local_n = prime_size / p
now for each set we check using eval()
now cosider e.g.:

let use have primes till 30 and p =5

2	3	5	7	9
11	13	15	17	19
21	23	25	27	29

above is the division. Note that we can check correctly for all elements in a group but we skip check of border cases i group i.e. we can't compare 9 and 11, 19 and 21 i a single set so hence we include the first element of a successor set in the curret set to check for the border case. Hence our new distribution

2	3	5	7	9	11
11	13	15	17	19	21
21	23	25	27	29		(no inclusion for last group)
And hence we can do the calculation parallely
*/

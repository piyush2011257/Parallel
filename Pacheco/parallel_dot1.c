/* parallel_dot1.c -- Computes a parallel dot product.  Uses MPI_Allreduce.
 *
 * Input: 
 *     n: order of vectors
 *     x, y:  the vectors
 *
 * Output:
 *     the dot product of x and y as computed by each process.
 *
 * Note:  Arrays containing vectors are statically allocated.  Assumes that
 *     n, the global order of the vectors, is evenly divisible by p, the
 *     number of processes.
 *
 * See Chap 5, pp. 76 & ff in PPMPI.
 */
#include <stdio.h>
#include "mpi.h"
#define MAX_LOCAL_ORDER 100

void Print_results( float dot, int my_rank, int p )
{	int q;
	float temp;
	MPI_Status  status;
	if (my_rank == 0)
	{	printf("dot = \n");
		printf("Process 0 > %f\n", dot);
		for (q = 1; q < p; q++)
		{	MPI_Recv(&temp, 1, MPI_FLOAT, q, 0, MPI_COMM_WORLD, &status);
			printf("Process %d > %f\n", q, temp);
		}
	}
	else
		MPI_Send(&dot, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);		// making process 0 print the result! all process have same value i.e the dot product of entire array
}

float Serial_dot( float x[], float y[], int n )
{	int i;
	float sum = 0.0;
	for (i = 0; i < n; i++)
		sum = sum + x[i]*y[i];
	return sum;
}

float Parallel_dot( float local_x[], float local_y[], int n_bar )
{	float  local_dot, dot = 0.0;
	local_dot = Serial_dot(local_x, local_y, n_bar);
	// where we find the value to be added-> MPI_AllReduce() but the change is reflected in variable in all the proceses NO ROOT HERE!
	MPI_Allreduce(&local_dot, &dot, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);		// no root here!
	return dot;
}

void Read_vector( char *prompt, float  local_v[], int n_bar, int p, int my_rank )
{	int i, q;
	float temp[MAX_LOCAL_ORDER];
	MPI_Status status;
	if (my_rank == 0)
	{	printf("Enter %s\n", prompt);
		for (i = 0; i < n_bar; i++)
			scanf("%f", &local_v[i]);
		for (q = 1; q < p; q++)
		{	for (i = 0; i < n_bar; i++)
				scanf("%f", &temp[i]);
			// MPI_Bcast() when from one process to all process same data!!
			MPI_Send(temp, n_bar, MPI_FLOAT, q, 0, MPI_COMM_WORLD);
// send data to required process parallely. n_bar is the no. of locations being sent ( array of size _bar each location of tyoe MPI_FLOAT. No logn method here as data transfered to each process is different ad mutually exclusive
		}
	}
	else
		MPI_Recv(local_v, n_bar, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
}

int main(int argc, char* argv[])
{	float  local_x[MAX_LOCAL_ORDER], local_y[MAX_LOCAL_ORDER], n_bar, dot;
	int p, my_rank, n;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	if (my_rank == 0)
	{	printf("Enter the order of the vectors\n");
		scanf("%d", &n);
	}
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	n_bar = n/p;
	Read_vector("the first vector", local_x, n_bar, p, my_rank);
	Read_vector("the second vector", local_y, n_bar, p, my_rank);
	// apply dot product on the local_x[] and local_y[]
	dot = Parallel_dot(local_x, local_y, n_bar);
	Print_results(dot, my_rank, p);		// value stored in all the processes
	MPI_Finalize();
	return 0;
}

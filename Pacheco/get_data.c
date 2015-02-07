#include <stdio.h>
#include "mpi.h"

float f(float x)
{	float return_val;
	return_val = x*x;
	return return_val;
}

float Trap( float  local_a, float  local_b, int local_n, float h )
{	float integral, x;
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

/********************************************************************/
/* Function Get_data
 * Reads in the user input a, b, and n.
 * Input parameters:
 *     1.  int my_rank:  rank of current process.
 *     2.  int p:  number of processes.
 * Output parameters:  
 *     1.  float* a_ptr:  pointer to left endpoint a.
 *     2.  float* b_ptr:  pointer to right endpoint b.
 *     3.  int* n_ptr:  pointer to number of trapezoids.
 * Algorithm:
 *     1.  Process 0 prompts user for input and
 *         reads in the values.
 *     2.  Process 0 sends input values to other
 *         processes.
 */
// define I/O for different processes and pass values from I/O of 1 process to the others
void Get_data( float*  a_ptr, float*  b_ptr, int* n_ptr, int my_rank, int p )
{	int source = 0, dest, tag;
	MPI_Status status;
	if (my_rank == 0)
	{	printf("Enter a, b, and n\n");
		scanf("%f %f %d", a_ptr, b_ptr, n_ptr);
		for (dest = 1; dest < p; dest++)		// p-1 times for-loop executed parallely
		{	tag = 0;
// this is also the order of sending. a single processor sends only 1 message at a time. order of sending is a, b and c and so is the order of receiving a, b and c
			printf("dest %d\n",dest);
			printf("sending a\n");
			MPI_Send(a_ptr, 1, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
			tag = 1;
			printf("sending b\n");
			MPI_Send(b_ptr, 1, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
			tag = 2;
			printf("sending c\n");
			MPI_Send(n_ptr, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
		}
	}
	else
	{	tag = 0;
		printf("source: %d\n", my_rank);
		MPI_Recv(a_ptr, 1, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);
		tag = 1;
		printf("received a\n");
		MPI_Recv(b_ptr, 1, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);
		printf("received b\n");
		tag = 2;
		MPI_Recv(n_ptr, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
		printf("received c\n");
	}
	// process having same tag can only exchange message
}

int main(int argc, char** argv)
{	int my_rank, p, source, dest = 0, tag = 0, n, local_n;
	float a, b, h, local_a, local_b, integral, total;
	MPI_Status  status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	//printf("Enter a, b, and n\n");		// different process take up mixed values!!
	//scanf("%f %f %d", &a, &b, &n);
	Get_data(&a, &b, &n, my_rank, p);		// Pointers so that we have call by reference!!
	h = (b-a)/n;
	local_n = n/p;					// no. of trapezoids handled by 1 process
	local_a = a + my_rank*local_n*h;
	local_b = local_a + local_n*h;
	integral = Trap(local_a, local_b, local_n, h);
	if (my_rank == 0)
	{	printf("a: %f b: %f n: %d\n",a,b,n);
		total = integral;
		for (source = 1; source < p; source++)
		{	MPI_Recv(&integral, 1, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);
			total = total + integral;
		}
	}
	else
	{	MPI_Send(&integral, 1, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
		printf("a: %f b: %f n: %d\n",a,b,n);
	}
	if (my_rank == 0)
		printf("With n = %d trapezoids, our estimate of the integral from %f to %f = %f\n",n, a, b, total);
	MPI_Finalize();
}

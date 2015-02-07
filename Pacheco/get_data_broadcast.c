/* get_data2.c -- Parallel Trapezoidal Rule.  Uses 3 calls to MPI_Bcast to
 *     distribute input data.
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
 * See Chap. 5, pp. 69 & ff in PPMPI.
 */
#include <stdio.h>
#include "mpi.h"

/* Function Get_data2
 * Reads in the user input a, b, and n.
 * Input parameters:
 *     1.  int my_rank:  rank of current process.
 *     2.  int p:  number of processes.
 * Output parameters:  
 *     1.  float* a_ptr:  pointer to left endpoint a.
 *     2.  float* b_ptr:  pointer to right endpoint b.
 *     3.  int* n_ptr:  pointer to number of trapezoids.
 * Algorithm:
 *     1.  Process 0 prompts user for input and reads in the values.
 *     2.  Process 0 sends input values to other processes using three calls to MPI_Bcast.
 */
// collective routine
void Get_data2( float *a_ptr, float *b_ptr, int *n_ptr, int my_rank )
{	if (my_rank == 0)
	{	printf("Enter a, b, and n\n");
		scanf("%f %f %d", a_ptr, b_ptr, n_ptr);
	}
// MPI_Bcast() from all processes and must ahve same count and data_type for all calls to eliminate the need of status. It does not involve use of tags. And hence cant be received by MPI_Recv()
	// MPI_Bcast() sends the values from process having rank 0 to all other processes in the communicator ( collective communication )
	MPI_Bcast(a_ptr, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(b_ptr, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

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

int main(int argc, char** argv)
{	int my_rank, p, source, dest = 0, tag = 0, n, local_n;
	float a, b, h, local_a, local_b, integral, total;
	MPI_Status  status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	Get_data2(&a, &b, &n, my_rank);		// Pointers so that we have call by reference!!
	h = (b-a)/n;
	local_n = n/p;					// no. of trapezoids handled by 1 process
	local_a = a + my_rank*local_n*h;
	local_b = local_a + local_n*h;
	integral = Trap(local_a, local_b, local_n, h);
	if (my_rank == 0)
	{	//printf("a: %f b: %f n: %d\n",a,b,n);
		total = integral;
		for (source = 1; source < p; source++)
		{	MPI_Recv(&integral, 1, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);
			total = total + integral;
		}
	}
	else
	{	MPI_Send(&integral, 1, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
		//printf("a: %f b: %f n: %d\n",a,b,n);
	}
	if (my_rank == 0)
		printf("With n = %d trapezoids, our estimate of the integral from %f to %f = %f\n",n, a, b, total);
	MPI_Finalize();
}

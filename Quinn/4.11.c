#include <stdio.h>
#include "mpi.h"

float f(float x)
{	float return_val;
	/* Calculate f(x). Store calculation in return_val. */
	return_val = (float)4.0 / ( (float)1 + x*x);
	return return_val;
}

float Rect( float  local_a, float  local_b, int local_n, float h )
{	float integral=0, x;
	int i;
	x = local_a;
	for (i = 1; i <= local_n; i++)
	{	//printf("%f %f %f\n", local_a, local_b, h);
		integral = integral + (  f( (x + x + h)/2.0 )*h );
		x = x + h;
	}
	return integral;
}

int main(int argc, char** argv)
{	int my_rank, p, source, dest = 0, tag = 0, n=1000000, local_n;
	float a=0, b=1, local_a, local_b, h, integral, total;
	MPI_Status  status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	h = (float)(b-a)/n;
	local_n = (float)n/p;
	local_a = (float)a + my_rank*local_n*h;
	local_b = (float)local_a + local_n*h;
	integral = Rect(local_a, local_b, local_n, h);
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
		printf("With n = %d rectangles, our estimate of the integral from %f to %f = %f\n",n, a, b, total);
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

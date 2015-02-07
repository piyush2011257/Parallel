/* get_data1.c -- Parallel Trapezoidal Rule; uses a hand-coded 
 *     tree-structured broadcast.
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
 * See Chap. 5, pp. 65 & ff. in PPMPI.
 */
#include <stdio.h>
#include "mpi.h"

// Ceiling of log_2(x) is just the number of times x-1 can be divided by 2 until the quotient is 0.  Dividing by 2 is the same as right shift.

int Ceiling_log2( int  x )
{	/* Use unsigned so that right shift will fill leftmost bit with 0 */
	int temp = x - 1;	// 0 -> (x-1)
	int result = 0;
	while (temp != 0)
	{	temp >>=1;
		result++;
	}
	return result;
}

int I_receive( int stage, int my_rank , int *source_ptr )
{	int power_2_stage;
	/* 2^stage = 1 << stage */
	power_2_stage = 1 << stage;
	if ((power_2_stage <= my_rank) && (my_rank < 2*power_2_stage) )
	{	*source_ptr = my_rank - power_2_stage;
		return 1;
	}
	else
		return 0;
}

int I_send( int stage, int my_rank, int p, int *dest_ptr )
{	int power_2_stage;
	/* 2^stage = 1 << stage */
	power_2_stage = 1 << stage;
	if (my_rank < power_2_stage)
	{	*dest_ptr = my_rank + power_2_stage;
		return (*dest_ptr >= p) ? 0 : 1;
	}
	else
		return 0;
}

void Send( float  a, float  b, int n, int dest )
{	// tags used to avoid inter-mixing of data being transferred!
	MPI_Send(&a, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
	MPI_Send(&b, 1, MPI_FLOAT, dest, 1, MPI_COMM_WORLD);
	MPI_Send(&n, 1, MPI_INT, dest, 2, MPI_COMM_WORLD);
}

void Receive( float *a_ptr, float *b_ptr, int *n_ptr, int source )
{	MPI_Status status;
	MPI_Recv(a_ptr, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, &status);
	MPI_Recv(b_ptr, 1, MPI_FLOAT, source, 1, MPI_COMM_WORLD, &status);
	MPI_Recv(n_ptr, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
}

/* Function Get_data1
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
 *     2.  Process 0 sends input values to other processes using hand-coded tree-structured broadcast.
 */
// collective routine. MPI_Bcast() simulation
void Get_data1( float *a_ptr, float *b_ptr, int *n_ptr, int my_rank, int p)
{	int source, dest, stage;
	if (my_rank == 0)
	{	printf("Enter a, b, and n\n");
		scanf("%f %f %d", a_ptr, b_ptr, n_ptr);
	}
	for (stage = 0; stage < Ceiling_log2(p); stage++)
/*
These for loop excuted parallely for all processes! so a total of log2(p) stages only instead of p-1 stages in simple Get_Data().
These stages executed parallely for all processes. note: < ceiling_log2() NOT <= ceiling_log2(). SImulate for stage=0 and you will understand. For stage = 0. 1 receives data from 0 and at the same time sends data to 1!. for stage = 1, 3 receives data from 1 and at the same time 1 sends data to 3. Similarly for (0,2) pair.
*/
		// change in this ordering-> deadlock!
		if (I_receive(stage, my_rank, &source))
			Receive(a_ptr, b_ptr, n_ptr, source);
		else if (I_send(stage, my_rank, p, &dest))
			Send(*a_ptr, *b_ptr, *n_ptr, dest);
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
	Get_data1(&a, &b, &n, my_rank, p);		// Pointers so that we have call by reference!!
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

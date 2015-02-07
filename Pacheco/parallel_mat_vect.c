/* parallel_mat_vect.c -- computes a parallel matrix-vector product.  Matrix
 *     is distributed by block rows.  Vectors are distributed by blocks.
 *
 * Input:
 *     m, n: order of matrix
 *     A, x: the matrix and the vector to be multiplied
 *
 * Output:
 *     y: the product vector
 *
 * Notes:  
 *     1.  Local storage for A, x, and y is statically allocated.
 *     2.  Number of processes (p) should evenly divide both m and n.
 *
 * See Chap 5, p. 78 & ff in PPMPI.
 */

#include <stdio.h>
#include "mpi.h"
#define MAX_ORDER 100

typedef float LOCAL_MATRIX_T[MAX_ORDER][MAX_ORDER];

void Read_matrix( char *prompt, LOCAL_MATRIX_T local_A, int local_m, int n, int my_rank, int p )
{	int i, j;
	LOCAL_MATRIX_T temp={{0},};
	/* Fill dummy entries in temp with zeroes */
	for (i = 0; i < p*local_m; i++)
		for (j = n; j < MAX_ORDER; j++)
			temp[i][j] = 0.0;
	if (my_rank == 0)
	{	printf("%s\n", prompt);
		for (i = 0; i < p*local_m; i++)
			for (j = 0; j < n; j++)
				scanf("%f",&temp[i][j]);
	}
	// Scatter to send.
	/* Understad how it works. Fitsr we take whole matrix temp of siuze m*n in rank 0 process. Then on using scatter. The temp matrix is divided. Each block of sending size local_m * max_order is sent each process i's local_A matrix of receive size localm * max_order the default order of rank. 
	i.e. if m= 6, np=2 and max_order =10. then local_m = 6/2 = 3. so temp[0->10] and temp[1->10] are stored in local_a[0->10] andlocal_a[1->10] of process 0. temp[2->10] and temp[3->10] are stored in local_a[0->10] andlocal_a[1->10] of process 1 and temp[4->10] and temp[5->10] are stored in local_a[0->10] andlocal_a[1->10] of process 2. Hence effect of scatter
	*/		
	// has to be called by each process like mpi_bcast()
	MPI_Scatter(temp, local_m*MAX_ORDER, MPI_FLOAT, local_A, local_m*MAX_ORDER, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

// a form of get_data()
void Read_vector( char *prompt, float local_x[], int local_n, int my_rank, int p )
{	int i;
	float temp[MAX_ORDER];
	if (my_rank == 0)
	{	printf("%s\n", prompt);
		for (i = 0; i < p*local_n; i++)
			scanf("%f", &temp[i]);
	}
	// same as above for read_matrix(). has to be called by each process like mpi_bcast()
	MPI_Scatter(temp, local_n, MPI_FLOAT, local_x, local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

/* All arrays are allocated in calling program */
/* Note that argument m is unused              */
void Parallel_matrix_vector_prod( LOCAL_MATRIX_T local_A, int m, int n, float local_x[], float global_x[], float local_y[], int local_m, int local_n, int my_rank)
{	/* local_m = m/p, local_n = n/p */
	int i, j;
	// allgather means gather for all process!! no root process here! every process gathers
	// has to be called by each process like mpi_bcast()
	MPI_Allgather(local_x, local_n, MPI_FLOAT, global_x, local_n, MPI_FLOAT, MPI_COMM_WORLD);
	// if we use mpi_gather() the we'll have to write it p times for p process so that each process calls it!
	// MPI_Gather(local_x, local_n, MPI_FLOAT, global_x, local_n, MPI_FLOAT, my_rank, MPI_COMM_WORLD);
	for (i = 0; i < local_m; i++)
	{	local_y[i] = 0.0;
		for (j = 0; j < n; j++)
			local_y[i] = local_y[i] + local_A[i][j]*global_x[j];
	}
}

void Print_matrix( char *title, LOCAL_MATRIX_T local_A, int local_m, int n, int my_rank, int p )
{	int i, j;
	float temp[MAX_ORDER][MAX_ORDER];
	/* receives and stores in the order of rank i.e. if m= 6, np=2 and max_order =10. then local_m = 6/2 = 3. so temp[0->10] and temp[1->10] will store local_a[0->10] andlocal_a[1->10] of process 0. temp[2->10] and temp[3->10] will store local_a[0->10] andlocal_a[1->10] of process 1 and temp[4->10] and temp[5->10] will store local_a[0->10] andlocal_a[1->10] of process 2. Hence effect of gather
	*/
	// has to be called by each process like mpi_bcast(). gather() for one root! i.e. only 1 process gathers all the value
	MPI_Gather(local_A, local_m*MAX_ORDER, MPI_FLOAT, temp, local_m*MAX_ORDER, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if (my_rank == 0)
	{	printf("%s\n", title);
		for (i = 0; i < p*local_m; i++)
		{	for (j = 0; j < n; j++)
				printf("%4.1f ", temp[i][j]);
			printf("\n");
		}
	} 
}

void Print_vector( char *title, float local_y[], int local_m, int my_rank, int p )
{	int i;
	float temp[MAX_ORDER];
	// has to be called by each process like mpi_bcast()
	MPI_Gather(local_y, local_m, MPI_FLOAT, temp, local_m, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if (my_rank == 0)
	{	printf("%s\n", title);
		for (i = 0; i < p*local_m; i++)
			printf("%4.1f ", temp[i]);
		printf("\n");
	}
}

int main(int argc, char* argv[])
{	int my_rank, p;
	LOCAL_MATRIX_T local_A;
	float global_x[MAX_ORDER], local_x[MAX_ORDER], local_y[MAX_ORDER];
	int m, n, local_m, local_n;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	if (my_rank == 0)
	{	printf("Enter the order of the matrix (m x n)\n");
		scanf("%d %d", &m, &n);
	}
	// transfer values of n and m to all process. MPI_Bcast()
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	// local_m and local_n could have also been broadcasted
	local_m = m/p;
	local_n = n/p;
	Read_matrix("Enter the matrix", local_A, local_m, n, my_rank, p);
	Print_matrix("We read", local_A, local_m, n, my_rank, p);
	Read_vector("Enter the vector", local_x, local_n, my_rank, p);
	Print_vector("We read", local_x, local_n, my_rank, p);
	Parallel_matrix_vector_prod(local_A, m, n, local_x, global_x, local_y, local_m, local_n, my_rank);
	Print_vector("The product is", local_y, local_m, my_rank, p);
	MPI_Finalize();
}

/*
Division of matrix ad vector ampng processes. Gather / scatter accordingly

	00	01	02	03		00		00
  P0	10	11	12	12		10		10
	
	20	21	22	23		20		20
  P1	30	31	32	33		30		30

  P2	40	41	42	43				40
	50	51	52	53				50

Algo:
I/p the vector and matrix
scatter n/p rows of matrix among process (0->p ) and same for vector
for a process p
{	gatherall (vector)
	take the product and store
	gather(process 0)
}
process 0 has the product matrix
*/

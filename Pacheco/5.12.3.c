#include <stdio.h>
#include "mpi.h"
#define MAX_ORDER 100

typedef float LOCAL_MATRIX_T[MAX_ORDER][MAX_ORDER];

void Read_matrix( char *prompt, LOCAL_MATRIX_T local_A, int local_m, int n, int my_rank, int p )
{	int i, j;
	LOCAL_MATRIX_T temp={{0},};
	for (i = 0; i < p*local_m; i++)
		for (j = n; j < MAX_ORDER; j++)
			temp[i][j] = 0.0;
	if (my_rank == 0)
	{	printf("%s\n", prompt);
		for (i = 0; i < p*local_m; i++)
			for (j = 0; j < n; j++)
				scanf("%f",&temp[i][j]);
	}
	MPI_Scatter(temp, local_m*MAX_ORDER, MPI_FLOAT, local_A, local_m*MAX_ORDER, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void Parallel_matrix_vector_prod( LOCAL_MATRIX_T local_A, LOCAL_MATRIX_T local_B, int m, int n, int o, float local_x[], float global_x[], float local_y[][MAX_ORDER], int local_m, int local_n, int my_rank)
{	int i, j, l;
	for ( l=0; l<o; l++)
	{	// this way works since all the processes work parallely and only one mpi_gather at once for a single loop
		for ( i=0; i<local_n; i++)
			local_x[i]= local_B[i][l];
		MPI_Allgather(local_x, local_n, MPI_FLOAT, global_x, local_n, MPI_FLOAT, MPI_COMM_WORLD);
		for (i = 0; i < local_m; i++)
		{	local_y[i][l] = 0.0;
			for (j = 0; j < n; j++)
				local_y[i][l] = local_y[i][l] + local_A[i][j]*global_x[j];
		}
	}
}

void Print_matrix( char *title, LOCAL_MATRIX_T local_A, int local_m, int n, int my_rank, int p )
{	int i, j;
	float temp[MAX_ORDER][MAX_ORDER];
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

int main(int argc, char* argv[])
{	int my_rank, p;
	LOCAL_MATRIX_T local_A, local_B;
	float global_x[MAX_ORDER], local_x[MAX_ORDER], local_y[MAX_ORDER][MAX_ORDER];
	int m, n, o, local_m, local_n, local_o;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	if (my_rank == 0)
	{	printf("Enter the order of the matrices (m x n x o)\n");
		scanf("%d %d %d", &m, &n, &o);
	}
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&o, 1, MPI_INT, 0, MPI_COMM_WORLD);
	local_m = m/p;
	local_n = n/p;
	Read_matrix("Enter the matrix", local_A, local_m, n, my_rank, p);
	Print_matrix("We read", local_A, local_m, n, my_rank, p);
	Read_matrix("Enter the matrix", local_B, local_n, o, my_rank, p);
	Print_matrix("We read", local_B, local_n, o, my_rank, p);
	Parallel_matrix_vector_prod(local_A, local_B, m, n, o, local_x, global_x, local_y, local_m, local_n, my_rank);
	Print_matrix("Matrix Product is", local_y, local_m, o, my_rank, p);
	MPI_Finalize();
}

/*
Division of matrix ad vector ampng processes. Gather / scatter accordingly

	00	01	02	03		00	01	02		00	01	02
  P0	10	11	12	12		10	11	12		10	11	12
	
	20	21	22	23		20	21	22		20	21	22
  P1	30	31	32	33		30	31	32		30	31	32

  P2	40	41	42	43						40	41	42
	50	51	52	53						50	51	52

Algo:
i/p the vector and matrix
scatter n/p rows of matrix among process (0->p ) and same for vector
for a process p
{	for ( each column )
	{	gatherall (column_vector)
		take the product and store
	}
	gather(process 0)
}
process 0 has the product matrix
*/

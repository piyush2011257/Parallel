/*
Matrix-matrix multiplication, Fox
Data distribution of matrix: col wise
*/

#include<stdio.h>
#include<mpi.h>
#include<string.h>
#include<stdlib.h>

typedef int dtype;					// Change these two definitions when the matrix and vector element types change
#define MPI_TYPE MPI_INT
#define BLOCK_OWNER(j,p,n) (((p)*((j)+1)-1)/(n))
#define BLOCK_LOW(id,p,n)  ((id)*(n)/(p))						// low_value= (in/p)
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)					// high value (i+1)n/p-1
#define BLOCK_SIZE(id,p,n) (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)			// size of the interval
#define PTR_SIZE           (sizeof(void*))						// size of a pointer

void allocate_matrix ( dtype ***a, int r, int c )
{	dtype *local_a = (void *)malloc(r*c*sizeof(dtype));
	(*a) = (void *)malloc(r*PTR_SIZE);
	dtype *l= &local_a[0];
	int i;
	for (i=0; i<r; i++ )
	{	(*a)[i]= l;
		l += c;
	}
}

/* This function creates the count and displacement arrays needed by scatter and gather functions, when the number of elements send/received to/from other processes varies.
*/
void create_mixed_xfer_arrays ( int id, int p, int n, int *count, int *disp)
{	int i;
	count[0] = BLOCK_SIZE(0,p,n);
	disp[0] = 0;
	for (i = 1; i < p; i++)
	{	disp[i] = disp[i-1] + count[i-1];
		count[i] = BLOCK_SIZE(i,p,n);
	}
}

void read_col_striped_matrix ( dtype ***a, MPI_Datatype dtype, int n, MPI_Comm comm )
{	int i, my_rank, p, j, local_cols;
	MPI_Status status;
	MPI_Comm_size (comm, &p);
	MPI_Comm_rank (comm, &my_rank);
	local_cols= BLOCK_SIZE(my_rank, p,n);
	int *send_count = malloc ( p * sizeof(int));
	int *send_disp = malloc ( p * sizeof(int));
	int *storage= malloc (sizeof(dtype)*n);
	create_mixed_xfer_arrays (my_rank,p,n,send_count,send_disp);
	for ( i=0; i< n; i++ )
	{	if ( my_rank == p-1 )
		{	for ( j=0; j < n; j++)
				storage[j]=rand()%50;
		}
		MPI_Scatterv (storage, send_count, send_disp, MPI_TYPE, &((*a)[i][0]), local_cols, MPI_TYPE, p-1, comm);
		// we cant use local_a as it would lead to aliasing!! Keep care of aliasing!
	}
	free (send_count);
	free (send_disp);
	free (storage);
}

void set_to_zero( dtype ***a, int r, int c )
{	int i,j;
	for ( i = 0; i < r; i++)
		for (j = 0; j < c; j++)
			(*a)[i][j]=0;
}

void print_col_striped_matrix ( dtype ***a, MPI_Datatype dtype, int n, MPI_Comm comm)
{	int *bstorage= malloc (n*sizeof(dtype)), i, my_rank, local_cols, p, j;
	MPI_Comm_rank (comm, &my_rank);
	MPI_Comm_size (comm, &p);
	local_cols = BLOCK_SIZE(my_rank,p,n);
	int *recv_cnt= malloc (p * sizeof(dtype));
	int *recv_disp= malloc (p * sizeof(dtype));
	create_mixed_xfer_arrays (my_rank, p, n, recv_cnt, recv_disp);
	for ( i=0; i<n; i++ )
	{	MPI_Gatherv (&((*a)[i][0]), BLOCK_SIZE(my_rank,p,n), MPI_TYPE, bstorage, recv_cnt, recv_disp, MPI_TYPE, 0, comm);
		if ( my_rank == 0 )
		{	for ( j =0; j<n; j++ )
				printf("%d\t", bstorage[j]);
			printf("\n");
		}
	}
	free(bstorage);
	free(recv_cnt);
	free(recv_disp);
}

void fox( dtype*** local_a, dtype*** local_b, dtype*** local_c, int n, MPI_Comm comm )
{	int my_rank, source, dest, i, j, stage, p, local_cols;
	MPI_Datatype matrix_type;
	MPI_Comm_size (comm, &p);
	MPI_Comm_rank (comm, &my_rank);
	local_cols= BLOCK_SIZE(my_rank,p,n);
	for (stage = 0; stage < n; stage++)
	{	for ( i=0; i<n; i++ )
		{	int bcast_col= (i+stage)%n, bcast_root;
			int bcast_owner= BLOCK_OWNER(bcast_col, p, n);
			int curr_owner= BLOCK_OWNER(BLOCK_LOW(my_rank,p,n), p, n);
			if ( curr_owner == bcast_owner )
			{	int offset= bcast_col- BLOCK_LOW(my_rank,p,n);
				bcast_root= (*local_a)[i][offset];				
				MPI_Bcast (&bcast_root, 1, MPI_INT, bcast_owner, MPI_COMM_WORLD);
			}
			else
				MPI_Bcast (&bcast_root, 1, MPI_INT, bcast_owner, MPI_COMM_WORLD);
			for ( j =0; j<local_cols; j++ )
				(*local_c)[i][j] += bcast_root * (*local_b)[(i+stage)%n][j];
		}
	}
}

int main (int argc, char *argv[])
{	dtype **a, **b, **c;
	int my_rank, i, j, n, p, local_cols;
	
	MPI_Status status;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	
	if ( my_rank == 0)
	{	printf("Enter dimension of matrix: ");
		scanf("%d",&n);
	}
	MPI_Bcast (&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	// we take only square matrices and hence local_rows = local_cols here
	local_cols = BLOCK_SIZE( my_rank, p, n);
	allocate_matrix(&a, n, local_cols);
	allocate_matrix(&b, n, local_cols);
	allocate_matrix(&c, n, local_cols);
	set_to_zero(&c, n, local_cols);
	MPI_Barrier(MPI_COMM_WORLD);

	read_col_striped_matrix ( &a, MPI_TYPE, n, MPI_COMM_WORLD);
	read_col_striped_matrix ( &b, MPI_TYPE, n, MPI_COMM_WORLD);
	print_col_striped_matrix (&a, MPI_TYPE, n, MPI_COMM_WORLD);
	print_col_striped_matrix (&b, MPI_TYPE, n, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	fox( &a, &b, &c, n, MPI_COMM_WORLD );
	free(a);
	free(b);
	print_col_striped_matrix (&c, MPI_TYPE, n, MPI_COMM_WORLD);
	free(c);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}

/*
Matrix-matrix multiplication, Cannon
Data distribution of matrix: checkerboard using scatter ( 1 broadcast only )
Refer to Quinn- Matrix Multiplication- Cannon
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
{	dtype *local_a = (dtype *)malloc(r*c*sizeof(dtype));
	(*a) = (int**)malloc(r*PTR_SIZE);
	dtype *l= &local_a[0];
	int i;
	for (i=0; i<r; i++ )
	{	(*a)[i]= l;
		l += c;
	}
}

/* This method reduces the no. of message broadcasting steps as it uses 1 scatter only BUT at the cost of huge memory utilization )
Note no re-shuffling in process done during MPI_Cart_Create():
so suppose: p=9
ordering in MPI_COMM_WORLD:
	0 1 2
	3 4 5
	6 7 8
	(row-wise ordered). This property is used in MPI_Scatter() below
*/

void read_checkerboard_matrix ( dtype ***a, MPI_Datatype dtype, int n, MPI_Comm grid_comm, int * grid_size)
{	int coords[2], dest_id, grid_coord[2], grid_id, i, j, k, p, i1;
	MPI_Status status;
	MPI_Comm_rank (grid_comm, &grid_id);
	MPI_Comm_size (grid_comm, &p);
	int **buffer, *tmp_a;
	int local_cols= BLOCK_SIZE(grid_coord[1],grid_size[1],n);
	int local_rows= BLOCK_SIZE(grid_coord[0],grid_size[0],n);		// assume local_cols and local_rows same for all processes
	
	allocate_matrix(&buffer,n,n);
	if (grid_id == 0)
	{	allocate_matrix(&buffer,n,n);
		tmp_a= (int *) malloc(sizeof(int) * n * n );
		for ( int i=0; i<n; i++ )
			for ( int j=0; j<n; j++ )
				buffer[i][j]= rand()%50;
		/*Rearrange the input matrices in one dim arrays by approriate order*/
		for (int i = 0; i < grid_size[0]; i++)
		{	for (int j = 0; j < grid_size[1]; j++)			// i, j are grid_coord[0], [1]
			{	int Proc_Id = i * grid_size[0] + j;		// id of process in MPI_COMM_WORLD (grid_coord[0]=i, [1]=j)
				for (int irow = 0; irow < local_rows; irow++)
				{	int Global_Row_Index = i * local_rows + irow;
					for (int icol = 0; icol < local_cols; icol++)
					{	// understand this- 0-> (local_rows*local_cols)-1 for 0, (local_rows*local_cols) -> 2*(local_rows*local_cols)-1 for process 1, 2*(local_rows*local_cols) -> 3*(local_rows*local_cols)-1 -> process 2, 3*(local_rows*local_cols) -> 4*(local_rows*local_cols) -1 for process 3, .. etc.
						int Local_Index  = (Proc_Id * local_rows*local_cols) + (irow * local_cols) + icol;
						int Global_Col_Index = j * local_cols + icol;
						printf("global_row: %d global_col: %d local_index: %d\n", Local_Index, Global_Row_Index, Global_Col_Index);
						tmp_a[Local_Index] = buffer[Global_Row_Index][Global_Col_Index];
					}
				}
			}
		}
	}
	MPI_Barrier(grid_comm);
	/* Scatter the Data  to all processes by MPI_SCATTER */
	// scatter elements in the order of elemnts belonging to the processes
	MPI_Scatter (&tmp_a[0], local_rows*local_cols, MPI_FLOAT, &((*a)[0][0]), local_rows*local_cols , MPI_FLOAT, 0, grid_comm);
	if (grid_id == 0)
		free (buffer);
}

void set_to_zero( dtype ***a, int r, int c )
{	int i,j;
	for ( i = 0; i < r; i++)
		for (j = 0; j < c; j++)
			(*a)[i][j]=0;
}

// must be squate matrix only. p is square number and n/sqrt(p) == 0
void local_matrix_multiply( dtype ***a, dtype*** b, dtype*** c, int row, int col)
{	int i,j,k;
	for ( i = 0; i < row; i++)
	{	for ( j = 0; j < col; j++)
		{	for ( k = 0; k < row; k++)
				(*c)[i][j] += ( (*a)[i][k] * (*b)[k][j] );
		}
	}
}

void pre_matrix_rearange_col( dtype ***local_b, int local_rows, int local_cols, int *grid_coords, int *grid_size, MPI_Comm comm)
{	MPI_Status status;
	if ( grid_coords[1]== 0 )
		return;
	int source, dest;
	source = (grid_coords[1] + grid_coords[0]) % grid_size[1];			// Calculate addresses for circular shift of B
	dest = (grid_coords[1] + grid_size[1] - grid_coords[0]) % grid_size[1];
	//printf("%d %d source: %d dest: %d\n", grid_coords[0], grid_coords[1], source, dest);
	// ranked according to row in row_comm
	MPI_Sendrecv_replace(&((*local_b)[0][0]),  local_rows*local_cols, MPI_TYPE, dest, 0, source, 0, comm, &status);
}

void pre_matrix_rearange_row( dtype ***local_a, int local_rows, int local_cols, int *grid_coords, int *grid_size, MPI_Comm comm )
{	MPI_Status status;
	if ( grid_coords[0]== 0 )
		return;
	int source, dest;
	source = (grid_coords[1] + grid_coords[0]) % grid_size[1];			// Calculate addresses for circular shift of B
	dest = (grid_coords[1] + grid_size[1] - grid_coords[0]) % grid_size[1];
	// ranked according to col in col_comm
	MPI_Sendrecv_replace(&((*local_a)[0][0]),  local_rows*local_cols, MPI_TYPE, dest, 0, source, 0, comm, &status);
	//printf("%d %d source: %d dest: %d\n", grid_coords[0], grid_coords[1], source, dest);
	
}

void cannon( dtype*** local_a, dtype*** local_b, dtype*** local_c, MPI_Comm col_comm, MPI_Comm row_comm, int *grid_size, int *grid_coords, int local_rows, int local_cols )
{	int my_rank, source, dest, i, j, stage, bcast_root;
	MPI_Datatype matrix_type;
	MPI_Comm_rank (row_comm, &my_rank);
	MPI_Status status;

	for (stage = 0; stage < grid_size[0]; stage++)			// sqrt(p) times
	{	local_matrix_multiply(local_a, local_b, local_c, local_rows, local_cols);
		source = (grid_coords[0] + 1) % grid_size[0];			// Calculate addresses for circular shift of B
		dest = (grid_coords[0] + grid_size[0] - 1) % grid_size[0];
		MPI_Sendrecv_replace(&((*local_b)[0][0]),  local_rows*local_cols, MPI_TYPE, dest, 0, source, 0, col_comm, &status);
		source = (grid_coords[1] + 1) % grid_size[1];			// Calculate addresses for circular shift of A
		dest = (grid_coords[1] + grid_size[1] - 1) % grid_size[1];
		MPI_Sendrecv_replace(&((*local_a)[0][0]),  local_rows*local_cols, MPI_TYPE, dest, 0, source, 0, row_comm, &status);
	}
}

void print_checkerboard_matrix ( dtype **c, MPI_Comm comm, int n, int local_cols, int local_rows, int *grid_size)
{	int mat_row, mat_col, grid_row, grid_col, source, coords[2], my_rank, i, j, k;
	MPI_Comm_rank(comm, &my_rank);
	dtype *temp= (dtype*)malloc(n*sizeof(dtype));
	MPI_Status status;
	if (my_rank == 0)
	{	for (i = 0; i < n; i++)
		{	coords[0] = BLOCK_OWNER(i,grid_size[0],n);
			for (j = 0; j < grid_size[1]; j++)
			{	coords[1] = j;
				MPI_Cart_rank(comm, coords, &source);
				if (source == 0)
				{	for(k = 0; k < local_cols; k++)
						printf("%d\t", c[i][k]);
				}
				else
				{	MPI_Recv(temp, local_cols, MPI_TYPE, source, 0, comm, &status);
					for(k = 0; k < local_cols; k++)
						printf("%d\t", temp[k]);
				}
			}
			printf("\n");
		}
	}
	else
	{	for (i = 0; i < local_rows; i++) 
			MPI_Send(&(c[i][0]), local_cols, MPI_TYPE, 0, 0, comm);
	}
	free(temp);
}

int main (int argc, char *argv[])
{	dtype **a, **b, **c;
	double max_seconds, seconds;
	int grid_id, my_rank, i, j, n, p, grid_size[2], grid_coords[2], coords[2], periodic[2], grid_period[2], free_coords[2], local_cols, local_rows;
	
	MPI_Comm grid_comm;
	MPI_Comm row_comm;
	MPI_Comm col_comm;
	MPI_Status status;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	grid_size[0] = grid_size[1] = 0;
	MPI_Dims_create (p, 2, grid_size);
	periodic[0] = periodic[1] = 1;					// for MPI_send_recv_replace(). Process 0-1 == n-1 periodicity
	MPI_Cart_create (MPI_COMM_WORLD, 2, grid_size, periodic, 1, &grid_comm);
	MPI_Comm_rank (grid_comm, &grid_id);						// get_rank in the communicator after the shuffle
	MPI_Cart_coords (grid_comm, grid_id, 2, grid_coords);				// coordinate in the topology
	free_coords[0]=0, free_coords[1]=1;
	MPI_Cart_sub(grid_comm, free_coords, &row_comm);
	free_coords[0]=1, free_coords[1]=0;
	MPI_Cart_sub(grid_comm, free_coords, &col_comm);
	
	if ( grid_id == 0)
	{	printf("Enter dimension of matrix: ");
		scanf("%d",&n);
	}
	MPI_Bcast (&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	// we take only square matrices and hence local_rows = local_cols here
	local_rows = BLOCK_SIZE(grid_coords[0],grid_size[0],n);
	local_cols = BLOCK_SIZE(grid_coords[1],grid_size[1],n);
	allocate_matrix(&a, local_rows, local_cols);
	allocate_matrix(&b, local_rows, local_cols);
	allocate_matrix(&c, local_rows, local_cols);
	set_to_zero(&c, local_rows, local_cols);
	MPI_Barrier(grid_comm);

	read_checkerboard_matrix ( &a, MPI_TYPE, n, grid_comm, grid_size); 	// done local distribution of matrices
	read_checkerboard_matrix ( &b, MPI_TYPE, n, grid_comm, grid_size);
	print_checkerboard_matrix ( a, grid_comm, n, local_rows, local_cols, grid_size); 	// done local distribution of matrices
	print_checkerboard_matrix ( b, grid_comm, n, local_rows, local_cols, grid_size); 	// done local distribution of matrices
	MPI_Barrier(MPI_COMM_WORLD);
	pre_matrix_rearange_row( &a, local_rows, local_cols, grid_coords, grid_size, row_comm);
	pre_matrix_rearange_col( &b, local_rows, local_cols, grid_coords, grid_size, col_comm);
	MPI_Barrier(MPI_COMM_WORLD);
	cannon( &a, &b, &c, col_comm, row_comm, grid_size, grid_coords, local_rows, local_cols );
	free(a);
	free(b);
	MPI_Barrier(MPI_COMM_WORLD);
	print_checkerboard_matrix ( c, grid_comm, n, local_rows, local_cols, grid_size); 	// done local distribution of matrices
	free(c);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}

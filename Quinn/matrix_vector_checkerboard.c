/*
Matrix-vector multiplication, Version 3
Data distribution of matrix: checkerboard
Data distribution of vector: blocked across procs in col 0
*/
// REFER TO Programs/Pacheco/pointer_malloc_reference.c for understanding the concept of 3 pointers and fox_modified.c to get no memory error

#include<stdio.h>
#include<mpi.h>
#include<string.h>
#include<stdlib.h>

/* Change these two definitions when the matrix and vector element types change */
typedef int dtype;
#define MPI_TYPE MPI_INT
#define TYPE_ERROR -3
#define MALLOC_ERROR -2
#define BLOCK_OWNER(j,p,n) (((p)*((j)+1)-1)/(n))
#define MIN(a,b)           ((a)<(b)?(a):(b))
#define BLOCK_LOW(id,p,n)  ((id)*(n)/(p))			// low_value= (in/p)
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)		// high value (i+1)n/p-1
#define BLOCK_SIZE(id,p,n) \
                     (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)	// size of the interval
#define PTR_SIZE           (sizeof(void*))			// size of a pointer. remeber sizeof(int *) = sizeof(float *) = sizeof(void *) 
 
void *my_malloc ( int id, int bytes)
{	void *buffer;
	if ((buffer = malloc ((size_t) bytes)) == NULL)
	{	printf ("Error: Malloc failed for process %d\n", id);
		fflush (stdout);
		MPI_Abort (MPI_COMM_WORLD, MALLOC_ERROR);
	}
	return buffer;
}

/*
Function 'read_checkerboard_matrix' reads a matrix from a file. The first two elements of the file are integers whose values are the dimensions of the matrix ('m' rows and 'n' columns). What follows are 'm'*'n' values representing the matrix elements stored in row-major
order.  This function allocates blocks of the matrix to the MPI processes. The number of processes must be a square number.
*/
// FOR PRINTING CHECKERBOARD MATRIX refer Programs/Pacheco/fox_modified.c
void read_checkerboard_matrix ( dtype **subs, dtype *storage, MPI_Datatype dtype, int m, int n, MPI_Comm grid_comm, int * grid_size)
{	int coords[2];			/* Coords of proc receiving next row of matrix */
	int dest_id;			/* Rank of receiving proc */
	int grid_coord[2];		/* Process coords */
	int grid_id;			/* Process rank */
	int i, j, k, p, i1;
	MPI_Status status;		/* Results of read */
	MPI_Comm_rank (grid_comm, &grid_id);
	MPI_Comm_size (grid_comm, &p);
	int *buffer, *raddr;
	int local_cols= BLOCK_SIZE(grid_coord[1],grid_size[1],n);
	
	// Grid process 0 reads in the matrix one row at a time and distributes each row among the MPI processes.
	if (grid_id == 0)
		buffer = (int*)my_malloc (grid_id, n * sizeof(dtype));
	
	//printf("%d %d\n", grid_size[0], grid_size[1]);
	MPI_Barrier(grid_comm);
	// For each row of processes in the process grid
	for (i = 0; i < grid_size[0]; i++)				// grid_size[0]=row, grid_size[1]=col
	{	coords[0] = i;
		// For each matrix row controlled by this proc row
		for (j = 0; j < BLOCK_SIZE(i,grid_size[0],m); j++)		// no. of rows in each processor block
		{	if (grid_id == 0)					// Read in a row of the matrix
			{	for ( i1=0; i1<n; i1++ )
				{	buffer[i1]=rand()%10;
					//printf("%d ", buffer[i1]);
				}
				//printf("\n");
			}
			
			// Distribute it among process in the grid row
			for (k = 0; k < grid_size[1]; k++)			// total column in the process
			{	coords[1] = k;
				raddr = buffer + BLOCK_LOW(k,grid_size[1],n);	// Find address of first element to send
				MPI_Cart_rank (grid_comm, coords, &dest_id);	// Determine the grid_ID of the process getting the subrow
				if (grid_id == 0)				// Process 0 is responsible for sending
				{	if (dest_id == 0)			// It is sending (copying) to itself
						memcpy (subs[j], raddr, local_cols * sizeof(dtype));	// It is sending to another process
					else
						MPI_Send (raddr, BLOCK_SIZE(k,grid_size[1],n), dtype, dest_id, 0, grid_comm);
				}
				else if (grid_id == dest_id)			// Process 'dest_id' is responsible for receiving
					MPI_Recv (subs[j], local_cols, dtype, 0, 0, grid_comm,&status);
				// else go idle since the row being read and transmitted does not belong to this processor. Wait for furthur iterations to encounter its data!
			}
		}
	}
	/*if (grid_id == 0)
		free (buffer);*/
}

void create_mixed_xfer_arrays ( int id, int p, int n, int *count, int *disp)
{	int i;
	// order in the process number. count = no. of elements being sent, disp is the local offset of ith process's data
	count[0] = BLOCK_SIZE(0,p,n);
	disp[0] = 0;
	for (i = 1; i < p; i++)
	{	disp[i] = disp[i-1] + count[i-1];
		count[i] = BLOCK_SIZE(i,p,n);
	}
}

void read_block_vector ( dtype *v, MPI_Datatype dtype, int n, MPI_Comm grid_comm, MPI_Comm row_comm, int *grid_size, int local_cols)
{	int i, my_rank, p, grid_coords[2];
	MPI_Comm_rank(grid_comm, &my_rank);
	MPI_Cart_coords (grid_comm, my_rank, 2, grid_coords);				// coordinate in the topologyMPI_Barrier(grid_comm);
	// bcasting later
	if ( grid_coords[0] != 0 )
		return;
	int *buffer= my_malloc( my_rank, (size_t)n * sizeof(dtype));
	int *send_count = my_malloc (my_rank, grid_size[1] * sizeof(int));
	int *send_disp = my_malloc (my_rank, grid_size[1] * sizeof(int));
	create_mixed_xfer_arrays (my_rank,grid_size[1],n,send_count,send_disp);
	//printf("vector enter %d %d\n", grid_coords[0], grid_coords[1]);
	if ( grid_coords[0] == 0 )
	{	if (my_rank == 0)	// Process 0 opens file, determines number of vector elements, and scattersitto others in row_comm
		{	for ( i=0; i<n; i++ )			// all stored in 0 first then scattered to individual blocks
				buffer[i]= 1;
		}
		MPI_Scatterv (buffer, send_count, send_disp, MPI_TYPE, v, local_cols, MPI_TYPE, 0, row_comm);	// process 0 in row_comm!! Important concept!!
		//printf("vector receive %d %d\n", grid_coords[0], grid_coords[1]);
	}
	free(send_count);
	free(send_disp);
	free(buffer);
}

void print_subvector ( dtype *a, MPI_Datatype dtype, int n)
{	int i;
	for (i = 0; i < n; i++)
	{	if (dtype == MPI_DOUBLE)
			printf ("%6.3f ", ((double *)a)[i]);
		else
		{	if (dtype == MPI_FLOAT)
				printf ("%6.3f ", ((float *)a)[i]);
			else if (dtype == MPI_INT)
				printf ("%6d ", ((int *)a)[i]);
		}
	}
}

void print_block_vector ( dtype *v, MPI_Datatype dtype, int n, MPI_Comm comm )
{	int i, p, my_rank;
	MPI_Status status;				/* Result of receive */
	MPI_Comm_size (comm, &p);
	MPI_Comm_rank (comm, &my_rank);
	int tmp[BLOCK_SIZE(p-1,p,n)];
	if (!my_rank)
	{	print_subvector (v, dtype, BLOCK_SIZE(my_rank,p,n));
		if (p > 1)
		{	// alternately we can use gatherv
			for (i = 1; i < p; i++)
			{	MPI_Recv (tmp, BLOCK_SIZE(i,p,n), dtype, i, i, comm, &status);
				print_subvector (tmp, MPI_TYPE, BLOCK_SIZE(i,p,n));
			}
		}
		printf ("\n\n");
	}
	else
		MPI_Send (v, BLOCK_SIZE(my_rank,p,n), dtype, 0, my_rank, comm);
	MPI_Barrier(comm);
}

int main (int argc, char *argv[])
{	dtype **a;       				/* First factor, a matrix */
	dtype *b, *l, *local_a;       			/* Second factor, a vector */
	dtype *c_block;  				/* Partial product vector */
	dtype *c_sums;
	double max_seconds, seconds;
	dtype *storage;					/* Matrix elements stored here */
	int grid_id, my_rank, i, j, m, n, nb, p;
	int grid_size[2]; 				/* Number of procs in each grid dimension */
	MPI_Comm grid_comm;
	MPI_Comm row_comm;
	MPI_Comm col_comm;
	int grid_coords[2], coords[2], periodic[2];
	int grid_period[2];				/* Wraparound */
	int local_cols, local_rows;
 	MPI_Status status;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	grid_size[0] = grid_size[1] = 0;
	MPI_Dims_create (p, 2, grid_size);
	periodic[0] = periodic[1] = 0;
	MPI_Cart_create (MPI_COMM_WORLD, 2, grid_size, periodic, 1, &grid_comm);
	MPI_Comm_rank (grid_comm, &grid_id);						// get_rank in the communicator after the shuffle
	MPI_Cart_coords (grid_comm, grid_id, 2, grid_coords);				// coordinate in the topology
	MPI_Comm_split (grid_comm, grid_coords[0], grid_coords[1], &row_comm);		// row and col communicators
	MPI_Comm_split (grid_comm, grid_coords[1], grid_coords[0], &col_comm);
	if ( grid_id == 0)
	{	m=n=6000;									// no i/p for an process except 0
		nb=6000;
	}
	MPI_Bcast (&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast (&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast (&nb, 1, MPI_INT, 0, MPI_COMM_WORLD);
	//Each process determines the size of the submatrix it is responsible for.
	//MPI_Cart_get (grid_comm, 2, grid_size, grid_period, grid_coord);
	local_rows = BLOCK_SIZE(grid_coords[0],grid_size[0],m);				// (i,p,n)
	local_cols = BLOCK_SIZE(grid_coords[1],grid_size[1],n);
	local_a = (dtype *) malloc ( (size_t) local_cols * local_rows * sizeof(dtype));
	b = (dtype *) malloc ( (size_t) local_cols * sizeof(dtype));
	a = (dtype **) malloc ( (size_t) local_rows * PTR_SIZE);			// pointer type
	l= &local_a[0];
	for (i=0; i<local_rows; i++ )
	{	a[i]= l;
		l += local_cols;
	}
	//printf("rank: %d cord: %d %d\n", grid_id, grid_coords[0], grid_coords[1]);
	
	MPI_Barrier(grid_comm);
	read_checkerboard_matrix ( a, storage, MPI_TYPE, m, n, grid_comm, grid_size);

	// Vector divided among processes in first column
	MPI_Barrier(MPI_COMM_WORLD);
	read_block_vector (b, MPI_TYPE, nb, grid_comm, row_comm, grid_size, local_cols);
	
	local_rows = BLOCK_SIZE(grid_coords[0],grid_size[0],m);				// (i,p,n)
	local_cols = BLOCK_SIZE(grid_coords[1],grid_size[1],n);

	c_block = (dtype *) malloc (local_rows * sizeof(dtype));
	c_sums = (dtype *) malloc (local_rows * sizeof(dtype));
	seconds = - MPI_Wtime();
	
	// Row 0 procs broadcast their subvectors to procs in same column
	MPI_Bcast (b, local_cols, MPI_TYPE, 0, col_comm);
	MPI_Barrier(grid_comm);
	/*printf("rank:%d row: %d col: %d\n",grid_id, grid_coords[0], grid_coords[1]);
	printf("matrix\n");
	for ( i=0; i<local_rows; i++)
	{	for (j=0; j<local_cols; j++ )
			printf("%d  ", a[i][j]);
		printf("\n");
	}
	printf("vector\n");
	for( i=0; i<local_cols; i++ )
		printf("%d  ", b[i]);
	printf("\n");*/
	for (i = 0; i < local_rows; i++)
	{	c_block[i] = 0.0;
		for (j = 0; j < local_cols; j++)
			c_block[i] += a[i][j] * b[j];
	}
	MPI_Barrier(grid_comm);
	MPI_Reduce(c_block, c_sums, local_rows, MPI_TYPE, MPI_SUM, 0, row_comm);
	// again due to row-major ordering the first column's process is 0th in row_comm (concept)
	//if (grid_coords[1] == 0)			// only for 0th olumn elements!
	//	print_block_vector (c_sums, MPI_TYPE, m, col_comm);
	MPI_Barrier(MPI_COMM_WORLD);
	seconds += MPI_Wtime();
	MPI_Reduce (&seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (!my_rank)
	{	printf ("MV5) N = %d, Processes = %d, Time = %12.6f sec,", n, p, max_seconds);
		printf ("Mflop = %6.2f\n", 2*n*n/(1000000.0*max_seconds));
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}
/* Documenting
here -np = 4

rank: 0 cord: 0 0
rank: 2 cord: 1 0
rank: 1 cord: 0 1
rank: 3 cord: 1 1

grid_comm
0	1
2	3

00	01
10	11

row_comm1
row_comm2

col_comm1	col_comm2

Remember a property that processes are always ordered in a communicator!
e.g.
row_comm1-	0 (0,0)		1 (0,1)
row_comm2-	0 (1,0)		1 (1,1)		( in local row_comm process numbering / my_rank starts from 0 i.e. first column of each row has rank 0 in row_comm. Same for column communicators )

Process 0 reads each row and distributes it among process in row_comm
1st 0th process in grid_comm reads 1 row and then distributes the columns among the processes in the row_comm.
Here we take 8 * 8 matrix:

3 6 7 5 3 5 6 2 
9 1 2 7 0 9 3 6 
0 6 2 6 1 8 7 9 
2 0 2 3 7 5 9 2 
2 8 9 7 3 6 1 2 
9 3 1 9 4 7 8 4 
5 0 3 6 1 0 6 3 
2 0 6 1 5 5 4 7

how we wish to scatter

3 6 7 5		 3 5 6 2 
9 1 2 7		 0 9 3 6 
0 6 2 6		 1 8 7 9 
2 0 2 3		 7 5 9 2 


2 8 9 7		 3 6 1 2 
9 3 1 9		 4 7 8 4 
5 0 3 6		 1 0 6 3 
2 0 6 1		 5 5 4 7

total processes in a row= grid_size[1] -> no. of columns in the grid == p
total columns in a row = n
using in/p concept we distribute each input rows among the processes.
Now which process in a column will get the row is decided by using the same logic
total processes in a col= grid_size[0] -> no. of rows in the grid == p
total rows in matrix = m
using im/p concept we distribute each input rows among the processes.
these give us the local_rows and local_cols for a process in the grid
( here it is local_cols = local_rows = 4 )

Now for vector
process 0 in grid_comm takes all the i/p vector and again distributes the vectors to the pricessor in the first row of every column.
vector
1 1 1 1 1 1 1 1
1 1 1 1			1 1 1 1

these processes are ranked in 0 in theor respective col_comms due to ordering property. These now bcast the vector to each process in the col_comm
3 6 7 5	  1		 3 5 6 2   1
9 1 2 7	  1		 0 9 3 6   1
0 6 2 6	  1		 1 8 7 9   1
2 0 2 3	  1		 7 5 9 2   1


2 8 9 7   1		 3 6 1 2   1
9 3 1 9	  1		 4 7 8 4   1
5 0 3 6	  1		 1 0 6 3   1
2 0 6 1	  1		 5 5 4 7   1

Now we do local matrix-vector multiplication.
21		16
19		18
14		25
7		23

26		12
22		23
14		10
9		21
which is followed by reduce ( total count of reduce is equal to the local_rows )

37
37
39
30

38
45
24
30

and hence the first col of evry row stores the matrix vector product of the block_size elements
*/

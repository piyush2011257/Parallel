#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Memory error in using double!
typedef int dtype;
#define TYPE_ERROR -3
#define MALLOC_ERROR -2
#define MPI_TYPE MPI_INT
#define BLOCK_OWNER(j,p,n) (((p)*((j)+1)-1)/(n))
#define MIN(a,b)           ((a)<(b)?(a):(b))
#define BLOCK_LOW(id,p,n)  ((id)*(n)/(p))			// low_value= (in/p)
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)		// high value (i+1)n/p-1
#define BLOCK_SIZE(id,p,n) \
                     (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)	// size of the interval
#define PTR_SIZE           (sizeof(void*))			// size of a pointer. remeber sizeof(int *) = sizeof(float *) = sizeof(void *) = .. (all pointer types have same size)

int get_size (MPI_Datatype t)
{	if (t == MPI_BYTE)			// associating MPI_Datatypes with corresponding C types
		return sizeof(char);
	if (t == MPI_DOUBLE)
		return sizeof(double);
	if (t == MPI_FLOAT)
		return sizeof(float);
	if (t == MPI_INT)
		return sizeof(int);
	printf ("Error: Unrecognized argument to 'get_size'\n");
	fflush (stdout);
	MPI_Abort (MPI_COMM_WORLD, TYPE_ERROR);
}

void terminate ( int id, char *error_message) 		/* IN - Message to print */
{	if (!id)
	{	printf ("Error: %s\n", error_message);
		fflush (stdout);
	}
	MPI_Finalize();					// finish off cleaning process
	exit (-1);
}

void *my_malloc ( int id, int bytes)
{	void *buffer;
	if ((buffer = malloc ((size_t) bytes)) == NULL)
	{	printf ("Error: Malloc failed for process %d\n", id);
		fflush (stdout);
		MPI_Abort (MPI_COMM_WORLD, MALLOC_ERROR);
	}
	return buffer;
}

/* This function creates the count and displacement arrays needed by scatter and gather functions, when the number of elements send/received to/from other processes varies.
*/
void create_mixed_xfer_arrays ( int id, int p, int n, int *count, int *disp)
{	int i;
	// order in the process number. count = no. of elements being sent, disp is the local offset of ith process's data
	// undertsand how this works
	count[0] = BLOCK_SIZE(0,p,n);
	disp[0] = 0;
	for (i = 1; i < p; i++)
	{	disp[i] = disp[i-1] + count[i-1];
		count[i] = BLOCK_SIZE(i,p,n);
	}
}

/* This function creates the count and displacement arrays needed in an all-to-all exchange, when a process gets the same number of elements from every other process.
*/
void create_uniform_xfer_arrays ( int id, int p, int n, int *count, int *disp)
{	int i;
	count[0] = BLOCK_SIZE(id,p,n);
	disp[0] = 0;
	for (i = 1; i < p; i++)
	{	disp[i] = disp[i-1] + count[i-1];
		count[i] = BLOCK_SIZE(id,p,n);
	}
}

void print_submatrix ( dtype **a, MPI_Datatype dtype, int rows, int cols)
{	int i, j;
	for (i = 0; i < rows; i++)
	{	for (j = 0; j < cols; j++)
		{	if (dtype == MPI_DOUBLE)
				printf ("%6.3f ", ((double **)a)[i][j]);
			else
			{	if (dtype == MPI_FLOAT)
					printf ("%6.3f ", ((float **)a)[i][j]);
				else if (dtype == MPI_INT)
					printf ("%6d", ((int **)a)[i][j]);
			}
		}
		putchar ('\n');
	}
}

void print_col_striped_matrix ( dtype **a, MPI_Datatype dtype, int m, int n, MPI_Comm comm)
{	MPI_Status status;
	int bstorage[n];
	int size_of_type, i, my_rank, local_cols, p, max_block_size, j;
	MPI_Comm_rank (comm, &my_rank);
	MPI_Comm_size (comm, &p);
	local_cols = BLOCK_SIZE(my_rank,p,m);
	//bstorage= my_malloc( my_rank, n * sizeof(dtype));
	int recv_cnt[p], recv_disp[p];
	create_mixed_xfer_arrays (my_rank, p, n, recv_cnt, recv_disp);
	/*for ( i=0; i<p; i++ )
		printf("recv_cnt[%d]: %d  recv_disp[%d]: %d\n", i, recv_cnt[i], i, recv_disp[i]);
	printf("\n");*/
	for ( i=0; i<m; i++ )
	{	MPI_Gatherv (a[i], BLOCK_SIZE(my_rank,p,n), MPI_TYPE, bstorage, recv_cnt, recv_disp, MPI_TYPE, 0, comm);
		//MPI_Gatherv(a[i], local_cols, MPI_TYPE, bstorage, local_cols, MPI_TYPE, 0, comm);		// a[i] not a+i
		if ( my_rank == 0 )
		{	for ( j =0; j<n; j++ )
				printf("%d  ", bstorage[j]);
			printf("\n");
		}
	}
	//free (bstorage);
}

void read_block_vector ( dtype *v, MPI_Datatype dtype, int n, MPI_Comm comm)
{	int i, j;
	MPI_Status status;       /* Result of receive */
	int my_rank, p;
	MPI_Comm_size(comm, &p);
	MPI_Comm_rank(comm, &my_rank);
	MPI_Barrier(comm);
	/* Process p-1 opens file, determines number of vector elements, and broadcasts this value to the other processes. */
	if (my_rank == (p-1))
	{	for (i = 0; i < p-1; i++)
		{	for ( j=0; j < BLOCK_SIZE(i,p,n); j++ )
				v[j]= 1;
			MPI_Send (v, BLOCK_SIZE(i,p,n), dtype, i, i, comm);
		}
		for ( j=0; j < BLOCK_SIZE(p-1,p,n); j++ )
			v[j]= 1;
	}
	else
		MPI_Recv (v, BLOCK_SIZE(my_rank,p,n), dtype, p-1, my_rank, comm, &status);
	MPI_Barrier(comm);
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

void print_block_vector ( dtype *v, MPI_Datatype dtype, int n, MPI_Comm comm)
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

void read_col_striped_matrix ( dtype **subs, MPI_Datatype dtype, int m, int n, MPI_Comm comm, int local_cols)
{	int i, my_rank, p,j;
	MPI_Status status;
	MPI_Comm_size (comm, &p);
	MPI_Comm_rank (comm, &my_rank);
	/*double *storage= my_malloc( my_rank, n * sizeof(dtype));*/
	int *send_count = my_malloc (my_rank, p * sizeof(int));
	int *send_disp = my_malloc (my_rank, p * sizeof(int));
	int storage[n];
	create_mixed_xfer_arrays (my_rank,p,n,send_count,send_disp);
	/*for ( i=0; i<m; i++ )
	{	for ( j=0; j< local_cols; j++ )
			printf("%p ", subs[i]+j);
		printf("\n");
	}
	printf("\n");
	for ( i=0; i<n; i++ )
		printf("rank: %d  storage: %p\n", my_rank, storage+i);
	printf("\n");*/
	/*for ( i=0; i<p; i++ )
		printf("rank: %d  disp: %p  disp[%d]: %d  count: %p  count[%d]: %d\n", my_rank, send_disp+i, i, send_disp[i], send_count+i, i, send_count[i]);
	printf("\n\n");*/
	MPI_Barrier(comm);
	{	int i1,j;
		for ( j=0; j< m; j++ )
		{	if ( my_rank == p-1 )
			{	for ( i1=0; i1 < n; i1++)
					storage[i1]=j+1;
				/*for ( i1=0; i1 < n; i1++ )
					printf("%d  ", storage[i1]);
				printf("\n");*/
			}
			//printf("scattered %dth row by %d process\n", j, my_rank);
			MPI_Scatterv (storage, send_count, send_disp, MPI_TYPE, subs[j], local_cols, MPI_TYPE, p-1, comm);
			/*for ( i1=0; i1 < n; i1++ )
				printf("%lf  ", subs[j][i1]);
			printf("\n");*/
			// we cant use local_a as it would lead to aliasing!! Keep care of aliasing!
		}
		MPI_Barrier(comm);
	}
	//free (send_count);
	//free (send_disp);
}

int main (int argc, char *argv[])
{	dtype **a;				/* matrix */
	dtype *b;				/* vector */
	dtype *c;				/* The product, a vector */
	dtype  *c_part_out;			/* Partial sums, sent */
	dtype  *c_part_in;			/* Partial sums, received */
	int *cnt_out;				/* Elements sent to each proc */
	int *cnt_in;				/* Elements received per proc */
	int *disp_out;				/* Indices of sent elements */
	int *disp_in;				/* Indices of received elements */
	int i, j, my_rank, m, n, p, nb, local_cols, size_of_type;
	double max_seconds, seconds;
	dtype *rptr, *local_a;			/* This process's portion of 'a' */
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	
	if ( my_rank == (p-1))
	{	m=n=6000;							// no i/p for an process except 0
		nb=6000;
	}
	MPI_Bcast (&m, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	MPI_Bcast (&n, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	MPI_Bcast (&nb, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	size_of_type = get_size(MPI_INT);
	local_cols = BLOCK_SIZE(my_rank,p,n);
	local_a = my_malloc (my_rank, local_cols * m * size_of_type);
	b = my_malloc (my_rank, local_cols * size_of_type);
	a = my_malloc (my_rank, m * PTR_SIZE);				// a will store pointer to a pointer type!!
	rptr = (dtype *)local_a;
	a[0]= rptr;
	for (i = 1; i < m; i++)
	{	rptr += (local_cols);					//size_of_type;		// go to next row
		a[i]= (dtype *) (rptr);
	}
	read_col_striped_matrix ( a, MPI_TYPE, m, n, MPI_COMM_WORLD, local_cols);
	//print_col_striped_matrix (a, MPI_TYPE, m, n, MPI_COMM_WORLD);
	read_block_vector (b, MPI_TYPE, nb, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	//print_block_vector (b, MPI_TYPE, nb, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	/* Each process multiplies its columns of 'a' and vector 'b', resulting in a partial sum of product 'c'. */
	c_part_out = (dtype *) my_malloc (my_rank, n * sizeof(dtype));
	MPI_Barrier (MPI_COMM_WORLD);
	seconds = -MPI_Wtime();
	for (i = 0; i < n; i++)
	{	c_part_out[i] = 0.0;
		for (j = 0; j < local_cols; j++)
			c_part_out[i] += a[i][j] * b[j];		// stores c[i][j] partial. store in a single element as they will eventually be added!! This will allow it to be alltoallexchange()
	}
	cnt_out = my_malloc (my_rank, p * sizeof(int));
	disp_out = my_malloc (my_rank, p * sizeof(int));
	cnt_in = my_malloc (my_rank, p * sizeof(int));
	disp_in = my_malloc (my_rank, p * sizeof(int));
	create_mixed_xfer_arrays (my_rank, p, n, cnt_out, disp_out);
	create_uniform_xfer_arrays ( my_rank, p, n, cnt_in, disp_in);
	/*printf("send count and send_disp\n", my_rank);
	for ( i=0; i <p; i++ )
		printf("%d send_disp[%d]: %d send_count[%d]: %d\n", my_rank, i, disp_out[i], i, cnt_out[i]);
	printf("\n");
	printf("receive count and receive disp\n");
	for ( i=0; i <p; i++ )
		printf("%d recv_disp[%d]: %d recv_count[%d]: %d\n", my_rank, i, disp_in[i], i, cnt_in[i]);
	printf("\n");*/
	c_part_in = (dtype*) my_malloc (my_rank, p*local_cols*sizeof(dtype));
	MPI_Alltoallv (c_part_out, cnt_out, disp_out, MPI_TYPE, c_part_in, cnt_in, disp_in, MPI_TYPE, MPI_COMM_WORLD);
	c = (dtype*) my_malloc (my_rank, local_cols * sizeof(dtype));
	for (i = 0; i < local_cols; i++)
	{	c[i] = 0.0;
		for (j = 0; j < p; j++)
			c[i] += c_part_in[i + j*local_cols];
	}
	MPI_Barrier (MPI_COMM_WORLD);
	seconds += MPI_Wtime();
	MPI_Allreduce (&seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	if (!my_rank)
	{	printf ("MV3) N = %d, Processes = %d, Time = %12.6f sec,", n, p, max_seconds);
		printf ("Mflop = %6.20lf\n", (double)2*n*n/((double)1000000.0*max_seconds));
	}
	//print_block_vector ((dtype *) c, MPI_TYPE, n, MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}
/* 
Col matrix vector
1- read m,n and nb
2- Bcast all 3
3- Read col matrix
Process p-1 reads each row and then sends via scatter to all the processes the corresponding columns
Sends and receive ( 1 * local_cols submatrix at a time. This is done m times )

Send-Recv
	0	1	2	3
0	1*2	1*2	1*2	1*2
1	1*2	1*2	1*2	1*2
2	1*2	1*2	1*2	1*2
3	1*2	1*2	1*2	1*2
5	1*2	1*2	1*2	1*2
6	1*2	1*2	1*2	1*2
7	1*2	1*2	1*2	1*2
8	1*2	1*2	1*2	1*2

so we get
	0	1	2	3
	8*2	8*2	8*2	8*2

4- Read vector by p-1 and divide it among the processes on the basis of local_cols
1	1	1	1	1	1	1	1

	 0	 1	 2	 3
	1 1	1 1	1 1	1 1

Now each process has its local matrix and vector

5- Do local computation
6-	2 ways now! First is to do an alltoallexchange as mentioned in the book and 2nd approach is to do reduce of m rows ( as is done in checkerboard method )

take an e.g. and solve:

m=n=nb=6
p=3;
local_cols=2;

matrix
  0	    1	        2
1  1	  1  1	      1  1  
2  2	  2  2	      2  2  
3  3	  3  3	      3  3  
4  4	  4  4	      4  4  
5  5	  5  5	      5  5  
6  6	  6  6	      6  6 

vector:
1  1	  1  1        1  1

for multipying:
1  1		1  1	  	    1  1  
2  2	  	2  2	  	    2  2  
3  3	1[0]  	3  3	1[2]	    3  3    1[4]
4  4	1[1]  	4  4	1[3] 	    4  4    1[5]
5  5	  	5  5		    5  5  
6  6	  	6  6	      	    6  6 

c_part_out
2	    2		2
4	    4		4
6	    6		6
8	    8		8
10	   10	       10
12	   12	       12

send_disp[0]: 0 send_count[0]: 2			allscatter
send_disp[1]: 2 send_count[1]: 2
send_disp[2]: 4 send_count[2]: 2

receive count and receive disp				allgather
recv_disp[0]: 0 recv_count[0]: 2
recv_disp[1]: 2 recv_count[1]: 2
recv_disp[2]: 4 recv_count[2]: 2

after all to all exhcange
c_part_in
2 4	   6 8		10 12
2 4	   6 8		10 12
2 4	   6 8		10 12

c
6 12	  18 24		30 36


understanding how : create_mixed_xfer_arrays () and create_uniform_xfer_arrays () works
let p=4
	
matrix
0	  1	 2       3
1  	1  1  	 1      1  1  
2 	2  2 	 2      2  2  
3 	3  3 	 3      3  3  
4 	4  4  	 4      4  4  
5 	5  5  	 5      5  5  
6 	6  6  	 6      6  6 

vector:
1 	1  1	 1 	1  1

for multipying:
1		1  1	  	1	    1  1  
2	  	2  2	  	2	    2  2  
3  1[0]  	3  3	1[1]	3 1[3]	    3  3    1[4]
4    		4  4	1[2] 	4	    4  4    1[5]
5  	  	5  5		5	    5  5  
6  	  	6  6	      	6	    6  6 

c_part_out
1	2	    1		2
2	4	    2		4
3	6	    3		6
4	8	    4		8
5	10	    5	       10
6	12	    6	       12

Read pg- 197 as well. 
understanding how receive and send work ( ALTERNATIVELY WE COULD HAVE DONE REDUCE AMONG THE ROW_COMMS. REFER TO CHECKRBOARD FOR THIS WAY )
note that at the end
process 0-	c[0] sum of all 0th row products
process 1-	c[1], c[2] sum of all 1st, 2nd row products
process 2-	c[3] sum of all 3rd row products
process 3-	c[4], c[5] sum of all 4th and 5th row products
and thus how send and receive behave

0 send_disp[0]: 0 send_count[0]: 1		// send_count= block_size()
0 send_disp[1]: 1 send_count[1]: 2
0 send_disp[2]: 3 send_count[2]: 1
0 send_disp[3]: 4 send_count[3]: 2

0 recv_disp[0]: 0 recv_count[0]: 1		// send_cnt[i] is the block_size! ( block_size of 0th column ) WRITE, SIMULATE, UNDERSTAND
0 recv_disp[1]: 1 recv_count[1]: 1
0 recv_disp[2]: 2 recv_count[2]: 1
0 recv_disp[3]: 3 recv_count[3]: 1

1 send_disp[0]: 0 send_count[0]: 1
1 send_disp[1]: 1 send_count[1]: 2
1 send_disp[2]: 3 send_count[2]: 1
1 send_disp[3]: 4 send_count[3]: 2

// recv count= block size ( rows of matrix handled ). This is bcuz size of vector = size of col= size of row of matrix
1 recv_disp[0]: 0 recv_count[0]: 2
1 recv_disp[1]: 2 recv_count[1]: 2
1 recv_disp[2]: 4 recv_count[2]: 2
1 recv_disp[3]: 6 recv_count[3]: 2

2 send_disp[0]: 0 send_count[0]: 1
2 send_disp[1]: 1 send_count[1]: 2
2 send_disp[2]: 3 send_count[2]: 1
2 send_disp[3]: 4 send_count[3]: 2

2 recv_disp[0]: 0 recv_count[0]: 1
2 recv_disp[1]: 1 recv_count[1]: 1
2 recv_disp[2]: 2 recv_count[2]: 1
2 recv_disp[3]: 3 recv_count[3]: 1

3 send_disp[0]: 0 send_count[0]: 1
3 send_disp[1]: 1 send_count[1]: 2
3 send_disp[2]: 3 send_count[2]: 1
3 send_disp[3]: 4 send_count[3]: 2

3 recv_disp[0]: 0 recv_count[0]: 2			// recv_count[i] is the no. of elements sent by process i = block_size of process i!
3 recv_disp[1]: 2 recv_count[1]: 2
3 recv_disp[2]: 4 recv_count[2]: 2
3 recv_disp[3]: 6 recv_count[3]: 2

c_part_in
1	2 3	4	5 6			process 0 as all 0th row elemnet of matrix
2	4 6	8	10 12			process 1 as all 1th and 2nd row element of matrix
1	2 3	4	5 6			process 2 as all 3rd row element of matrix
2	4 6	8	10 12			process 3 as all 4th and 5th row element of matrix


c
6	12 18	24	30 36
*/

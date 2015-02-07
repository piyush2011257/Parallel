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

void read_replicated_vector ( dtype *v, MPI_Datatype dtype, int n, MPI_Comm comm)
{	int i, my_rank, p;
	MPI_Comm_rank (comm, &my_rank);
	MPI_Comm_size (comm, &p);
	if (my_rank == (p-1)) 
	{	for ( i=0; i<n; i++ )
			v[i]= (double) my_rank * (my_rank+1);
	}
	MPI_Bcast (v, n, dtype, p-1, MPI_COMM_WORLD);
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

void print_replicated_vector ( dtype *v, MPI_Datatype dtype, int n, MPI_Comm comm)
{	int my_rank;
	MPI_Comm_rank (comm, &my_rank);
	if (!my_rank)
	{	print_subvector (v, dtype, n);
		printf ("\n\n");
	}
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
	{	m=n=6;							// no i/p for an process except 0
		nb=6;
	}
	MPI_Bcast (&m, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	MPI_Bcast (&n, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	MPI_Bcast (&nb, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	size_of_type = get_size(MPI_INT);
	local_cols = BLOCK_SIZE(my_rank,p,n);
	local_a = my_malloc (my_rank, local_cols * m * size_of_type);
	b = my_malloc (my_rank, n * size_of_type);
	a = my_malloc (my_rank, m * PTR_SIZE);				// a will store pointer to a pointer type!!
	rptr = (dtype *)local_a;
	a[0]= rptr;
	for (i = 1; i < m; i++)
	{	rptr += (local_cols);					//size_of_type;		// go to next row
		a[i]= (dtype *) (rptr);
	}
	read_col_striped_matrix ( a, MPI_TYPE, m, n, MPI_COMM_WORLD, local_cols);
	print_col_striped_matrix (a, MPI_TYPE, m, n, MPI_COMM_WORLD);
	read_replicated_vector (b, MPI_TYPE, nb, MPI_COMM_WORLD);
	print_replicated_vector (b, MPI_TYPE, nb, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	/* Each process multiplies its columns of 'a' and vector 'b', resulting in a partial sum of product 'c'. */
	c_part_out = (dtype *) my_malloc (my_rank, n * sizeof(dtype));
	MPI_Barrier (MPI_COMM_WORLD);
	/*printf("local_matrix\n");
	for ( i=0; i<m; i++ )
	{	for( j=0; j<local_cols; j++)
			printf("%d  ", a[i][j]);
		printf("\n");
	}
	printf("local_vector\n");
	for( i=0; i<n;i++ )
		printf("%d  ", b[i]);
	printf("\n");*/
	seconds = -MPI_Wtime();
	for (i = 0; i < m; i++)
	{	c_part_out[i] = 0.0;
		int l;
		for (l=0, j = BLOCK_LOW(my_rank,p,n); j <= BLOCK_HIGH(my_rank,p,n); j++, l++)
			c_part_out[i] += a[i][l] * b[j];
	// stores c[i][j] partial. store in a single element as they will eventually be added!! This will allow it to be alltoallexchange()
	}
	c = (dtype*) my_malloc (my_rank, m * sizeof(dtype));
	MPI_Allreduce (c_part_out, c, m, MPI_TYPE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Barrier (MPI_COMM_WORLD);
	seconds += MPI_Wtime();
	MPI_Allreduce (&seconds, &max_seconds, 1, MPI_TYPE, MPI_MAX, MPI_COMM_WORLD);
	if (!my_rank)
	{	print_replicated_vector ((dtype *) c, MPI_TYPE, m, MPI_COMM_WORLD);
		printf ("MV3) N = %d, Processes = %d, Time = %12.6f sec,", n, p, max_seconds);
		printf ("Mflop = %6.20lf\n", (double)2*n*n/((double)1000000.0*max_seconds));
	}
	MPI_Finalize();
	return 0;
}

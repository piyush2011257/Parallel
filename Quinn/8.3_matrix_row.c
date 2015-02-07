#include <stdio.h>
#include <mpi.h>
#include<stdlib.h>

/* Change these two definitions when the matrix and vector element types change */
typedef double dtype;
#define MPI_TYPE MPI_DOUBLE
#define MALLOC_ERROR       -2
#define TYPE_ERROR         -3
#define BLOCK_OWNER(j,p,n) (((p)*((j)+1)-1)/(n))
#define MIN(a,b)           ((a)<(b)?(a):(b))
#define BLOCK_LOW(id,p,n)  ((id)*(n)/(p))			// low_value= (in/p)
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)		// high value (i+1)n/p-1
#define BLOCK_SIZE(id,p,n) \
                     (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)	// size of the interval
#define PTR_SIZE           (sizeof(void*))			// size of a pointer. remeber sizeof(int *) = sizeof(float *) = sizeof(void *) = .. (all pointer types have same size)

// determine size of the data type. used in malloc for memory allocations
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

// end process due to error
void terminate ( int id, char *error_message) 		/* IN - Message to print */
{	if (!id)
	{	printf ("Error: %s\n", error_message);
		fflush (stdout);
	}
	MPI_Finalize();					// finish off cleaning process
	exit (-1);
}

// allocate bytes size of memory
void *my_malloc ( int id, int bytes)
{	void *buffer;
	if ((buffer = malloc ((size_t) bytes)) == NULL)
	{	printf ("Error: Malloc failed for process %d\n", id);
		fflush (stdout);
		MPI_Abort (MPI_COMM_WORLD, MALLOC_ERROR);
	}
	return buffer;
}

// read matrix row wise. memory already allocated to each local_a and a
void read_row_striped_matrix ( dtype **subs, dtype *storage, MPI_Datatype dtype, int m, int n, MPI_Comm comm, int local_rows)
{	int i, i1, my_rank, p;
	int *rptr;
	MPI_Status status;
	MPI_Comm_size (comm, &p);
	MPI_Comm_rank (comm, &my_rank);
	if (my_rank == (p-1))					// read and send done by MPI_Row()
	{	for (i = 0; i < p-1; i++)
		{	for ( i1=0; i1 < BLOCK_SIZE(i,p,m) * n; i1++)
			{	//scanf("%d",&storage[i1]);		// no i/p for any process except for 0
				storage[i1]=(my_rank+i1)+i1;		// allocate value to local matrix
			}
			MPI_Send (storage, BLOCK_SIZE(i,p,m) * n, dtype, i, i, comm);		// send value to respective process in the order of i(0->p-2). all oder is preserved using my_rank as tags
		}
		for ( i1=0; i1 < BLOCK_SIZE(i,p,m) * n; i1++)
			storage[i1]=my_rank+i1;//scanf("%d",&storage[i1]);		// no i/p for any process except for 0
	}
	else
		MPI_Recv (storage, local_rows * n, dtype, p-1, my_rank, comm, &status);
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

void print_row_striped_matrix ( dtype **a, MPI_Datatype dtype, int m, int n, MPI_Comm comm)
{	MPI_Status status;
	double *bstorage;
	double **b;
	int size_of_type, i, my_rank, local_rows, p, max_block_size;
	MPI_Comm_rank (comm, &my_rank);
	MPI_Comm_size (comm, &p);
	local_rows = BLOCK_SIZE(my_rank,p,m);
	if (!my_rank)
	{	//printf("rank:%d a:%p local_rows:%d n: %d\n", my_rank, a, local_rows, n);
		print_submatrix (a, dtype, local_rows, n);
		if (p > 1)
		{	size_of_type = get_size (dtype);
			local_rows = BLOCK_SIZE(p-1,p,m);
			//printf("size: %d local_rows: %d\n", size_of_type, local_rows);
			bstorage = (double *)my_malloc (my_rank, local_rows * n * size_of_type);
			b = (double **) my_malloc (my_rank, local_rows * size_of_type);
			b[0] = bstorage;	
			for (i = 1; i < local_rows; i++)
				b[i] = b[i-1] + n;
			for (i = 1; i < p; i++)
			{	MPI_Recv (bstorage, BLOCK_SIZE(i,p,m)*n, dtype, i, i, MPI_COMM_WORLD, &status);
				print_submatrix (b, dtype, BLOCK_SIZE(i,p,m), n);
			}
			free (b);
			free (bstorage);
		}
		putchar ('\n');
	}
	else
		MPI_Send (*a, local_rows * n, dtype, 0, my_rank, MPI_COMM_WORLD);	// understand this *a == a[0]
}

void read_block_vector ( dtype *v, MPI_Datatype dtype, int n, MPI_Comm comm)
{	int i, j;
	MPI_Status status;       /* Result of receive */
	int my_rank, p;
	MPI_Comm_size(comm, &p);
	MPI_Comm_rank(comm, &my_rank);
	// Process p-1 opens file, determines number of vector elements, and broadcasts this value to the other processes.
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
	MPI_Status status;
	MPI_Comm_size (comm, &p);
	MPI_Comm_rank (comm, &my_rank);
	int tmp[BLOCK_SIZE(p-1,p,n)];
	if (!my_rank)
	{	print_subvector (v, dtype, BLOCK_SIZE(my_rank,p,n));
		if (p > 1)
		{	for (i = 1; i < p; i++)
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

/* This function creates the count and displacement arrays needed by scatter and gather functions, when the number of elements send/received to/from other processes varies.
*/

void create_mixed_xfer_arrays ( int id, int p, int n, int *count,  /* OUT - Array of counts */ int *disp)   /* OUT - Array of displacements */
{	int i;
	// order in the process number. coun = no. of elements being sent, disp is the local offset of ith process's data
	count[0] = BLOCK_SIZE(0,p,n);
	disp[0] = 0;
	for (i = 1; i < p; i++)
	{	disp[i] = disp[i-1] + count[i-1];
		count[i] = BLOCK_SIZE(i,p,n);
	}
}

// replicate_block_vector (c_block, n, (void *) c, MPI_TYPE, MPI_COMM_WORLD)
void replicate_block_vector ( dtype *ablock, int n, dtype *arep, MPI_Datatype dtype, MPI_Comm comm)
{	int my_rank, p;
	MPI_Comm_size (comm, &p);
	MPI_Comm_rank (comm, &my_rank);
	int *cnt = my_malloc (my_rank, p * sizeof(int));
	int *disp = my_malloc (my_rank, p * sizeof(int));
	create_mixed_xfer_arrays (my_rank, p, n, cnt, disp);				// deciding the order and initializing the 2 arrays
	MPI_Allgatherv (ablock, cnt[my_rank], dtype, arep, cnt, disp, dtype, comm);
	free (cnt);
	free (disp);
}

int main (int argc, char *argv[])
{	dtype **a;						        // matrix 
	dtype *b, *local_a;					        // vector 
	dtype *c_block;  						// Partial product vector 
	dtype *c;
	double max_seconds, seconds;
	int i, j, my_rank, m, n, p, local_rows, its, nb, size_of_type;   // nb-Elements in vector
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	
	// part of read_row_stripped()
	if ( my_rank == (p-1))
	{	m=n=6;							// no i/p for an process except 0
		nb= 6;							// nb must be equal to n
	}
	MPI_Bcast (&m, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	MPI_Bcast (&n, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	MPI_Bcast (&nb, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	local_rows = BLOCK_SIZE(my_rank,p,m);
	int local_rows_vector = BLOCK_SIZE(my_rank,p,n);
	local_a = (dtype *) my_malloc (my_rank, local_rows * n * sizeof(dtype));
	a = (dtype **) my_malloc (my_rank, local_rows * PTR_SIZE);
	b = (dtype *) my_malloc (my_rank, local_rows_vector * sizeof(dtype));
	dtype *rptr = local_a;
	a[0]= rptr;
	for (i = 1; i < local_rows; i++)
	{	rptr += (n);						//size_of_type, go to next row
		a[i]= (dtype *) (rptr);
	}

	read_row_striped_matrix ( a, local_a, MPI_TYPE, m, n, MPI_COMM_WORLD, local_rows);
	print_row_striped_matrix ( a, MPI_TYPE, m, n, MPI_COMM_WORLD);

	read_block_vector (b, MPI_TYPE, nb, MPI_COMM_WORLD);
	print_block_vector (b, MPI_TYPE, nb, MPI_COMM_WORLD);
	
	c_block = (dtype *) malloc (local_rows * sizeof(dtype));
	c = (dtype *) malloc (m * sizeof(dtype));
	dtype *global_b= (dtype *) malloc ( n * sizeof(dtype));
	replicate_block_vector (b, n, global_b, MPI_TYPE, MPI_COMM_WORLD);
	MPI_Barrier (MPI_COMM_WORLD);
	seconds = - MPI_Wtime();
	/*printf("local_matrix\n");
	for (i = 0; i < local_rows; i++)
	{	for ( j=0; j <n; j++ )
			printf("%lf  ", a[i][j]);
		printf("\n");
	}
	printf("vector\n ");
	for ( i=0; i<n; i++ )
		printf("%lf  ", global_b[i]);
	printf("\n");*/
	for (i = 0; i < local_rows; i++)
	{	c_block[i] = 0.0;
		for (j = 0; j < n; j++)
			c_block[i] += a[i][j] * global_b[j];
	}
	
	MPI_Barrier (MPI_COMM_WORLD);
	seconds += MPI_Wtime();
	print_block_vector (c_block, MPI_TYPE, m, MPI_COMM_WORLD);
	MPI_Allreduce (&seconds, &max_seconds, 1, MPI_TYPE, MPI_MAX, MPI_COMM_WORLD);
	if (!my_rank)
	{	printf ("MV1) N = %d, Processes = %d, Time = %12.6f sec,", n, p, max_seconds);
		printf ("Mflop = %6.2f\n", 2*n*n/(1000000.0*max_seconds));
	}
	MPI_Finalize();
	return 0;
}

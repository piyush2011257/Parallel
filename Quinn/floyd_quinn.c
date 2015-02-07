#include <stdio.h>
#include <mpi.h>
#include<stdlib.h>

typedef int dtype;			// int as dtype
#define MPI_TYPE MPI_INT
#define OPEN_FILE_ERROR    -1
#define MALLOC_ERROR       -2
#define TYPE_ERROR         -3
#define DATA_MSG           0
#define PROMPT_MSG         1
#define RESPONSE_MSG       2
#define BLOCK_OWNER(j,p,n) (((p)*((j)+1)-1)/(n))
#define MIN(a,b)           ((a)<(b)?(a):(b))
#define BLOCK_LOW(id,p,n)  ((id)*(n)/(p))			// low_value= (in/p)
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)		// high value (i+1)n/p-1
#define BLOCK_SIZE(id,p,n) \
                     (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)	// size of the interval
#define PTR_SIZE           (sizeof(void*))			// size of a pointer. remeber sizeof(int *) = sizeof(float *) = sizeof(void *) = .. (all pointer types have same size)

int get_size (MPI_Datatype t)
{	if (t == MPI_BYTE)
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

void terminate ( int id, char *error_message) /* IN - Message to print */
{	if (!id)
	{	printf ("Error: %s\n", error_message);
		fflush (stdout);
	}
	MPI_Finalize();
	exit (-1);
}

void *my_malloc ( int id, int bytes)  /* IN - Bytes to allocate */
{	void *buffer;
	if ((buffer = malloc ((size_t) bytes)) == NULL)
	{	printf ("Error: Malloc failed for process %d\n", id);
		fflush (stdout);
		MPI_Abort (MPI_COMM_WORLD, MALLOC_ERROR);
	}
	return buffer;
}

void read_row_striped_matrix ( char *s,/* IN - File name */ void ***subs, /* OUT - 2D submatrix indices */ void **storage,  /* OUT - Submatrix stored here */ MPI_Datatype dtype, int *m, /* OUT - Matrix rows */ int *n, /* OUT - Matrix cols */ MPI_Comm comm)
{	int datum_size;			/* Size of matrix element */
	int i, my_rank, p;
	FILE *infileptr;		/* Input file pointer */
	int local_rows;			/* Rows on this process */
	void **lptr;			/* Pointer into 'subs' */
	void *rptr;			/* Pointer into 'storage' */
	MPI_Status status;		/* Result of receive */
	int x;				/* Result of read */
	MPI_Comm_size (comm, &p);
	MPI_Comm_rank (comm, &my_rank);
	datum_size = get_size (dtype);
	/* Process p-1 opens file, reads size of matrix, and broadcasts matrix dimensions to other procs */
	if ( my_rank == (p-1))
	{	infileptr = fopen (s, "r");			// open the file to read
		if (infileptr == NULL)
			*m = 0;					// no element -> no rows
		else
		{	fread (m, sizeof(int), 1, infileptr);		// file starts with no. of rows and columns
			fread (n, sizeof(int), 1, infileptr);
		}
	}
	MPI_Bcast (m, 1, MPI_INT, p-1, comm);				// b_cast no. of rows to all process
	if (!(*m))
		MPI_Abort (MPI_COMM_WORLD, OPEN_FILE_ERROR);		// in case matrix was empty
	MPI_Bcast (n, 1, MPI_INT, p-1, comm);				// b_cast no. fo columns in each row
	local_rows = BLOCK_SIZE(my_rank,p,*m);				// total no. of rows in the process
	/* Dynamically allocate matrix. Allow double subscripting through 'a'. */
	// for local matrix of size ( local_row * n ). access as a linear array of contigous elements
	*storage = (void *) my_malloc (my_rank, local_rows * (*n) * datum_size);	// == malloc ( (local_m*n) * sizeof(matrix_type) )
	// subs stores original 2d matrix of size - (local_row* n) here we just create rows no column till now
	*subs = (void **) my_malloc (my_rank, local_rows * PTR_SIZE);		// sub is a pointer to a 2-d matrix
	// allocate memory for rows for a 2-d matrix. Memory not allocated to the rows yet. sizeof each row = 0 still
	lptr = (void *) &(*subs[0]);						// lptr-> pointer to start of subs
	rptr = (void *) *storage;						// rptr-> pointer to local matrix starting
	for (i = 0; i < local_rows; i++)
	{	*(lptr++)= (void *) rptr;
		// copy the memory allocated to local_matrix to the global matrix
		rptr += *n * datum_size;		// go to next row
	}
	// at this point each process ahs allocated memory for local matrix and a refernce to global matrix
	/* Process p-1 reads blocks of rows from file and sends each block to the correct destination process. The last block it keeps. */
	if (my_rank == (p-1))
	{	for (i = 0; i < p-1; i++)
		{	// store local_row * n contiguous elemenst of size (of data type)-> datum size
			x = fread (*storage, datum_size, BLOCK_SIZE(i,p,*m) * *n, infileptr);
			MPI_Send (*storage, BLOCK_SIZE(i,p,*m) * *n, dtype, i, DATA_MSG, comm);		// send it to corresponding process
		}
		x = fread (*storage, datum_size, local_rows * *n, infileptr);
		fclose (infileptr);
	}
	else
		MPI_Recv (*storage, local_rows * *n, dtype, p-1, DATA_MSG, comm, &status);
}

void print_submatrix ( void       **a,       /* OUT - Doubly-subscripted array */ MPI_Datatype dtype,   /* OUT - Type of array elements */ int          rows,    /* OUT - Matrix rows */ int          cols)    /* OUT - Matrix cols */
{	int i, j;
	for (i = 0; i < rows; i++)
	{	for (j = 0; j < cols; j++)
		{	if (dtype == MPI_DOUBLE)
				printf ("%6.3f ", ((double **)a)[i][j]);
			else
			{	if (dtype == MPI_FLOAT)
					printf ("%6.3f ", ((float **)a)[i][j]);
				else if (dtype == MPI_INT)
					printf ("%6d ", ((int **)a)[i][j]);
			}
		}
		putchar ('\n');
	}
}

void print_row_striped_matrix ( void **a, /* IN - 2D array */ MPI_Datatype dtype,  /* IN - Matrix element type */ int m, /* IN - Matrix rows */
 int n, /* IN - Matrix cols */ MPI_Comm comm)       /* IN - Communicator */
{	MPI_Status  status;          /* Result of receive */
	void *bstorage;        /* Elements received from another process */
	void **b;               /* 2D array indexing into 'bstorage' */
	int datum_size;      /* Bytes per element */
	int  i, my_rank, local_rows, p;
	int max_block_size;  /* Most matrix rows held by any process */
	int prompt;          /* Dummy variable */
	MPI_Comm_rank (comm, &my_rank);
	MPI_Comm_size (comm, &p);
	local_rows = BLOCK_SIZE(my_rank,p,m);
	if (!my_rank)
	{	print_submatrix (a, dtype, local_rows, n);
		if (p > 1)
		{	datum_size = get_size (dtype);
			max_block_size = BLOCK_SIZE(p-1,p,m);
			bstorage = my_malloc (my_rank, max_block_size * n * datum_size);
			b = (void **) my_malloc (my_rank, max_block_size * datum_size);
			b[0] = bstorage;	
			for (i = 1; i < max_block_size; i++)
				b[i] = b[i-1] + n * datum_size;
			for (i = 1; i < p; i++)
			{	MPI_Send (&prompt, 1, MPI_INT, i, PROMPT_MSG, MPI_COMM_WORLD);
				MPI_Recv (bstorage, BLOCK_SIZE(i,p,m)*n, dtype, i, RESPONSE_MSG, MPI_COMM_WORLD, &status);
				print_submatrix (b, dtype, BLOCK_SIZE(i,p,m), n);
			}
			free (b);
			free (bstorage);
		}
		putchar ('\n');
	}
	else
	{	MPI_Recv (&prompt, 1, MPI_INT, 0, PROMPT_MSG, MPI_COMM_WORLD, &status);
		MPI_Send (*a, local_rows * n, dtype, 0, RESPONSE_MSG, MPI_COMM_WORLD);
	}
}

void compute_shortest_paths (int my_rank, int p, dtype **a, int n)
{	int i, j, k, offset, root;
	/* offset -Local index of broadcast row, root - Process controlling row to be bcast. It is very useful in telling the starting row and column of a process in the global array. Rest can be determined using the size occupied by the process of the matrix */
	int *tmp;     					/* Holds the broadcast row */
	tmp = (dtype *) malloc (n * sizeof(dtype));	// linear array- row
	for (k = 0; k < n; k++)
	{	root = BLOCK_OWNER(k,p,n);
		if (root == my_rank)
		{	offset = k - BLOCK_LOW(my_rank,p,n);
			for (j = 0; j < n; j++)
				tmp[j] = a[offset][j];
		}
		MPI_Bcast (tmp, n, MPI_TYPE, root, MPI_COMM_WORLD);
		for (i = 0; i < BLOCK_SIZE(my_rank,p,n); i++)
			for (j = 0; j < n; j++)
				a[i][j] = MIN(a[i][j],a[i][k]+tmp[j]);
	}
	free (tmp);
}

int main (int argc, char *argv[])
{	dtype** a;         			/* Doubly-subscripted array */
	dtype*  local_a;   			/* Local portion of array elements */
	int i, j, k, my_rank, m, n, p;         /* m-Rows in matrix, n- Columns in matrix */
	double  time, max_time;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	read_row_striped_matrix (argv[1], (void *) &a, (void *) &local_a, MPI_TYPE, &m, &n, MPI_COMM_WORLD);
	if (m != n)
		terminate (my_rank, "Matrix must be square\n");
	/*
	print_row_striped_matrix ((void **) a, MPI_TYPE, m, n, MPI_COMM_WORLD);
	*/
	MPI_Barrier (MPI_COMM_WORLD);
	time = -MPI_Wtime();
	compute_shortest_paths (my_rank, p, (dtype **) a, n);
	time += MPI_Wtime();
	MPI_Reduce (&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (!my_rank)
		printf ("Floyd, matrix size %d, %d processes: %6.2f seconds\n", n, p, max_time);
	/*
	print_row_striped_matrix ((void **) a, MPI_TYPE, m, n, MPI_COMM_WORLD);
	*/
	MPI_Finalize();
}

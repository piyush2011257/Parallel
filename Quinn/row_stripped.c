// Refer to appendix as well for detaled explanation
// for generic use void* -> int* and file handling instead of direct allocation
#include <stdio.h>
#include <mpi.h>
#include<stdlib.h>

typedef int dtype;			// int as dtype
#define MPI_TYPE MPI_INT
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
void read_row_striped_matrix ( char *s, int **subs, int *storage, MPI_Datatype dtype, int m, int n, MPI_Comm comm, int local_rows)
{	int i, my_rank, p;
	int *rptr;			/* Pointer into 'storage'. void * can point to any type */
	MPI_Status status;
	MPI_Comm_size (comm, &p);
	MPI_Comm_rank (comm, &my_rank);
	if (my_rank == (p-1))
	{	for (i = 0; i < p-1; i++)
		{	int i1;
			for ( i1=0; i1 < BLOCK_SIZE(i,p,m) * n; i1++)
			{	//scanf("%d",&storage[i1]);		// no i/p for any process except for 0
				storage[i1]=(my_rank+i1)+i1;		// allocate value to local matrix
			}
			MPI_Send (storage, BLOCK_SIZE(i,p,m) * n, dtype, i, i, comm);		// send value to respective process in the order of i(0->p-2). all oder is preserved using my_rank as tags
		}
		int i1;
		for ( i1=0; i1 < BLOCK_SIZE(i,p,m) * n; i1++)
			storage[i1]=my_rank+i1;//scanf("%d",&storage[i1]);		// no i/p for any process except for 0
	}
	else
		MPI_Recv (storage, local_rows * n, dtype, p-1, my_rank, comm, &status);
}

void print_submatrix ( int **a, MPI_Datatype dtype, int rows, int cols)
{	int i, j;
	for (i = 0; i < rows; i++)
	{	for (j = 0; j < cols; j++)
		{	if (dtype == MPI_DOUBLE)
				printf ("%6.3f ", ((double **)a)[i][j]);
			else
			{	if (dtype == MPI_FLOAT)
					printf ("%6.3f ", ((float **)a)[i][j]);
				else if (dtype == MPI_INT)
				{	//printf("\n%d %d\n",i,j);
					printf ("%6d", ((int **)a)[i][j]);
				}
			}
		}
		putchar ('\n');
	}
}

void print_row_striped_matrix ( int **a, MPI_Datatype dtype, int m, int n, MPI_Comm comm)
{	MPI_Status status;
	int *bstorage;
	int **b;
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
			bstorage = (int *)my_malloc (my_rank, local_rows * n * size_of_type);
			b = (int **) my_malloc (my_rank, local_rows * size_of_type);
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

void compute_shortest_paths (int my_rank, int p, dtype **a, int n)
{	int i, j, k, offset, root;
	int *tmp;     					/* Holds the broadcast row */
	tmp = (dtype *) malloc (n * sizeof(dtype));	// linear array- row
	for (k = 0; k < n; k++)
	{	root = BLOCK_OWNER(k,p,n);		// n == total no. of rows = m !!!
		// at kth step kth row is to be broadcasted. root-> rankl of process having kth row
		//printf("my_rank: %d  k: %d  root: %d  p: %d  n: %d\n", my_rank,k,root,p,n);
		if (root == my_rank)
		{	offset = k - BLOCK_LOW(my_rank,p,n);		// offset. BLOCK_LOW()-> gives the index of the row starting in this process in the global array
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
	int i, j, k, my_rank, m, n, p, size_of_type;
	double  time, max_time;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	//printf("%d %d\n",p,my_rank);
	if ( my_rank == (p-1))
		m=n=4;//scanf("%d %d", &m, &n);				// no i/p for an process except 0
	MPI_Bcast (&m, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	MPI_Bcast (&n, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	//printf("m:%d n:%d\n",m,n);
	size_of_type = get_size(MPI_INT);
	int local_rows = BLOCK_SIZE(my_rank,p,m);
	//printf("%d %d %d\n", local_rows, my_rank, size_of_type);
	local_a = (int *) my_malloc (my_rank, local_rows * n * size_of_type);
	/*for ( i=0; i< n * local_rows; i++ )
		printf("my_rank %d  local_a %p ", my_rank,local_a+i);
	printf("\n");*/
	a = (int **) my_malloc (my_rank, local_rows * PTR_SIZE);
	/*printf("a: %d\n", my_rank);
	for ( i=0; i< local_rows; i++ )
		printf("myrank %d a %d ", my_rank, a+i);
	printf("\n");*/
	int *rptr = local_a;
	a[0]= rptr;
	//printf("my_rank: %d %d %d\n", my_rank, rptr, subs[0]);
	for (i = 1; i < local_rows; i++)
	{	rptr += (n);			//size_of_type;		// go to next row
		a[i]= (int *) (rptr);
		//printf("my_rank:%d %d %d", my_rank, rptr,subs[i]);
	}
	read_row_striped_matrix (argv[1], (int **) a, (int *) local_a, MPI_TYPE, m, n, MPI_COMM_WORLD, local_rows);
	MPI_Barrier(MPI_COMM_WORLD);
	/*for ( i=0; i<2; i++)
	{	for ( j=0; j<4; j++ )
			printf("%d ", *(a[i]+j));
		printf("\n");
	}*/
	if (m != n)
		terminate (my_rank, (char *)"Matrix must be square\n");
	//printf("%d %p\n",my_rank,a);
	print_row_striped_matrix ((int **) a, MPI_TYPE, m, n, MPI_COMM_WORLD);
	MPI_Barrier (MPI_COMM_WORLD);
	time = -MPI_Wtime();
	compute_shortest_paths (my_rank, p, (dtype **) a, n);
	time += MPI_Wtime();
	MPI_Reduce (&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (!my_rank)
	{	printf ("Floyd, matrix size %d, %d processes: %6.2f seconds\n", n, p, max_time);
		printf("Shortest path matrix\n");
	}
	print_row_striped_matrix ((int **) a, MPI_TYPE, m, n, MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}

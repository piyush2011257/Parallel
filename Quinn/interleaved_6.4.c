#include <stdio.h>
#include <mpi.h>
#include<stdlib.h>

typedef int dtype;			// int as dtype
#define MPI_TYPE MPI_INT
#define MALLOC_ERROR       -2
#define TYPE_ERROR         -3
#define BLOCK_OWNER(j,p,n) j%p
#define MIN(a,b)           ((a)<(b)?(a):(b))
#define BLOCK_SIZE(id,p,n) ( ( n/p ) + ( (n%p) >= ( id+1 ) ) )
#define PTR_SIZE (sizeof(void*))

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

void terminate ( int id, char *error_message)
{	if (!id)
	{	printf ("Error: %s\n", error_message);
		fflush (stdout);
	}
	MPI_Finalize();
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

void read_row_striped_matrix ( int *storage, MPI_Datatype dtype, int m, int n, MPI_Comm comm, int local_rows)
{	int i, my_rank, p, i1;
	int *rptr;
	MPI_Status status;
	MPI_Comm_size (comm, &p);
	MPI_Comm_rank (comm, &my_rank);
	MPI_Barrier(comm);
	if ( my_rank == (p-1) )
	{	for (i = 0; i < m; i++)
		{	for ( i1=0; i1 < n; i1++)
				storage[i1]=(my_rank+i1)+i1;
			//printf("send %d\n", i%p);
			if ( i % p != p-1 )
				MPI_Send (storage, n, dtype, i%p, i%p, comm);
			else
			{	//printf("receive 1\n");
				storage += n;
			}
		}
	}
	else
	{	for ( i=0; i< local_rows; i++ )
		{	MPI_Recv (storage, n, dtype, p-1, my_rank, comm, &status);
			//printf("recieve %d\n", my_rank);
			storage += n;
		}
	}
}

void print_row_striped_matrix ( int **a, MPI_Datatype dtype, int m, int n, MPI_Comm comm)
{	MPI_Status status;
	int size_of_type, i, my_rank, local_rows, p, max_block_size, j, l;
	MPI_Comm_rank (comm, &my_rank);
	MPI_Comm_size (comm, &p);
	local_rows = BLOCK_SIZE(my_rank,p,m);
	int *bstorage= my_malloc( my_rank, sizeof(int)*n);
	int ** rptr= a;
	MPI_Barrier(comm);
	i=0;
	if (!my_rank)
	{	rptr=a;
		for ( j=0; i<n; i++, j = (j+1)%p )
		{	if ( j == 0 )
			{	for ( l=0; l<n; l++ )
					printf("%d  ", rptr[0][l]);
				printf("\n");
				rptr++;
				continue;
			}
			//printf("receive %d i: %d n :%d\n",j, i, n);
			MPI_Recv (bstorage, n, dtype, j, j, comm, &status);
			for ( l=0; l<n; l++ )
				printf("%d  ", bstorage[l]);
			printf("\n");
		}
	}
	else
	{	rptr=a;
		for ( i=0; i<local_rows; i++ )
		{	//printf("sent %d local_row: %d n:%d\n", my_rank, local_rows, n);
			MPI_Send (*rptr, n, dtype, 0, my_rank, comm);	// understand this *a == a[0]
			rptr++;
		}
	}
}

void compute_shortest_paths (int my_rank, int p, dtype **a, int n)
{	int i, j, k, offset, root;
	int *tmp;     					/* Holds the broadcast row */
	tmp = (dtype *) malloc (n * sizeof(dtype));	// linear array- row
	for (k = 0; k < n; k++)
	{	root = BLOCK_OWNER(k,p,n);		// n == total no. of rows = m !!!
		MPI_Barrier(MPI_COMM_WORLD);
		if (root == my_rank)
		{	offset = k/p;
			for (j = 0; j < n; j++)
				tmp[j] = a[offset][j];
		}
		MPI_Bcast (tmp, n, MPI_INT, root, MPI_COMM_WORLD);
		for (i = 0; i < BLOCK_SIZE(my_rank,p,n); i++)
			for (j = 0; j < n; j++)
				a[i][j] = MIN(a[i][j],a[i][k]+tmp[j]);
	}
	free (tmp);
}

int main ( int argc, char *argv[] )
{	dtype** a;
	dtype*  local_a;
	int i, j, k, my_rank, m, n, p, size_of_type;
	double  time, max_time;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	if ( my_rank == (p-1))
		m=n=4;
	MPI_Bcast (&m, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	MPI_Bcast (&n, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	size_of_type = get_size(MPI_INT);
	int local_rows = BLOCK_SIZE(my_rank,p,m);
	local_a = (int *) my_malloc (my_rank, local_rows * n * size_of_type);
	a = (int **) my_malloc (my_rank, local_rows * PTR_SIZE);
	int *rptr = local_a;
	a[0]= rptr;
	for (i = 1; i < local_rows; i++)
	{	rptr += (n);
		a[i]= (int *) (rptr);
	}
	/*printf("rank: %d local_rows:%d\n", my_rank, local_rows);
	for ( i=0; i<local_rows; i++)
	{	for ( j=0; j<n; j++ )
			printf("%p   ", a[i]+j);
		printf("\n");
	}*/
	//int **ta=a;
	/*for ( i=0; i<local_rows; i++ )
	{	printf("%p %p\n", ta, ta[0]);
		ta++;
	}
	printf("\n");*/	
	read_row_striped_matrix ( (int *) local_a, MPI_TYPE, m, n, MPI_COMM_WORLD, local_rows);
	MPI_Barrier(MPI_COMM_WORLD);
	if (m != n)
		terminate (my_rank, (char *)"Matrix must be square\n");
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

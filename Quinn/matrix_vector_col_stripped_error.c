#include <stdio.h>
#include<stdlib.h>
#include <mpi.h>

typedef double dtype;
#define TYPE_ERROR -3
#define MALLOC_ERROR -2
#define MPI_TYPE MPI_DOUBLE
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
	//printf("my_malloc: %p\n", buffer);
	return buffer;
}

/* This function creates the count and displacement arrays needed by scatter and gather functions, when the number of elements send/received to/from other processes varies.
*/

void create_mixed_xfer_arrays ( int id, int p, int n, int *count,  /* OUT - Array of counts */ int *disp)   /* OUT - Array of displacements */
{	//printf("enter send count rank: %d\n", id);
	int i;
	// order in the process number. coun = no. of elements being sent, disp is the local offset of ith process's data
	count[0] = BLOCK_SIZE(0,p,n);
	disp[0] = 0;
	for (i = 1; i < p; i++)
	{	disp[i] = disp[i-1] + count[i-1];
		count[i] = BLOCK_SIZE(i,p,n);
	}
	//for ( i=0; i<p; i++ )
	//	printf("rank: %d  disp: %p  disp[%d]: %d  count: %p  count[%d]: %d\n", id, disp+i, i, disp[i], count+i, i, count[i]);
}

void print_submatrix ( dtype *a[], MPI_Datatype dtype, int rows, int cols)
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

void print_col_striped_matrix ( dtype *a[], MPI_Datatype dtype, int m, int n, MPI_Comm comm)
{	MPI_Status status;
	double bstorage[n];
	int size_of_type, i, my_rank, local_cols, p, max_block_size, j;
	MPI_Comm_rank (comm, &my_rank);
	MPI_Comm_size (comm, &p);
	local_cols = BLOCK_SIZE(my_rank,p,m);
	//bstorage= my_malloc( my_rank, n * sizeof(dtype));
	for ( i=0; i<m; i++ )
	{	MPI_Gather(a[i], local_cols, MPI_DOUBLE, bstorage, local_cols, MPI_DOUBLE, 0, comm);		// a[i] not a+i
		if ( my_rank == 0 )
		{	for ( j =0; j<n; j++ )
				printf("%lf  ", bstorage[j]);
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
	/* Process p-1 opens file, determines number of vector elements, and broadcasts this value to the other processes. */
	if (my_rank == (p-1))
	{	for (i = 0; i < p-1; i++)
		{	for ( j=0; j < BLOCK_SIZE(i,p,n); j++ )
				v[i]= my_rank * my_rank;
			MPI_Send (v, BLOCK_SIZE(j,p,n), dtype, j, j, comm);
		}
		for ( j=0; j < BLOCK_SIZE(j,p,n); j++ )
			v[j]= my_rank * my_rank;
	}
	else
		MPI_Recv (v, BLOCK_SIZE(my_rank,p,n), dtype, p-1, my_rank, comm, &status);
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
	MPI_Status status;     /* Result of receive */
	double *tmp;
	MPI_Comm_size (comm, &p);
	MPI_Comm_rank (comm, &my_rank);
	if (!my_rank)
	{	print_subvector (v, dtype, BLOCK_SIZE(my_rank,p,n));
		if (p > 1)
		{	tmp = my_malloc (my_rank,BLOCK_SIZE(p-1,p,n)*get_size (dtype));
			// alternately we can use gatherv
			for (i = 1; i < p; i++)
			{	MPI_Recv (tmp, BLOCK_SIZE(i,p,n), dtype, i, i, comm, &status);
				print_subvector (tmp, dtype, BLOCK_SIZE(i,p,n));
			}
			free (tmp);
		}
		printf ("\n\n");
	}
	else
		MPI_Send (v, BLOCK_SIZE(my_rank,p,n), dtype, 0, my_rank, comm);
}

void read_col_striped_matrix ( dtype *subs[], MPI_Datatype dtype, int m, int n, MPI_Comm comm, int local_cols)
{	int i, my_rank, p,j;
	MPI_Status status;
	MPI_Comm_size (comm, &p);
	MPI_Comm_rank (comm, &my_rank);
	/*double *storage= malloc ((size_t) n * sizeof(dtype));		//my_malloc( my_rank, n * sizeof(dtype));
	int *send_count = malloc ((size_t) p * sizeof(int));		//my_malloc (my_rank, p * sizeof(int));
	int *send_disp = malloc ((size_t) p * sizeof(int));		//my_malloc (my_rank, p * sizeof(int));*/
	double storage[n];
	int send_count[p], send_disp[p];
	
	create_mixed_xfer_arrays (my_rank,p,n,send_count,send_disp);
	/*for ( i=0; i<p; i++)
		printf("cnt[%d]: %d  disp[%d]: %d\n", i, send_count[i], i, send_disp[i]);
	printf("\n");
	printf("m: %d n: %d rank: %d\n", m,n,my_rank);
	for ( i=0; i<m; i++ )
	{	for ( j=0; j< local_cols; j++ )
			printf("%p ", subs[i]+j);
		printf("\n");
	}
	printf("\n");
	for ( i=0; i<n; i++ )
		printf("rank: %d  storage: %p\n", my_rank, storage+i);
	printf("\n");*/
	for ( i=0; i<p; i++ )
		printf("rank: %d  disp: %p  disp[%d]: %d  count: %p  count[%d]: %d\n", my_rank, send_disp+i, i, send_disp[i], send_count+i, i, send_count[i]);
	printf("\n\n");
	MPI_Barrier(comm);
	{	int i1,j;
		for ( j=0; j< m; j++ )
		{	if ( my_rank == p-1 )
			{	for ( i1=0; i1 < n; i1++)
					storage[i1]=rand()%100;
				for ( i1=0; i1 < n; i1++)
					printf("%lf ", storage[i1]);
				printf("\n");
			}
			//create_mixed_xfer_arrays (my_rank,p,n,send_count,send_disp);
			/*for ( i=0; i< p; i++ )
				printf("rank: %d  disp: %p  disp[%d]: %d  count: %p  count[%d]: %d\n", my_rank, send_disp+i, i, send_disp[i], send_count+i, i, send_count[i]);
			/*for ( i1=0; i1 < n; i1++)
				printf("rank:%d  storage+i1:%p\n", my_rank, storage+i1);
			printf("\n");*/
			//printf("scattered %dth row by %d process\n", j, my_rank);
			MPI_Scatterv (storage, send_count, send_disp, MPI_DOUBLE, subs[j], local_cols, MPI_DOUBLE, p-1, comm);
			//MPI_Barrier(comm);
			printf("my_rank: %d\n", my_rank);
			{	for ( i1=0; i1 < local_cols; i1++)
					printf("%lf ", subs[j][i1]);
				printf("\n");
			}
			MPI_Barrier(comm);
			// we cant use local_a as it would lead to aliasing!! Keep care of aliasing!
		}
	/*	for ( i=0; i<p; i++ )
			printf("rank: %d  disp: %p  disp[%d]: %d  count: %p  count[%d]: %d\n", my_rank, send_disp+i, i, send_disp[i], send_count+i, i, send_count[i]);
		printf("\n\n");*/
		MPI_Barrier(comm);
	}
	int i1;
	for ( j=0; j<m; j++ )
	{	for ( i1=0; i1 < local_cols; i1++)
			printf("%p  ", subs[j]+i1);
		printf("\n");
	}
	//free (send_count);
	//free (send_disp);
}

int main (int argc, char *argv[])
{	//dtype **a;				/* matrix */
	//dtype *b;				/* vector */
	dtype *c;				/* The product, a vector */
	dtype  *c_part_out;			/* Partial sums, sent */
	dtype  *c_part_in;			/* Partial sums, received */
	int *cnt_out;				/* Elements sent to each proc */
	int *cnt_in;				/* Elements received per proc */
	int *disp_out;				/* Indices of sent elements */
	int *disp_in;				/* Indices of received elements */
	int i, j, my_rank, m, n, p, nb, local_cols, size_of_type;
	double max_seconds, seconds;
	dtype *rptr;			/* This process's portion of 'a' */
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	
	if ( my_rank == (p-1))
	{	m=n=3;							// no i/p for an process except 0
		nb=3;
	}
	MPI_Bcast (&m, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	MPI_Bcast (&n, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	MPI_Bcast (&nb, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	size_of_type = get_size(MPI_INT);
	local_cols = nb = BLOCK_SIZE(my_rank,p,n);
	//local_a = (dtype *) malloc ( (size_t) local_cols * m * size_of_type);		//my_malloc (my_rank, local_cols * m * size_of_type);
	//b = (dtype *) malloc ( (size_t) local_cols * size_of_type);			//my_malloc (my_rank, local_cols * size_of_type);
	//a = (dtype **) malloc ( (size_t) m * PTR_SIZE);			//my_malloc (my_rank, m * PTR_SIZE);		// a will store pointer to a pointer type!!
	dtype local_a[m * local_cols];
	dtype *l= &local_a[0];
	dtype b[n];
	dtype *a[m];
	for (i=0; i<m; i++ )
	{	a[i]= l;
		printf("%p\n", a[i]);
		l += local_cols;
	}
	/*rptr = (dtype *)local_a;
	a[0]= rptr;
	for (i = 1; i < m; i++)
	{	rptr += (local_cols);					//size_of_type;		// go to next row
		a[i]= (dtype *) (rptr);
	}*/
	read_col_striped_matrix ( a, MPI_TYPE, m, n, MPI_COMM_WORLD, local_cols);
	for (i=0; i<m; i++ )
	{	for ( j=0; j<local_cols; j++ )
			printf("%lf  ", *(a[i]+j));
		printf("\n");
	}
	print_col_striped_matrix (a, MPI_TYPE, m, n, MPI_COMM_WORLD);

	return 0;
	read_block_vector ((dtype *) b, MPI_TYPE, nb, MPI_COMM_WORLD);
	print_block_vector ((dtype *) b, MPI_TYPE, nb, MPI_COMM_WORLD);
	/* Each process multiplies its columns of 'a' and vector 'b', resulting in a partial sum of product 'c'. */
	c_part_out = (dtype *) my_malloc (my_rank, n * sizeof(dtype));
	MPI_Barrier (MPI_COMM_WORLD);
	seconds = -MPI_Wtime();
	for (i = 0; i < n; i++)
	{	c_part_out[i] = 0.0;
		for (j = 0; j < local_cols; j++)
			c_part_out[i] += a[i][j] * b[j];
	}
	/*
	create_mixed_xfer_arrays (id, p, n, &cnt_out, &disp_out);
	create_uniform_xfer_arrays (id, p, n, &cnt_in, &disp_in);
	c_part_in = (dtype*) my_malloc (id, p*local_els*sizeof(dtype));
	MPI_Alltoallv (c_part_out, cnt_out, disp_out, mpitype, c_part_in, cnt_in, disp_in, mpitype, MPI_COMM_WORLD);
	c = (dtype*) my_malloc (id, local_els * sizeof(dtype));
	for (i = 0; i < local_els; i++)
	{	c[i] = 0.0;
		for (j = 0; j < p; j++)
			c[i] += c_part_in[i + j*local_els];
	}
	MPI_Barrier (MPI_COMM_WORLD);
	seconds += MPI_Wtime();
	MPI_Allreduce (&seconds, &max_seconds, 1, mpitype, MPI_MAX, MPI_COMM_WORLD);
	if (!id)
	{	printf ("MV3) N = %d, Processes = %d, Time = %12.6f sec,", n, p, max_seconds);
		printf ("Mflop = %6.2f\n", 2*n*n/(1000000.0*max_seconds));
	}
	print_block_vector ((dtype *) c, mpitype, n, MPI_COMM_WORLD);*/
	MPI_Finalize();
	return 0;
}

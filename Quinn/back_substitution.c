#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

/* Change these two definitions when the matrix and vector element types change */
typedef double dtype;
#define MPI_TYPE MPI_DOUBLE
#define MAalloOC_ERROR       -2
#define TYPE_ERROR         -3
#define BLOCK_OWNER(j,p,n)	j%p
#define MIN(a,b)           ((a)<(b)?(a):(b))
#define BLOCK_SIZE(id,p,n)	n/p + ( id < n%p )
#define PTR_SIZE           (sizeof(void*))

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

void allocate_vector ( dtype **a, int r )
{	dtype *local_a = (void *)malloc(r*sizeof(dtype));
	(*a) = (void *)malloc(r*PTR_SIZE);
	dtype *l= &local_a[0];
	(*a)= l;
}

void read_row_interleaved_striped_matrix ( dtype ***a, MPI_Datatype dtype, int n, MPI_Comm comm)
{	int i, i1, j1, my_rank, p, idx=0;
	MPI_Status status;
	MPI_Comm_size (comm, &p);
	MPI_Comm_rank (comm, &my_rank);
	double *storage;
	if (my_rank == (p-1))
	{	allocate_vector(&storage, n);
		for (i = 0; i < n; i++)
		{	for ( i1=0; i1 < n; i1++)				// read in order
			{	//scanf("%d",&storage[i1]);
				storage[i1]= rand()%50;
			}
			int dest= i%p;
			if ( dest != p-1 )
				MPI_Send (storage, n, dtype, dest, dest, comm);
			else
			{	for ( j1=0; j1< n; j1++ )
					(*a)[idx][j1]=rand()%50;
				idx++;
			}
		}
		free(storage);
		
	}
	else
	{	for ( i=0; i< BLOCK_SIZE(my_rank,p,n); i++ )
			MPI_Recv (&((*a)[i][0]), n, dtype, p-1, my_rank, comm, &status);
	}
	MPI_Barrier(comm);
}

void read_row_interleaved_striped_vector ( dtype **a, MPI_Datatype dtype, int n, MPI_Comm comm)
{	int i, i1=0, j1, my_rank, p, idx=0;
	MPI_Status status;
	MPI_Comm_size (comm, &p);
	MPI_Comm_rank (comm, &my_rank);
	double *storage;
	if (my_rank == (p-1))
	{	allocate_vector(&storage, n);
		for (i = 0; i < n; i++)
		{	for ( i1=0; i1 < n; i1++)				// read in order
			{		//scanf("%d",&storage[i1]);
					storage[i1]=rand()%50;
			}
			int dest= i%p;
			if ( dest != p-1 )
				MPI_Send (storage, n, dtype, dest, dest, comm);
			else
			{	for ( j1=0; j1< n; j1++ )
					(*a)[idx]=rand()%50;
				idx++;
			}
		}
		free(storage);
	}
	else
	{	for ( i=0; i< BLOCK_SIZE(my_rank,p,n); i++ )
			MPI_Recv (&((*a)[i]), n, dtype, p-1, my_rank, comm, &status);
	}
	MPI_Barrier(comm);
}

/*
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
				{	//printf("\n%d %d\n",i,j);
					printf ("%6d", ((int **)a)[i][j]);
				}
			}
		}
		putchar ('\n');
	}
}

void print_row_interleaved_striped_matrix ( dtype **a, MPI_Datatype dtype, int n, MPI_Comm comm)
{	MPI_Status status;
	int size_of_type, i, my_rank, local_rows, p, max_block_size;
	MPI_Comm_rank (comm, &my_rank);
	MPI_Comm_size (comm, &p);
	local_rows = BLOCK_SIZE(my_rank,p,n);
	if (!my_rank)
	{	print_submatrix (a, dtype, local_rows, n);
		double **b;
		allocate_matrix(&b,local_rows,n);
		if (p > 1)
		{	for (i = 1; i < p; i++)
			{	MPI_Recv (&(b[0][0]), BLOCK_SIZE(i,p,n)*n, MPI_TYPE, i, i, MPI_COMM_WORLD, &status);
				print_submatrix (b, dtype, BLOCK_SIZE(i,p,n), n);
			}
			putchar ('\n');
		}
		free (b);
	}
	else
		MPI_Send (&(a[0][0]), local_rows * n, MPI_TYPE, 0, my_rank, MPI_COMM_WORLD);
}
*/

int main (int argc, char *argv[])
{	dtype **a, *b;			// A=a, B=b
	int my_rank, i, j, n, p, local_rows;
	
	MPI_Status status;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	
	if ( my_rank == 0)
	{	printf("Enter dimension of matrix: ");
		scanf("%d",&n);		// n*n matrix and n*1 vector
	}
	MPI_Bcast (&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	// we have only square matrices and hence local_rows = local_cols here
	local_rows = BLOCK_SIZE( my_rank, p, n);
	allocate_matrix(&a, local_rows, n);
	allocate_vector(&b, n);
	MPI_Barrier(MPI_COMM_WORLD);

	read_row_interleaved_striped_matrix ( &a, MPI_TYPE, n, MPI_COMM_WORLD); 	// done local distribution of matrices
	read_row_interleaved_striped_vector ( &b, MPI_TYPE, n, MPI_COMM_WORLD);
	//print_row_interleaved_striped_matrix (a, MPI_TYPE, n, MPI_COMM_WORLD);
	//print_row_interleaved_striped_vector (b, MPI_TYPE, n, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	free(a);
	free(b);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}


/* We now have an upper triangular system.  This must now be solved using back substitution. */
	/* Allocate space for and initialize to false every entry in the array that indicates which x values we have.*/
	have_x = (int *) malloc(sizeof(int) * order);
	if (have_x == NULL) ABORT;
	for (i = 0; i < order; i++)
		have_x[i] = 0;
	/* Allocate space for the data structure which indicates which processes we've sent a computed x value to.*/
	sent_yet = (int *) malloc(sizeof(int) * num_processes);
	if (sent_yet == NULL) ABORT;
	/* Allocate space for the array which holds computed x values.*/
	computed_x = (double *) malloc(sizeof(double) * order);
	if (computed_x == NULL) ABORT;
	/* Perform back substitution. */
	for (i = order - 1; i >= 0; i--) /* Rows */
	{	if (rp_map[perm_vector[i]] == my_rank)
		{	total = 0.0;
			for (j = i + 1; j < order; j++) /* X values */
			{	if (!have_x[j])
				{	 MPI_Recv(&computed_x[j], 1, MPI_DOUBLE, rp_map[perm_vector[j]], j, MPI_COMM_WORLD, &status);
					have_x[j] = 1;
				}
				total += computed_x[j] * my_a[perm_vector[i]][j];
				flops += 2;
			}
			computed_x[i] = (my_b[perm_vector[i]] - total) / my_a[perm_vector[i]][i];
			flops++;
			have_x[i] = 1;
			/* Send this value to the other processors. */
			for (j = 0; j < num_processes; j++)
				sent_yet[j] = 0;
			for (j = i - 1; j >= 0; j--) /* Rows */
			{	if (rp_map[perm_vector[j]] != my_rank)
				{	if (!sent_yet[rp_map[perm_vector[j]]])
					{	MPI_Send(&computed_x[i], 1, MPI_DOUBLE, rp_map[perm_vector[j]], i, MPI_COMM_WORLD);
						sent_yet[rp_map[perm_vector[j]]] = 1;
					}
				}
			}
		}
	}
	/* Write the computed values of x to disk, if we are the process that ended up being responsible for row 0. Also, perform the solution 	verfication step. */
	numerator   = 0.0;
	denominator = 0.0;
	if (rp_map[perm_vector[0]] == my_rank)
	{	result_file = fopen("result.txt", "w");
		fprintf(result_file, "Computed solution vector:\n\n");
		for (i = 0; i < order; i++)
		{	fprintf(result_file, "%lf\n", computed_x[i]);
			numerator   += pow((actual_x[i] - computed_x[i]), 2.0);
			denominator += pow(actual_x[i], 2.0);
		}
		fclose(result_file);
		numerator   = sqrt(numerator);
		denominator = sqrt(denominator);
		printf("\n\nError in computed solution: %e\n\n", numerator / denominator);
	}
	printf("Process %d performed %d floating point operations.\n", my_rank, flops);
	MPI_Finalize();
}  

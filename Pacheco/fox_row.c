/*
Matrix-matrix multiplication, Fox
Data distribution of matrix: row wise
Refer to Quinn- Matrix Multiplication- Row-wise parallel
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
{	dtype *local_a = (void *)malloc(r*c*sizeof(dtype));
	(*a) = (void *)malloc(r*PTR_SIZE);
	dtype *l= &local_a[0];
	int i;
	for (i=0; i<r; i++ )
	{	(*a)[i]= l;
		l += c;
	}
}

void read_row_striped_matrix ( dtype ***a, MPI_Datatype dtype, int n, MPI_Comm comm)
{	int i, i1, j1, my_rank, p;
	MPI_Status status;
	MPI_Comm_size (comm, &p);
	MPI_Comm_rank (comm, &my_rank);
	int local_rows= BLOCK_SIZE(my_rank,p,n);
	if (my_rank == (p-1))
	{	int *storage= malloc(local_rows*n*sizeof(dtype));
		for (i = 0; i < p-1; i++)
		{	for ( i1=0; i1 < BLOCK_SIZE(i,p,n) * n; i1++)				// read in order
			{		//scanf("%d",&storage[i1]);
					storage[i1]=rand()%50;
			}
			MPI_Send (storage, BLOCK_SIZE(i,p,n) * n, dtype, i, i, comm);
		}
		for ( i1=0; i1 < local_rows; i1++)
		{	for ( j1=0; j1< n; j1++ )
				(*a)[i1][j1]=rand()%50;
		}
		free(storage);
	}
	else
		MPI_Recv (&((*a)[0][0]), local_rows * n, dtype, p-1, my_rank, comm, &status);
	MPI_Barrier(comm);
}

void set_to_zero( dtype ***a, int r, int c )
{	int i,j;
	for ( i = 0; i < r; i++)
		for (j = 0; j < c; j++)
			(*a)[i][j]=0;
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
				{	//printf("\n%d %d\n",i,j);
					printf ("%6d", ((int **)a)[i][j]);
				}
			}
		}
		putchar ('\n');
	}
}

void print_row_striped_matrix ( dtype **a, MPI_Datatype dtype, int n, MPI_Comm comm)
{	MPI_Status status;
	int size_of_type, i, my_rank, local_rows, p, max_block_size;
	MPI_Comm_rank (comm, &my_rank);
	MPI_Comm_size (comm, &p);
	local_rows = BLOCK_SIZE(my_rank,p,n);
	if (!my_rank)
	{	print_submatrix (a, dtype, local_rows, n);
		int **b;
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

// must be squate matrix only. p is square number and n/sqrt(p) == 0
void local_matrix_multiply( dtype ***a, dtype*** b, dtype*** c, int row, int n, int my_rank, int stage, int p)
{	int i,j,k;
	for ( i = 0; i < row; i++)
	{	int r1= i;					// consider process != 0, they store local matrice starting by index 0
		int c1= ( BLOCK_LOW(my_rank,p,n)+ i + stage ) % n;
		// a[r1][c1] is the element that is bcasted i.e. multiplied to all columns of the row of B
		for ( j = 0; j < n; j++)
		{	//printf("r1: %d c1: %d i: %d j: %d Block_low: %d\n", r1, c1, i, j, BLOCK_LOW(my_rank,p,n));
			//printf("%d %d\n", (*a)[r1][c1], (*b)[(i+stage)%row][j] );
			(*c)[i][j] += ( (*a)[r1][c1] * (*b)[(i+stage)%(row)][j] );
			// multiply bcasted value with all values in the column of the given row
		}
	}
}

void fox( dtype*** local_a, dtype*** local_b, dtype*** local_c, int local_rows, int n, MPI_Comm comm )
{	int my_rank, source, dest, i, j, stage, bcast_root, p;
	MPI_Status status;
	MPI_Datatype matrix_type;
	MPI_Comm_size (comm, &p);
	MPI_Comm_rank (comm, &my_rank);
	source = (my_rank + 1) % p;			// Calculate addresses for circular shift of B
	dest = (my_rank + p - 1) % p;
	/* Set aside storage for the broadcast block of A */
	// refer to ./pointer_malloc_reference.c for understanding the concept and ./pass_2d_dynamic_matrix_bcast.c for how 2 pass dynamically allocated matrix. we pass the starting address of the continous blocks of array
	for (stage = 0; stage < n; stage++)
	{	if ( stage != 0 && stage % local_rows == 0 )
			MPI_Sendrecv_replace(&((*local_b)[0][0]),  (local_rows+1)*n, MPI_TYPE, dest, 0, source, 0, comm, &status);
		local_matrix_multiply(local_a, local_b, local_c, local_rows, n, my_rank, stage, p);		
	}
}

void send_extra_row( dtype ***b, int n, int local_rows, MPI_Comm comm)		// a process can do send and receive parallely
{	MPI_Status status;
	int my_rank, p, source, dest;
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	source = (my_rank + 1) % p;			// Calculate addresses for circular shift of B
	dest = (my_rank + p - 1) % p;
	MPI_Send ( &((*b)[0][0]), n, MPI_TYPE, dest, my_rank, comm);
	MPI_Recv (&((*b)[local_rows][0]), n, MPI_TYPE, source, source, comm, &status);
}

int main (int argc, char *argv[])
{	dtype **a, **b, **c;
	int my_rank, i, j, n, p, local_rows;
	
	MPI_Status status;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	
	if ( my_rank == 0)
	{	printf("Enter dimension of matrix: ");
		scanf("%d",&n);
	}
	MPI_Bcast (&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	// we take only square matrices and hence local_rows = local_cols here
	local_rows = BLOCK_SIZE( my_rank, p, n);
	allocate_matrix(&a, local_rows, n);
	allocate_matrix(&b, local_rows+1, n);				// B stores 1 extra
	allocate_matrix(&c, local_rows, n);
	set_to_zero(&c, local_rows, n);
	MPI_Barrier(MPI_COMM_WORLD);

	read_row_striped_matrix ( &a, MPI_TYPE, n, MPI_COMM_WORLD); 	// done local distribution of matrices
	read_row_striped_matrix ( &b, MPI_TYPE, n, MPI_COMM_WORLD);
	send_extra_row(&b, n, local_rows, MPI_COMM_WORLD);
	print_row_striped_matrix (a, MPI_TYPE, n, MPI_COMM_WORLD);
	print_row_striped_matrix (b, MPI_TYPE, n, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	fox( &a, &b, &c, local_rows, n, MPI_COMM_WORLD );
	free(a);
	free(b);
	print_row_striped_matrix (c, MPI_TYPE, n, MPI_COMM_WORLD);
	free(c);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}

/*
let A= 4*4 B= 4*4
Now p=2
distribution is:
p=0
A-	00	01	02	03				B-	00	01	02	03
	10	11	12	13					10	11	12	13
									20	21	22	23
B has 1 extra row because ot is needed in local_matrix_multiplication. We Send_Recv_Replace() the entire 3 riows of B(local_rows+1)

p=1
A-	20	21	22	23				B-	20	21	22	23
	30	31	32	33					30	31	32	33
									10	11	12	13

after Send_recv_replace()
distribution is:
p=0
A-	00	01	02	03				B-	20	21	22	23
	10	11	12	13					30	31	32	33
									00	01	02	03

p=1
A-	20	21	22	23				B-	00	01	02	03
	30	31	32	33					10	11	12	13
									20	21	22	23
*/

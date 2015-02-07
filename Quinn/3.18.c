#include<stdio.h>
#include<mpi.h>
#include<math.h>
#define MAX 100

// algorithm in solution pdf
// First reduction goves maximum among all, broadcast it to all and then find second maximum
// complexity O(n/p + log p )

int max ( int a, int b)
{	return a> b ? a : b;	}

int max_val(int a[], int n )
{	int mx=-100000000, i;
	for ( i=0; i<n; i++ )
		mx= max (mx, a[i]);
	return mx;
}

int max_val2(int a[], int n, int f_mx )
{	int mx=-100000000, i;
	for ( i=0; i<n; i++ )
	{	if ( a[i] != f_mx )
			mx= max (mx, a[i]);
	}
	return mx;
}

void Build_derived_type( int *a, int *n, MPI_Datatype *mesg_mpi_t_ptr )
{	int block_lengths[2];
	MPI_Aint displacements[2];
	MPI_Datatype typelist[2];  
	MPI_Aint start_address;
	MPI_Aint address;
	block_lengths[0] = MAX;
	block_lengths[1] = 1;
	typelist[0] = MPI_INT;
	typelist[1] = MPI_INT;
	displacements[0] = 0;
	MPI_Address(a, &start_address);
	MPI_Address(n, &address);
	displacements[1] = address - start_address;
	MPI_Type_struct(2, block_lengths, displacements, typelist, mesg_mpi_t_ptr);
	MPI_Type_commit(mesg_mpi_t_ptr);
}

void Get_data( int *a, int *n, int my_rank )
{	MPI_Datatype mesg_mpi_t;
	int i;
	if (my_rank == 0)
	{	scanf("%d", n);
		for ( i=0; i< *n ; i++ )
			a[i]= rand()%1000;	//scanf("%d", a+i);
		for ( i=0; i< *n; i++)
			printf("%d ", a[i]);
		printf("\n");
	}
	Build_derived_type(a, n, &mesg_mpi_t);
	MPI_Bcast(a, 1, mesg_mpi_t, 0, MPI_COMM_WORLD);
}

int main (int argc, char *argv[])
{	int i, my_rank, p, j, a[100], n;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	Get_data(a,&n,my_rank);
	int local_a[MAX], local_n= n/p;
	for ( i= my_rank *local_n, j=0; j < local_n; j++, i++ )			// alterantively use MPI_Scatter
		local_a[j]= a[i];
	int mx= max_val(local_a, local_n), g_mx;
	MPI_Reduce (&mx, &g_mx, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Bcast(&g_mx, 1, MPI_INT, 0, MPI_COMM_WORLD);
	int s_mx;
	mx= max_val2(local_a, local_n, g_mx);
	MPI_Reduce (&mx, &s_mx, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
	if ( my_rank == 0 )
		printf("%d\n", s_mx);
	MPI_Finalize();
	return 0;
}

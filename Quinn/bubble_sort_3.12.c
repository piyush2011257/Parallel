#include<mpi.h>
#include<stdio.h>
#define MAX 1000000

void Build_derived_type( int *a,int *n, MPI_Datatype *mesg_mpi_t_ptr )
{	int block_lengths[2];
	MPI_Aint displacements[2];
	MPI_Datatype typelist[2];
	MPI_Aint start_address;
	MPI_Aint address;
	block_lengths[0] = MAX;			// no *n, as we have no vale of n for my_rank =1 (itv also calls this process and hence wrong size and SIGSEGV )
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

void Get_data3( int *a, int *n, int my_rank )
{	MPI_Datatype mesg_mpi_t;
	if (my_rank == 0)
	{	//scanf("%d", n);
		*n= rand()%50 + 50;
		*n += 4-(*n % 4);
		int i;
		for ( i=0; i< *n; i++ )
			a[i]= rand()%100;//scanf("%d",a+i);
		//for ( i = 0; i< *n; i++ )
		//	printf("%d ", a[i]);
		//printf("\n ");
	}
	Build_derived_type(a, n, &mesg_mpi_t);
	MPI_Bcast(a, 1, mesg_mpi_t, 0, MPI_COMM_WORLD);
	/*MPI_Bcast(a, 4, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);*/
}

void seq_bubble ( int *a, int n )
{	int i;
	//if ( a[n-1] == 3 )
	//	printf("%d ", a[n-1]);
	for ( i=1; i< n; i++ )
	{	if ( a[i] < a[i-1] )
		{	a[i]=a[i]^a[i-1];
			a[i-1]=a[i]^a[i-1];
			a[i]=a[i]^a[i-1];
		}
	}
}

void bubble_sort( int *a, int n, int my_rank, int p )
{	MPI_Status  status;
	seq_bubble(a,n);
	int tmp;
	if ( my_rank != p-1 )
		MPI_Send(&a[n-1], 1, MPI_INT, my_rank+1, 0, MPI_COMM_WORLD);	//MPI_Send(a[n-1], p+1);
	if ( my_rank != 0 )
	{	MPI_Recv(&tmp, 1, MPI_INT, my_rank-1, 0, MPI_COMM_WORLD, &status);	//MPI_Recv(a[n-1], p-1);
		if ( a[0] < tmp )
		{	tmp = tmp ^ a[0];
			a[0] = tmp ^ a[0];
			tmp = tmp ^ a[0];
		}
		MPI_Send(&tmp, 1, MPI_INT, my_rank-1, 1, MPI_COMM_WORLD);	//MPI_Send(p-1);
	}
	if ( my_rank != p-1 )
		MPI_Recv(&a[n-1], 1, MPI_INT, my_rank+1, 1, MPI_COMM_WORLD, &status);	//MPI_Recv(a[n-1], p-1);
}

int main(int argc, char** argv)
{	int t=1000;
	MPI_Init(&argc, &argv);
	while (t-- > 0 )
	{	int my_rank, p, n, local_n, local_a[MAX], a[MAX], i, j;
		MPI_Status  status;
		MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
		MPI_Comm_size(MPI_COMM_WORLD, &p);
		Get_data3(a,&n,my_rank);
		local_n = n/p;
		for ( i=my_rank*local_n, j=0; j<local_n; i++, j++ )
			local_a[j]=a[i];
		for ( i=0; i < n; i++ )
			bubble_sort(local_a,local_n, my_rank, p);
		MPI_Gather(local_a, local_n, MPI_INT, a, local_n, MPI_INT, 0, MPI_COMM_WORLD);
		if (my_rank == 0)
		{	for ( i = 0; i<n-1; i++ )
				if ( a[i] > a[i+1] )
					printf("error! %d\n", a[i]);
		//	for ( i = 0; i<n; i++ )
		//		printf("%d ", a[i]);
		}
		//printf("\n");
	}
	MPI_Finalize();
}
// tested for 1000 random generated test cases with no error!

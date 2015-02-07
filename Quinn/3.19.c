#include<stdio.h>
#include<mpi.h>
#include<math.h>
#define MAX 100

// algorithm in solution pdf
// First reduction goves maximum among all, broadcast it to all and then find second maximum
// complexity O(n/p + log p )
// usage of MPI_MINLOC / MPI_MAXLOC

struct node
{	float val;
	int pos;
};
// for MPI_Reduce ( MPI_MAXLOC ). Every process will pass this structure in reduce(). It stores the value and position of the first occurence of the maximum value in the array

int max ( int a, int b)
{	return a> b ? a : b;	}

// find maximum value in the process and the idex of its first occurence in the global array NOT local array
struct node max_val( float a[], int n, int skip_pos )
{	struct node tmp;
	int mx=-100000000, i;
	for ( i=0; i<n; i++ )
	{	if ( a[i] > mx )
// so as to get the first occurence of maximum element in the array. By this we get the first occurence on maximum elemnt in the global array
		{	mx= max (mx, a[i]);
			tmp.val= mx;
			tmp.pos=i + skip_pos ;			// return position in the original array
			// i is the position in the local array. skip_pos is added to get the position in the original array
		}
	}
	return tmp;
}

struct node max_val2(float a[], int n, int pos, int skip_pos )
{	// finds the maximum elemnt in the array apart from the element at position pos ( since it is the global maximum ) and the position of the forst occurence of the element
	struct node tmp;
	int mx=-100000000, i;
	for ( i=0; i<n; i++ )
	{	if ( i != pos )
		{	if ( a[i] > mx )
// so as to get the first occurence of maximum element in the array. By this we get the first occurence on maximum elemnt in the global array
			{	mx= max (mx, a[i]);
				tmp.val= mx;
				tmp.pos=i + skip_pos ;			// return position in the original array
			}
		}
	}
	return tmp;
}

void Build_derived_type( float *a, int *n, MPI_Datatype *mesg_mpi_t_ptr )
{	int block_lengths[2];
	MPI_Aint displacements[2];
	MPI_Datatype typelist[2];  
	MPI_Aint start_address;
	MPI_Aint address;
	block_lengths[0] = MAX;
	block_lengths[1] = 1;
	typelist[0] = MPI_FLOAT;
	typelist[1] = MPI_INT;
	displacements[0] = 0;
	MPI_Address(a, &start_address);
	MPI_Address(n, &address);
	displacements[1] = address - start_address;
	MPI_Type_struct(2, block_lengths, displacements, typelist, mesg_mpi_t_ptr);
	MPI_Type_commit(mesg_mpi_t_ptr);
}

void Get_data( float *a, int *n, int my_rank )
{	MPI_Datatype mesg_mpi_t;
	int i;
	if (my_rank == 0)
	{	scanf("%d", n);
		for ( i=0; i< *n ; i++ )
			a[i]= rand()%1000;	//scanf("%d", a+i);
		for ( i=0; i< *n; i++)
			printf("%f ", a[i]);
		printf("\n");
	}
	Build_derived_type(a, n, &mesg_mpi_t);
	MPI_Bcast(a, 1, mesg_mpi_t, 0, MPI_COMM_WORLD);
}

int main (int argc, char *argv[])
{	int i, my_rank, p, j, n;
	float a[100];
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	Get_data(a,&n,my_rank);
	float local_a[MAX];
	int local_n= n/p;
	for ( i= my_rank *local_n, j=0; j < local_n; j++, i++ )			// alterantively use MPI_Scatter
		local_a[j]= a[i];
	// find global maximum and its first occurence
	struct node mx= max_val(local_a, local_n, local_n*my_rank), g_mx, s_mx;
	MPI_Reduce (&mx, &g_mx, 1, MPI_FLOAT_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
	// it finds the value maximum among the structures passed by different processes and if the value is same then the structure having lower value of index is returned. Thus we get the maximum value and its first occurence
	// mx is a struct of type float of value and in of positio and hence MPI_FLOAT_INT ( no type as MPI_INT_INT and hence we have to use MPI_FLOAT_INT)
	MPI_Bcast(&g_mx, 1, MPI_FLOAT_INT, 0, MPI_COMM_WORLD);
	// bcast the global value and its position in the original array
	if ( g_mx.pos / local_n == my_rank )		// g_mx.pos / local_n return the process whose g_mx.pos % local_n th position is the global element and hence this has to be skipped since we want the second largest element
		mx= max_val2(local_a, local_n, g_mx.pos % local_n, local_n*my_rank);
	else
		mx= max_val(local_a, local_n, local_n*my_rank);
	MPI_Reduce (&mx, &s_mx, 1, MPI_FLOAT_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
	// get the largest element from the set
	if ( my_rank == 0 )
		printf("%f %d\n", s_mx.val, s_mx.pos);
	MPI_Finalize();
	return 0;
}

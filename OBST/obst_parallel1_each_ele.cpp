#include<ctime>
#include <cstdio>
#include <cstring>
#include<algorithm>
#include<limits.h>
#include<mpi.h>
using namespace std;
#define BLOCK_OWNER(j,p,n) (((p)*((j)+1)-1)/(n))
#define BLOCK_LOW(id,p,n)  ((id)*(n)/(p))						// low_value= (in/p)
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)					// high value (i+1)n/p-1
#define BLOCK_SIZE(id,p,n) (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)			// size of the interval
#define PTR_SIZE sizeof(void*)

struct node
{	float val;	// cost
	int pos; 	// pos's cost ( index ) for root value
		node( float f=(float)INT_MAX, int v=-1)
	{	val=f;
		pos=v;
	}
};

int n;

void allocate_matrix ( int ***a, int r, int c )
{	int *local_a = (int *)malloc(r*c*sizeof(int));
	(*a) = (int **)malloc(r*PTR_SIZE);
	int *l= &local_a[0];
	int i;
	for (i=0; i<r; i++ )
	{	(*a)[i]= l;
		l += c;
	}
}

void set_to_val( int ***a, int r, int c, int val )
{	int i,j;
	for ( i = 0; i < r; i++)
		for (j = 0; j < c; j++)
			(*a)[i][j]=val;
}

void optimalSearchTree(int *freq, int *sum_frwd, MPI_Comm comm)
{	int my_rank, p;
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	int **cost, **root, block_n;
	allocate_matrix(&cost, n+3, n+3);
	allocate_matrix(&root, n+3, n+3);
	set_to_val(&cost, n+3, n+3, 0);
	set_to_val(&root, n+3, n+3, -1);
	for ( int i=1; i<=n ; i++ )
	{	cost[i][i]=freq[i];
		root[i][i]=i;
	}
	for ( int i=n; i>0; i-- )
	{	for ( int j=i+1; j<=n; j++ )
		{	int tmp=INT_MAX, tcost;
			if ( my_rank == 0 )
			{	block_n= root[i+1][j] - root[i][j-1] + 1;
				// we can avoid this bcast
				MPI_Bcast(&block_n, 1, MPI_INT, 0, comm);
			}
			else
				MPI_Bcast(&block_n, 1, MPI_INT, 0, comm);
			if ( BLOCK_SIZE(my_rank, p, block_n) > 0 )
			{	for ( int k=root[i][j-1]+ BLOCK_LOW(my_rank,p,block_n); (k <= root[i][j-1]+ BLOCK_HIGH(my_rank,p,block_n) && k <= root[i+1][j] ); k++)
				{	tcost = cost[i][k-1] + cost[k+1][j];
					if ( tcost <= tmp )
					{	tmp= tcost;
						root[i][j]=k;
					}
				}
			}
			struct node temp((float)tmp, (int)root[i][j]), g_temp;
			MPI_Allreduce (&temp, &g_temp, 1, MPI_FLOAT_INT, MPI_MINLOC, MPI_COMM_WORLD);
			cost[i][j]= sum_frwd[j] - sum_frwd[i-1] + (int)g_temp.val;
			root[i][j]=g_temp.pos;
			//printf("i: %d j: %d cost[i][j]: %d root[i][j]: %d\n",i,j,cost[i][j], root[i][j]);
		}
	}
	MPI_Barrier(comm);
	/*if ( my_rank == 0 )
	{	for ( int i=1; i<=n; i++ )
		{	for ( int j=i; j<=n; j++ )
				printf("%d\t", cost[i][j]);
			printf("\n");
		}
	}*/
	if ( my_rank == 0 )
		printf("%d\n", cost[1][n]);
	free(cost);
	free(root);
}

int main(int argc, char *argv[])
{	int my_rank, i, j, p, local_rows;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	
	int *freq, *sum_frwd;
	if ( my_rank == 0 )
	{	scanf("%d",&n);
		freq= (int *)malloc(sizeof(int)*(n+3));
		sum_frwd= (int *)malloc(sizeof(int)*(n+3));
		for ( int i=1; i<=n;i++ )
		{	freq[i]= rand()%100;//scanf("%d", freq+i);
			//printf("%d ",freq[i]);
		}
		//printf("\n");
	}
	printf("time=%.3lf sec.\n",(double) (clock())/CLOCKS_PER_SEC);		// total time till this place
	// 1st heurestic is derived type here!
	MPI_Bcast (&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if ( my_rank != 0 )
	{	freq= (int *)malloc(sizeof(int)*(n+3));
		sum_frwd= (int *)malloc(sizeof(int)*(n+3));
	}
	MPI_Bcast (freq, n+3, MPI_INT, 0, MPI_COMM_WORLD);
	memset(sum_frwd, 0, sizeof(sum_frwd[0])*(n+3));
	// compare Mops
	for ( int i=1; i<=n; i++ )
		sum_frwd[i]= sum_frwd[i-1]+freq[i];
	optimalSearchTree(freq, sum_frwd, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	printf("time=%.3lf sec.\n",(double) (clock())/CLOCKS_PER_SEC);		// total time till this place
	return 0;
}

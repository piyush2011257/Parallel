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
{	float cost;
	int root;
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

void allocate_struct ( struct node ***a, int r, int c )
{	struct node *local_a = (struct node *)malloc(r*c*sizeof(struct node));
	(*a) = (struct node **)malloc(r*PTR_SIZE);
	struct node *l= &local_a[0];
	int i;
	for (i=0; i<r; i++ )
	{	(*a)[i]= l;
		l += c;
	}
}

void initialize_struct( struct node ***a, int rc )
{	int i,j;
	for ( i = 0; i < rc; i++)
		for (j = 0; j < rc; j++)
		{	(*a)[i][j].cost=(float)INT_MAX;
			(*a)[i][j].root=-1;
		}
}

/* new storage form in the order of diagonal:
11 22 33 44 55 66
   12 23 34 45 56
      13 24 35 46
         14 25 36
	    15 26
	       16
*/
void optimalSearchTree(int *freq, int *sum_frwd, MPI_Comm comm)
{	int my_rank, p;
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	int block_n;
	struct node **nd, *temp_nd;
	allocate_struct(&nd,n+3, n+3);
	initialize_struct(&nd, n+3);
	for ( int i=1; i<=n ; i++ )
	{	nd[1][i].cost=freq[i];
		nd[1][i].root=i;
	}
	temp_nd= (struct node *)malloc (sizeof(struct node)*(n+3));
	block_n=n-1;
	for ( int i=2; i<=n; i++ )
	{	//printf("block_n: %d BLOCK_SIZE(my_rank,p,block_n): %d\n", block_n, BLOCK_SIZE(my_rank,p,block_n));
		if ( BLOCK_SIZE(my_rank,p,block_n) > 0 )
		{	for ( int j=i+BLOCK_LOW(my_rank,p,block_n); j<=i+BLOCK_HIGH(my_rank,p,block_n) && j <=n; j++ )
			{	int tmp=INT_MAX, tcost, diag_row, diag_col;
				diag_row=j-i+1;
				diag_col=j;
				int diag_n= nd[i-1][j].root - nd[i-1][j-1].root + 1;
				int sl= ( diag_col - diag_row + 1) - ( nd[i-1][j-1].root - diag_row );
				int sr= ( nd[i-1][j-1].root - diag_row ) + 1;
				//printf("rank: %d i: %d j: %d diag_row: %d diag_col: %d sl: %d sr: %d root[i][j-1]: %d root[i+1][j]: %d\n",my_rank, i,j, diag_row, diag_col, sl, sr, nd[i-1][j-1].root, nd[i-1][j].root);
				for ( int l=1; l<=diag_n ; sl--, sr++, l++)
				{	//printf("l: %d i-sl: %d j-sl: %d i-sr: %d j: %d\n",l, i-sl, j-sl, i-sr, j);
					tcost=INT_MAX;
					if ( i-sl >0 && j-sl > 0 )
						tcost = nd[i-sl][j-sl].cost;
					else
						tcost=0;
					if ( i-sr > 0 )
						tcost += nd[i-sr][j].cost;
					//printf("tcost: %d\n", tcost);
					if ( tcost <= tmp )
					{	tmp= tcost;
						nd[i][j].root=nd[i-1][j-1].root+l-1;
					}
				}
				nd[i][j].cost= (int)sum_frwd[diag_col] - sum_frwd[diag_row-1] + tmp;
				//printf("nd[i][j].cost: %d tmp: %d\n", (int)nd[i][j].cost, tmp);
			}
		}
		MPI_Allreduce (&(nd[i][i]), temp_nd, block_n, MPI_FLOAT_INT, MPI_MINLOC, MPI_COMM_WORLD);
		for ( int k=0; k<block_n; k++ )
			nd[i][i+k]=temp_nd[k];
		block_n--;
	}
	MPI_Barrier(comm);
	if ( my_rank == 0 )
	{	/*for ( int i=1; i<=n; i++ )
		{	for ( int j=i; j<=n; j++ )
				printf("%d\t", (int)nd[i][j].cost);
			printf("\n");
		}*/
	}
	if ( my_rank == 0 )
		printf("%d\n",(int) nd[n][n].cost);
	free(nd);
	free(temp_nd);
}

// documentation of this in the book
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
		//	printf("%d ",freq[i]);
		}
		//printf("\n");
	}
	// 1st heurestic is derived type here!
	printf("time=%.3lf sec.\n",(double) (clock())/CLOCKS_PER_SEC);		// total time till this place
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

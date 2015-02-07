#include<ctime>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <limits.h>
#include <mpi.h>
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

void allocate_struct ( struct node ***a, int rc )
{	int size= rc*(rc+1);
	size = size>>1;
	size--;
	//printf("size: %d\n", size);
	struct node *local_a = (struct node *)malloc(size*sizeof(struct node));
	(*a) = (struct node **)malloc((rc-1)*PTR_SIZE);
	struct node *l= &local_a[0];
	/* as our index startes from 1 not 0 first row and col is absolutely wasted
	for n=3, rc=5:
	00 01 02 03 04
	10 11 12 13
	20 21 22
	30 31
	*/
	for (int j=0, i=rc; j<rc-1 && i>0; i--, j++ )
	{	(*a)[j]= l;
		l += i;
	}
}

void initialize_struct( struct node ***a, int rc )
{	for ( int i = 0; i < rc; i++)
		for (int j = 0; j <= rc-i; j++)
		{	//printf("i: %d j: %d\n",i,j);
			(*a)[i][j].cost=(float)INT_MAX;
			(*a)[i][j].root=-1;
		}
}

/* new storage form in the order of diagonal:

base index=1
(diag_r, diag_c)		    i,j
   i/p diag			   storage format
11 22 33 44 55 66		11 12 13 14 15 16
12 23 34 45 56			21 22 23 24 25
13 24 35 46			31 32 33 34
14 25 36		        41 42 43
15 26				51 52
16			 	61
*/

// DEBUG and you are through
void solve ( int i, int j, struct node ***nd, int *sum_frwd)
{	//printf("i: %d j: %d\n",i,j);
	int tmp=INT_MAX, tcost, diag_row, diag_col;
	diag_row=j;
	diag_col=j+(i-1);
	// heurestic. Optimization in base case
	int diag_n= (*nd)[i-1][j+1].root - (*nd)[i-1][j].root + 1;
	int sl= ( diag_col - diag_row + 1) - ( (*nd)[i-1][j].root - diag_row);
	int sr= ( (*nd)[i-1][j].root - diag_row ) + 1;
	//printf("diag_n: %d sl: %d sr: %d\n", diag_n,sl,sr);
	for ( int l=1; l<=diag_n ; sl--, sr++, l++)
	{	//printf("p1: %d %d p2: %d %d\n", i-sr,j+sr,i-sl,j);
		tcost=INT_MAX;
		if ( i-sr >0 && j+sr <= n )
			tcost = (*nd)[i-sr][j+sr].cost;
		else
			tcost=0;
		if ( i-sl > 0 )
			tcost += (*nd)[i-sl][j].cost;
		if ( tcost <= tmp )
		{	tmp= tcost;
			(*nd)[i][j].root=(*nd)[i-1][j].root+l-1;
		}
	}
	(*nd)[i][j].cost= (int)sum_frwd[diag_col] - sum_frwd[diag_row-1] + tmp;
	//printf("root:%d cost: %f\n\n", (*nd)[i][j].root, (*nd)[i][j].cost);
}

void optimalSearchTree(int *freq, int *sum_frwd, MPI_Comm comm)
{	int my_rank, p;
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	int block_n;
	struct node **nd, **temp_nd;
	allocate_struct(&nd,n+2);
	initialize_struct(&nd,n+1);
	allocate_struct(&temp_nd,n+2);
	for ( int i=1; i<=n ; i++ )
	{	nd[1][i].cost=freq[i];
		nd[1][i].root=i;
	}
	int sc, ec, sr, er, size, block_size;
	sr=2;
	block_size=BLOCK_SIZE(my_rank,p,n);
	//printf("%d %d %d %d %d %d\n",sc,sl,sr,er,el,block_size);
		/*
		sl--sr,er
		-      -
		-     -
		-    -
		-   -
		-  -
	        - -
		el
	*/
	size=n-1;
	while ( size > 2*n )
	{	// ADD OVERLAP +1 FUNCTIONALITY
		sc= BLOCK_LOW(my_rank,p,size)+1;
		ec= BLOCK_HIGH(my_rank,p,size)+1;
		if ( my_rank != p-1 )
			ec++;
		er=sr+BLOCK_SIZE(my_rank,p,size)-1;
		for ( int i=sr, j=block_size; i<=er; i++, j-- )
		{	for ( int k=0; k<j; k++ )
			{	if ( nd[i][sc+k].root == -1 )
					solve(i, sc+k, &nd, sum_frwd);
			}
		}
		block_n=0;
		for ( int j=sr, i=n-sr+2; j<=er; j++, i-- )		// +1 fr 0th column
			block_n += i;
		MPI_Allreduce (&(nd[sr][0]), &(temp_nd[sr][0]), block_n, MPI_FLOAT_INT, MPI_MINLOC, MPI_COMM_WORLD);
		for ( int i= sr; i<=er; i++ )
			for ( int j=1; j<=n+1-i; j++ )
				nd[i][j]=temp_nd[i][j];
		size -= 2;
		sr += 2;
	}
	for ( int i=sr; size > 0; i++, size-- )
	{	//printf("block_n: %d BLOCK_SIZE(my_rank,p,block_n): %d\n", block_n, BLOCK_SIZE(my_rank,p,block_n));
		if ( BLOCK_SIZE(my_rank,p,size) > 0 )
		{	for ( int j=BLOCK_LOW(my_rank,p,size)+1; j<=BLOCK_HIGH(my_rank,p,size)+1 && j <=n; j++ )
			{	if ( nd[i][j].root == -1 )
					solve(i,j,&nd,sum_frwd);
			}
		}
		MPI_Allreduce (&(nd[i][0]), &(temp_nd[i][0]), n-i+2, MPI_FLOAT_INT, MPI_MINLOC, MPI_COMM_WORLD);
		for ( int k=0; k<n-i+2; k++ )
			nd[i][k]=temp_nd[i][k];
	}
	MPI_Barrier(comm);
	/*if ( my_rank == 0 )
	{	for ( int i=1; i<=n; i++ )
		{	for ( int j=1; j<=n+1-i; j++ )
				printf("%d\t", (int)nd[i][j].cost);
			printf("\n");
		}
	}*/
	if ( my_rank == 0 )
		printf("%d\n",(int) nd[n][1].cost);
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
		{	freq[i]= rand()%100;
			//scanf("%d", freq+i);
			//printf("%d ",freq[i]);
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

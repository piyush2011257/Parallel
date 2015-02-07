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

base index=1
  i/p diag			   storage format
11 22 33 44 55 66		11 12 13 14 15 16
   12 23 34 45 56		   22 23 24 25 26
      13 24 35 46		      33 34 35 36
         14 25 36		         44 45 46
	    15 26			    55 56
	       16			       66

*/

// can be madse inline type to reduce function call overhead
void solve ( int i, int j, struct node ***nd, int *sum_frwd)
{	//printf("i: %d j: %d\n",i,j);
	int tmp=INT_MAX, tcost, diag_row, diag_col;
	diag_row=j-i+1;
	diag_col=j;
	// heurestic. Optimization in base case
	int diag_n= (*nd)[i-1][j].root - (*nd)[i-1][j-1].root + 1;
	int sl= ( diag_col - diag_row + 1) - ( (*nd)[i-1][j-1].root - diag_row);
	int sr= ( (*nd)[i-1][j-1].root - diag_row ) + 1;
	for ( int l=1; l<=diag_n ; sl--, sr++, l++)
	{	tcost=INT_MAX;
		if ( i-sl >0 && j-sl > 0 )
			tcost = (*nd)[i-sl][j-sl].cost;
		else
			tcost=0;
		if ( i-sr > 0 )
			tcost += (*nd)[i-sr][j].cost;
		if ( tcost <= tmp )
		{	tmp= tcost;
			(*nd)[i][j].root=(*nd)[i-1][j-1].root+l-1;
		}
	}
	(*nd)[i][j].cost= (int)sum_frwd[diag_col] - sum_frwd[diag_row-1] + tmp;
}

void optimalSearchTree(int *freq, int *sum_frwd, MPI_Comm comm)
{	int my_rank, p;
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	int block_n;
	struct node **nd, **temp_nd;
	allocate_struct(&nd,n+1, n+1);
	initialize_struct(&nd, n+1);
	allocate_struct(&temp_nd,n+1, n+1);
	// TIME ELAPSED FROM HERE!!
	/*for ( int i=0; i<n+3; i++ )
	{	for ( int j=0; j<n+3; j++ )
			printf("%p  ", nd+i+j);
		printf("\n");
	}
	for ( int i=0; i<n+3; i++ )
	{	for ( int j=0; j<n+3; j++ )
			printf("%p  ", temp_nd+i+j);
		printf("\n");
	}*/
	for ( int i=1; i<=n ; i++ )
	{	nd[1][i].cost=freq[i];
		nd[1][i].root=i;
	}
	int sc, sl, sr, er, el, curr_proc=my_rank, block_size;
	// reverse from right to left
	// ec=n-1-BLOCK_LOW(my_rank,p,n);
	sc=n-BLOCK_HIGH(my_rank,p,n);
	sr=sl=el=1;
	er=sr+BLOCK_SIZE(my_rank,p,n)-1;
	block_size=BLOCK_SIZE(my_rank,p,n);
	//printf("%d %d %d %d %d %d\n",sc,sl,sr,er,el,block_size);
	/*
		sl,el--sr
		 -     -
		  -    -
		   -   -
		    -  -
		     - -
		       er
	*/
	
	/*
		sl
		- -
		-  -
		-   -
		-    -
		-------sr
		-      -
		-      -
		el------
		 -     -
		  -    -
		   -   -
		    -  -
		     - -
		       er
	*/
	for ( int iter=0; iter < p; iter++, curr_proc++ )
	{	//printf("curr: %d p-rank: %d\n", curr_proc, p-my_rank);
		if ( curr_proc < p-my_rank )
		{	//printf("yo\n");
			for ( int i= sl, j=1; i<sr; i++, j++ )
			{	for ( int k=0; k<j; k++ )
				{	if ( nd[i][sc+k].root == -1 )
						solve(i, sc+k, &nd, sum_frwd);
					// solve a[i][sc+k] if not already solved
				}
			}
			for ( int i=sr; i<el; i++ )
			{	for ( int k=0; k< block_size; k++ )
				{	if ( nd[i][sc+k].root == -1 )
						solve(i, sc+k, &nd, sum_frwd);
					// solve a[i][sc+k] if not already solved
				}
			}
			for ( int i=el, j=block_size; i<=er; i++, j-- )
			{	for ( int k=0; k<j; k++ )
				{	if ( nd[i][sc+block_size-1-k].root == -1 )
						solve(i, sc+block_size-1-k, &nd, sum_frwd);
					// solve a[i][sc+k] if not already solved
				}
			}
		}
		int tsl=sl, ter=er;
		// can be merged to 1 single reduce
		MPI_Allreduce (&sl, &tsl, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
		MPI_Allreduce (&er, &ter, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
		block_n= (n+1) * ( ter-tsl+1 );
		//printf("block_n: %d %d %d\n", block_n, ter, tsl);
		MPI_Allreduce (&(nd[tsl][0]), &(temp_nd[0][0]), block_n, MPI_FLOAT_INT, MPI_MINLOC, MPI_COMM_WORLD);
		/*for ( int i=0; i<n; i++ )
		{	for ( int j=i+1; j<=n; j++ )
				printf("%f ", temp_nd[i][j].cost);
			printf("\n");
		}
		printf("\n");
		for ( int i=1; i<=n; i++ )
		{	for ( int j=i; j<=n; j++ )
				printf("%f ", nd[i][j].cost);
			printf("\n");
		}*/
		for ( int i= tsl; i<=ter; i++ )
			for ( int j=1; j<=n; j++ )
				nd[i][j]=temp_nd[i-tsl][j];
		if ( curr_proc < p-my_rank )
		{	sl = el+1;
			sr = er+1;
			el += BLOCK_SIZE(my_rank+1,p,n) - 1;
			er += BLOCK_SIZE(my_rank+1,p,n) - 1;
		}
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

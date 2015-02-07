#include<stdio.h>
#include<mpi.h>
#include<math.h>
#define MAX 100

float min ( float f1, float f2 )
{	return f1 < f2 ? f1 : f2;	}

float cal_dist( float x1, float y1, float x2, float y2 )
{	float x= (float)(x1-x2)*(x1-x2);
	float y= (float)(y1-y2)*(y1-y2);
	return sqrt((float)x+y);
}

float min_dist ( int x[], int y[], int n, int x0, int y0 )
{	float tmp, min_tmp=1e9;
	int i;
	for ( i=0; i<n; i++ )
	{	tmp= cal_dist(x[i], y[i],x0,y0);
		min_tmp= min(tmp, min_tmp);
	}
	return min_tmp;
}

int main (int argc, char *argv[])
{	int i, my_rank, p, j;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	int n=8, x[8]={7,8,2,4,-4,-5,-1,9}, y[8]={1,-1,4,-5,-6, 7, 0, 3}, x0=0, y0=0;
	int local_x[MAX], local_y[MAX], local_n= n/p;
	for ( i= my_rank *local_n, j=0; j < local_n; j++, i++ )
	{	local_x[j]= x[i];
		local_y[j]= y[i];
	}
	float tmp= min_dist(local_x, local_y, local_n, x0, y0), min_val;
	MPI_Reduce (&tmp, &min_val, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
	if ( my_rank == 0 )
		printf("%f\n", min_val);
	MPI_Finalize();
	return 0;
}

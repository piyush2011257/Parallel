#include<stdio.h>
#include <stdlib.h>
#include <mpi.h>

int malloc2dfloat(float ***array, int n, int m)		// passing address will ensure that scope doesnt die out (its call by pointer) using ** does a call by value!
{	/* allocate the n*m contiguous items */
	float *p = (float *)malloc(n*m*sizeof(float));
	if (!p)
		return -1;
	/* allocate the row pointers into the memory */
	(*array) = (float **)malloc(n*sizeof(float*));
	if (!(*array))
	{	free(p);
		return -1;
	}
	/* set up the pointers into the contiguous memory */
	int i;
	for ( i=0; i<n; i++) 
		(*array)[i] = &(p[i*m]);
	return 0;
}

int main(int argc, char **argv)
{   
    float **array, ***arr;
    int rank,size,i,j;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

 malloc2dfloat(&array, 10, 10);
	arr= &array;
if (rank == 0) {
    for(i=0;i<10;i++)
         for(j=0;j<10;j++)
              array[i][j]=i+j;
}
MPI_Bcast(&((*arr)[0][0]), 10*10, MPI_FLOAT, 0, MPI_COMM_WORLD);			// understand this concept
   if ( rank != 0 )
	{	for ( i=0; i<10; i++)
		{	for ( j=0; j<10; j++ )
				printf("%lf\t", array[i][j]);
			printf("\n");
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}

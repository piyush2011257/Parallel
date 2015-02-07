#include<stdio.h>
#include <stdlib.h>

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
{   	float **array, ***arr;
	int size,i,j;
	malloc2dfloat(&array, 5, 5);
	arr= &array;
	// array is of pointer type == &array[0]. STartinjg address of the row malloc
	printf("array: %p \t&array[0]: %p \nIn bcast we pass these address: &(arr[0][0]): %p \t&((*arr)[0][0]): %p\n", array, &array[0], &(array[0][0]), &((*arr)[0][0]));
	// the above addresses are the base address of the continuous memory locations that store values!
	printf("local_a\n");
	for ( i=0; i<5; i++ )
	{	for( j=0; j<5; j++)
			printf("%p\t",array[i]+j);
		printf("\n");
	}
	printf("a\n");
	for ( i=0; i<5; i++ )
		printf("%p\n",array+i);
	for(i=0;i<5;i++)
		for(j=0;j<5;j++)
			array[i][j]=i+j;
	return 0;
}

#include<stdio.h>
#include<mpi.h>
#include<math.h>
#define MAX 100

int main (int argc, char *argv[])
{	int i, my_rank, p, j;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	int x[2]={ rand()%100, rand()%100};
	x[0]-=my_rank;
	x[1]-=my_rank;
	printf("%d %d %d\n", my_rank, x[0],x[1]);
	int ans_x[2];
	MPI_Reduce (&x, &ans_x, 2, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
	// ans_x[0]= min ( x[0] ) for all processes from 0 -> p-1
	// ans_x[1]= min ( x[1] ) for all processes from 0 -> p-1
	if ( my_rank == 0 )
		printf("%d %d\n", ans_x[0], ans_x[1]);
	MPI_Finalize();
	return 0;
}

/*
O/P:
piyush@piyush-TravelMate-4740:~/Parallel Computing/Programs/Quinn$ mpirun -np 10 ./a.out 
1 82 85
3 80 83
5 78 81
9 74 77
6 77 80
7 76 79
8 75 78
2 81 84
4 79 82
0 83 86
74 77

74 = min [x0's]
77 = min [x1's]
*/

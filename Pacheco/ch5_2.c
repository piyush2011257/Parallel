#include<stdio.h>
#include<mpi.h>

int main( int argc, char **argv )
{	int my_rank, p;
	MPI_Status  status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	int x,y,z;
	switch(my_rank)
	{	case 0:	{	x=0, y=1, z=2;
				MPI_Bcast( &x, 1, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Send(&y, 1, MPI_INT, 2, 43, MPI_COMM_WORLD);
				MPI_Bcast( &z, 1, MPI_INT, 0, MPI_COMM_WORLD);
				break;
			}
		case 1:	{	x=3, y=4, z=5;
				MPI_Bcast( &x, 1, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Bcast( &y, 1, MPI_INT, 0, MPI_COMM_WORLD);
				break;
			}
		case 2:	{	x=6, y=7, z=8;
				MPI_Bcast( &z, 1, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Recv(&x, 1, MPI_INT, 0, 43, MPI_COMM_WORLD, &status);
				MPI_Bcast( &y, 1, MPI_INT, 0, MPI_COMM_WORLD);
				break;
			}
	}
	printf("my_rank: %d x: %d y: %d z: %d\n", my_rank,x,y,z);
	MPI_Finalize();
	return 0;
}

/*
o/p:
my_rank: 0 x: 0 y: 1 z: 2
my_rank: 1 x: 0 y: 2 z: 5
my_rank: 2 x: 1 y: 2 z: 0

Time	Process 0	Process 1	Process 2
1	Bcast(x)	Local work	Local work
2	Send(y)		Local work	Local work
3	Bcast(z)	Local work	Local Work
4	LOcal Work	Local work	Receive(x)		// process 2 receives value of y of process 0 in x of process 2
5	Local work	Bcast(x)	Local work		// process 1 receives x of 0 in x of 1 and z of 0 in y of 1
6	Local work	Bcast(y)	Local work
7	Local work	Local work	Bcast(z)		// process 2 receives x of 0 in z of 2 and z of 0 in y of 2
8	Local work	Local work	Bcast(y)

Hence the output
*/

#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{	int i, rank, nprocs, count, start, stop, nloops;
	MPI_Init(&argc, &argv);
	// get the number of processes, and the id of this process
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	// we want to perform 1000 iterations in total. Work out the number of iterations to perform per process...
	count = 1000 / nprocs;
	// we use the rank of this process to work out which iterations to perform.
	start = rank * count;
	stop = start + count;
	// now perform the loop
	nloops = 0;
	for (i=start; i<stop; ++i)
		++nloops;
	printf("Process %d performed %d iterations of the loop.\n", rank, nloops);
	MPI_Finalize();
	return 0;
}
/*
You can only run as many processes in parallel as there are different functions to be run. If you have more processes than functions, then the extra processes will be idle. Also, if different functions take different amounts of time, then some processes may finish earlier than other processes, and they will be left idle waiting for the remaining processes to finish.

One way of achieving better performance is to use MPI to parallelise loops within your code. Lets imagine you have a loop that requires 1000 iterations. If you have two processes in the MPI team, then it would make sense for one process to perform 500 of the 1000 iterations while the other process performs the other 500 of 1000 iterations. This will scale as as more processes are added, the iterations of the loop can be shared evenly between them, e.g.

    2 processes : 500 iterations each
    4 processes : 250 iterations each
    100 processes : 10 iterations each
    1000 processes : 1 iteration each

Of course, this only scales up to the number of iterations in the loop, e.g. if there are 1500 processes, then 1000 processes will have 1 iteration each, while 500 processes will sit idle.

Also, and this is quite important, this will only work if each iteration of the loop is independent. This means that it should be possible to run each iteration in the loop in any order, and that each iteration does not affect any other iteration. This is necessary as running loop iterations in parallel means that we cannot guarantee that loop iteration 99 will be performed before loop iteration 100.
*/

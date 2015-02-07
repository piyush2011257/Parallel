/* C Example */
#include <mpi.h>
#include <stdio.h>

int main (int argc, char* argv[])
{	int rank, size;
	MPI_Init (&argc, &argv);    		/* starts MPI */
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);  /* get current process id */
	MPI_Comm_size (MPI_COMM_WORLD, &size);	/* get number of processes */
	printf( "Hello world from process %d of %d\n", rank, size );
	MPI_Finalize();
	return 0;
}

/*
To run the program you type:
mpirun hello_mpi

This runs your program as a single process on the computer you are using. If you want to run your program using multiple processes, then type;

mpirun -np 4 hello_mpi

The option -np 4 tells mpirun that you want to run the program as four processes. You can choose as many processes as is sensible, e.g. if you have 8 processor cores on your computer, then you may choose -np 8. Each process executed every line on code in the main function of the program. The four processes in the team ran in parallel together. However, note that there is no guarantee that the processes will print in order.	

In the above example, when -np 4 was used, the four copies of the program hello_mpi were run in parallel as a single program. Each copy had its own main function, and its own thread of execution. Each copy executed the same code, and so each copy printed Hello MPI! to the screen. In this example, all four copies of the program were run on the local computer. The principle advantage of MPI over a parallel programming technology such as OpenMP, is that you can run the multiple copies of the program as multiple processes over multiple computers. So, if you had four computers available, then you can set up the mpirun command such that hello_mpi is run as four processes, with one process per computer. How to set up mpirun in this way depends on the details of the specific MPI library that you are using. Generally, this is achieved by using a file that lists the names of computers that you wish to use. Normally, the queuing system you use to submit your job to a cluster should set up this host file for you, and automatically link it to mpirun. If a host file has been set up listing four nodes, e.g.

node001
node002
node003
node004

and this file has been connected to mpirun, then the command

mpirun -np 4 hello_mpi

would run hello_mpi as a single program of four processes, with the first process on computer node001, the second on node002, the third on node003 and the fourth process on node004.
*/

/* greetings.c -- greetings program
 *
 * Send a message from all processes with rank != 0 to process 0.
 *    Process 0 prints the messages received.
 *
 * Input: none.
 * Output: contents of messages received by process 0.
 *
 * See Chapter 3, pp. 41 & ff in PPMPI.
 */

/*
mpicc greetings.c -o ./a.out
mpirun -np 4 ./a.out
*/
#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char* argv[])
{	int my_rank;       /* rank of process      */
	int p;             /* number of processes  */
	int source;        /* rank of sender       */
	int dest;          /* rank of receiver     */
	int tag = 0;       /* tag for messages     */
	//tag is same for allMessgae Passing
	char message[100]; /* storage for message  */
	MPI_Status  status;/* return status for receive	*/
	/* Start up MPI */
	MPI_Init(&argc, &argv);		// this creates a for loop ( my_rank=0-> p-1 , once for each process )
	/* Find out process rank  */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	/* Find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	//printf("my rank: %d\n",my_rank);// no_of_process: %d\n", my_rank, p);
	if (my_rank != 0)
	{	/* Create message */
		sprintf(message, "Greetings from process %d!\0",my_rank);
		dest = 0;
		/* Use strlen+1 so that '\0' gets transmitted */
		MPI_Send(message, strlen(message)+1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
	}
	else
	{	/* my_rank == 0 */
		for (source = 1; source < p; source++)
		{	MPI_Recv(message, 100, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
			printf("%s\n", message);
		}
	}
	/* Shut down MPI */
	MPI_Finalize();
	return 0;
}

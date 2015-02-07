// mpicc greetings.c -o ./a.out
// mpirun -np 4 ./a.out

#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char* argv[])
{	int my_rank, p, source, dest, tag = 0;
	char message[100];
	MPI_Status  status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	printf("my rank: %d\n",my_rank);
	sprintf(message, "Greetings from process %d!",my_rank);
	dest = (my_rank+1)%p;
	// first send then receive
	MPI_Send(message, strlen(message)+1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
	source = (my_rank-1);
	if ( source == -1 )
		source = p-1;
	MPI_Recv(message, 100, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
	printf("%s\n", message);
	MPI_Finalize();
	return 0;
}

/*
No effect of the order as all the programs run simultanously using multiple copies. let ith process send message to i+1th process. At the same instant or before the i+1th process executes MPI_Recv() from ith process. Now since ith tag is just about to send the message, i+1th process will wait to receive the message from ith process and continue execution. They all run parallely sending message to ther and receiving from other. In cases of delays between send and recieve, that process waits to receive from the corresponding process. Refer to point 3 of ch3_q3.c and you will understand the concept.
Each process must send its message first and then receive. In the other order each process hangs waiting for messages that never arrive. In the source code coming from this problem we see coded another message sending strategy where the even processors send first and then receive while the odd processor receive first and then send. This message scheduling works as well.
When run on one processor, processor 0 sends and receives a message from itself.
*/


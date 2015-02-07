/* Problem 3.6.3 in PPMPI
dest to 1 in all sending process (invoking an incorrect MPI_Recv). Hangs waiting forever ... MPI_Recv never completes. THis is because if MPI_Recv exists for a source then there musrt exists MPI_Send() from the source else MPI_Recv() will keep on waiting till it gets message from the source!
*/

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
	if (my_rank != 0)
	{	sprintf(message, "Greetings from process %d!",my_rank);
		dest = 1;			// dest changed
		MPI_Send(message, strlen(message)+1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
	}
	else
	{	for (source = 1; source < p-1; source++)
		{	MPI_Recv(message, 100, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
			printf("%s\n", message);
		}
	}
	MPI_Finalize();
	return 0;
}
// run this modified code and you wont get the error!
/*int main(int argc, char* argv[])
{	int my_rank, p, source, dest, tag = 0;
	char message[200];
	MPI_Status  status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	printf("my rank: %d\n",my_rank);
	if (my_rank > 0 && my_rank < 2)
	{	sprintf(message, "Greetings from process %d!",my_rank);
		dest = 0;
		MPI_Send(message, strlen(message)+1, MPI_INT, dest, tag, MPI_COMM_WORLD);
	}
	else if ( my_rank == 0 )
	{	for (source = 1; source < p-1; source++)
		{	MPI_Recv(message, 105, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
			printf("%s\n", message);
		}
	}
	else
		sprintf(message, "Greetings from process %d!",my_rank);
	MPI_Finalize();
	return 0;
}*/

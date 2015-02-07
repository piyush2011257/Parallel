/* Problem 3.6.3 in PPMPI

1) Incorrect MPI data type in MPI_Send call / MPI_Recv
CHAR->INT: code crashed (larger data type)
Fatal error in MPI_Recv: Message truncated, error stack:
MPI_Recv(186).....................: MPI_Recv(buf=0xbfc09f74, count=100, MPI_CHAR, src=1, tag=0, MPI_COMM_WORLD, status=0xbfc09f4c) failed
MPIDI_CH3U_Receive_data_found(129): Message from rank 1 and tag 0 truncated; 104 bytes received but buffer size is 100

This was because char->int so total size of message became 104 bytes but the MSG_Recv(_, 100,.. ) had buffer size of 100 bytes which was lesser than 104 bytes.
Rectification:
char message[100] -> char message[200]
MPI_Recv(message, 100, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status) -> MPI_Recv(message, 200, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status)
Now since the size of buffer was increased to 200> 104, the code ran without any error
No error even on changing MPI_CHAR to MPI_INT / MPI_FLOAT in either MPI_Send or MPI_Recv or both.

2) Incorrect recieve size (tried 10)
code crashed (Refer to 1st point)

3) dest to 1 in all sending process (invoking an incorrect MPI_Recv). Hangs waiting forever ... MPI_Recv never completes
refer to ch3_q3_2.c

4) Incorrect string length (removed +1 from MPI_Send call)
	-Still worked...(removing more elements)
	-Prints until print process gets "\0"  (depends on contents of memory)
	- Worked correct is +2,+3,..+n instead of +1.
	+1 actually adds '\0' char to the Message to mark end of it!

note:
no effect of following changes:
sprintf(message, "Greetings from process %d!",my_rank);		// add '\0' in the messgae itself !!
all valid:
MPI_Send(message, strlen(message)-3, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
MPI_Send(message, strlen(message)-100, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
MPI_Send(message, strlen(message)+3, MPI_CHAR, dest, tag, MPI_COMM_WORLD);

5) Incorrect tag field in MPI recieve
     -program hangs
made following changes:
MPI_Send(message, strlen(message)+1, MPI_CHAR, dest, my_rank, MPI_COMM_WORLD);
MPI_Recv(message, 100, MPI_CHAR, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
program now runs correctly as we use MPI_ANY_TAG
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
	if (my_rank != 0 )
	{	sprintf(message, "Greetings from process %d!",my_rank);
		dest = 0;
		MPI_Send(message, strlen(message)+1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
	}
	else if ( my_rank == 0 )
	{	for (source = 1; source < p; source++)
		{	MPI_Recv(message, 100, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
			printf("%s\n", message);
		}
	}
	MPI_Finalize();
	return 0;
}

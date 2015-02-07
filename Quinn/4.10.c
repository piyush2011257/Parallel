#include<stdio.h>
#include<mpi.h>

int eval ( int n )
{	int count=0, j1, j2, j3, j4, j5;
	for ( j1=0; j1<=9; j1++)
	{	if ( j1 == n )
			continue;
		for ( j2=0; j2<=9; j2++)
		{	if ( j2 == j1 )
				continue;
			for ( j3=0; j3<=9; j3++)
			{	if (j3 == j2)
					continue;
				for ( j4=0; j4<=9; j4++)
				{	if ( j4 == j3 )
						continue;
					for ( j5=0; j5<=9; j5++)
					{	if (j5 == j4 )
							continue;
						int sum= j1+j2+j3+j4+j5+n;
						if ( sum % 7 == 0 || sum % 11 ==0 || sum % 13 == 0 )
							continue;
						//if ( j1 == j2 || j2 == j3 || j3 == j4 || j5 == j4 || j1 == n )
						//	printf("yo yo\n");
						count++;
					}
				}
			}
		}
	}
	return count;
}

int main( int argc, char **argv)
{	int my_rank, p;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &p);
	int local_n= 9/p, i, count=0, j;
	for ( i= my_rank*local_n + 1, j=0; j < local_n; i++, j++)
	{	printf("i: %d\n",i);
		count += eval(i);
	}
	printf("my_rank: %d count: %d\n", my_rank, count);
	MPI_Finalize();
	return 0;
}


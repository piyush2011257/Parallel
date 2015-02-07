/* send_col.c -- send the third column of a matrix from process 0 to
 *     process 1
 *
 * Input:  None
 * Output:  The column received by process 1
 *
 * Note:  This program should only be run with 2 processes
 *
 * See Chap 6., pp. 96 & ff in PPMPI
 */
#include <stdio.h>
#include "mpi.h"

int main(int argc, char* argv[])
{	int p, my_rank;
	float A[10][10];
	MPI_Status status;
	MPI_Datatype column_mpi_t;
	int i, j;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	/* note for communication type signatue is (t0,d0), (t1,d1), (t2,d2), (t3,d3)... (tn,dn). In getdata3() we specify each thing explicitly. For MPI_Vector: MPI_Type_Vector(blocks, elements_in_each_block, space_bw_elemnts, type_ new_data_type)
	no_of_blocks aka no_of_elements_that_are_being transerred. here we want to send third column-> a[0][2], a[1][2],...a[9][2] -> total of 10 elemnts that are to be send-> 10 copies of equally spced elements-> 10 blocks
	elements in each block-> for each copy among 10 blocks there is just 1 elment whose copy is created for each block. a[0][2] for 0th block, a[1][2] for 1st block.... a[9][2] for 9th block hence elements_in_ach_block= 1
	space between each block= 10 here!! 10x10 array
	Hence we get a signature:
	(1*MPI_Float,0), (1*MPI_Float, 10*sizeof(MPI_Float)), ...... (1*MPI_Float, 90*sizeof(MPI_Float))-> hence our signature
	1*MPI_Floatv as no.of elements in each block =1 for each copy
	*/
	MPI_Type_vector(10, 2, 10, MPI_FLOAT, &column_mpi_t);			// it creates a new data type just like int / float but of discontinuous memory locations BUT remeber sending is based on data type / type signature not on the basis of displacement. REFER to send_col_to_row.c
	// see the effect of it. it send 2 continuous elements i.e 2nd and 3rd column
	//MPI_Type_vector(10, 2, 10, MPI_FLOAT, &column_mpi_t);
	// understand the concept in terms of memory locations. int = 4 consecutive memory locations. The above type will mean 10 sepatae memory location, but allseparated by same amount of memory locations ( 10 locations here ) e.g. 100, 110, 120, 130,...
	// increasing block length will send continuous arrays- if block length = 3 then -> a[0][2,3,4], a[1][2,3,4], a[2][2,3,4] ... and so on
	// use this to sed alternate types a[0][2], a[2][2], a[4][2]...
	// MPI_Type_vector(10, 1, 10, MPI_FLOAT, &column_mpi_t);
	// used with all derived type communications
	MPI_Type_commit(&column_mpi_t);
	if (my_rank == 0)
	{	for (i = 0; i < 10; i++)
			for (j = 0; j < 10; j++)
				A[i][j] = (float) j*i;
// now here derived type used. base address= &a[0][2]. creates 10 copies of equally spaced elemenst-> a[0][2], a[1][2], .... a[9][2] and send them
		MPI_Send(&(A[0][2]), 1, column_mpi_t, 1, 0, MPI_COMM_WORLD);		// just acts like a data type
		// means 10 element starting from a[0][2], then a[1][2], a[2][2], a[3][2] ..
		// try changing send. this sends data from a[1][2]->a[9][2] -> 9 elents not 10
		// MPI_Send(&(A[1][2]), 1, column_mpi_t, 1, 0, MPI_COMM_WORLD);
		// try changing send. this sends data from a[3][2]->a[12][2] -> 6 elements not 10
		// MPI_Send(&(A[3][2]), 1, column_mpi_t, 1, 0, MPI_COMM_WORLD);
	}
	else
	{	/* my_rank = 1 */
		// receives 10 values i the a[i][2] as defied in type_vector and sent by mpi_send()
		MPI_Recv(&(A[0][2]), 1	, column_mpi_t, 0, 0, MPI_COMM_WORLD, &status);
		// receives 10 elements in the order a[0][2], a[1][2], a[2][2], a[3][2], ..
		// use same recv with modified send ad see the difference. it might lead to bad terination depending on memory reference
		//MPI_Recv(&(A[1][2]), 1, column_mpi_t, 0, 0, MPI_COMM_WORLD, &status);
		for (i = 0; i < 10; i++)
			printf("%3.1f ", A[i][2]);
		printf("\n");
		/*for (i = 0; i < 10; i++)
		{	for (j = 0; j < 10; j++)
				printf("%3.1f\t", A[i][j]);
			printf("\n");
		}*/
	}
	MPI_Finalize();
}

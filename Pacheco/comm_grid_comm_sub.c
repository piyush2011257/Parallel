/* top_fcns.c -- test basic topology functions
 *
 * Input: none
 * Output: results of calls to various functions testing topology
 *     creation
 *
 * Algorithm:
 *     1.  Build a 2-dimensional Cartesian communicator from
 *         MPI_Comm_world
 *     2.  Print topology information for each process
 *     3.  Use MPI_Cart_sub to build a communicator for each
 *         row of the Cartesian communicator
 *     4.  Carry out a broadcast across each row communicator
 *     5.  Print results of broadcast
 *     6.  Use MPI_Cart_sub to build a communicator for each
 *         column of the Cartesian communicator
 *     7.  Carry out a broadcast across each column communicator
 *     8.  Print results of broadcast
 *
 * Note: Assumes the number of processes, p, is a perfect square
 *
 * See Chap 7, pp. 121 & ff in PPMPI
 */
#include <stdio.h>
#include "mpi.h"
#include <math.h>

/*
mpi_cart_create()-> creates a communicator with virtual topology of processes. topology is a mechanism of addressing the processes. hence we map the processes in a 2d grid and each process is recognized by its 2d coordinate ( addressing ). it creates a communicator with group + context + topology !!
virtual topology means that there is no simple relation between how we address the process and how processes are physically connected (physical topology ) and addressed
*/

int main(int argc, char* argv[])
{	int p, my_rank, q, row_test, col_test, dim_sizes[2], wrap_around[2], reorder = 1, coordinates[2], my_grid_rank, grid_rank, free_coords[2];
	MPI_Comm  grid_comm;
    	MPI_Comm row_comm;
	MPI_Comm col_comm;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	q = (int) sqrt((double) p);
	// Let p= 16
	dim_sizes[0] = dim_sizes[1] = q;
	// wrap order[0] = 0/1 but wrap_order[1]=1 ( we need circular shift for send() / bcast() )
	wrap_around[0] = wrap_around[1] = 1;
	// divide MPI_COMM_WORLD into a communicator of grid-> grid_comm
	// grid_comm, row_comm, col_comm are all commuicators (having group, context AND TOPOLOGY to address individual processes )
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim_sizes, wrap_around, reorder, &grid_comm);
	MPI_Comm_rank(grid_comm, &my_grid_rank);
	MPI_Cart_coords(grid_comm, my_grid_rank, 2, coordinates);
	MPI_Cart_rank(grid_comm, coordinates, &grid_rank);
	printf("Process %d > my_grid_rank = %d, coords = (%d,%d), grid_rank = %d\n", my_rank, my_grid_rank, coordinates[0], coordinates[1], grid_rank);
	// after reordering, the processes are arranged i row major order. the mapig of processes in grid to the physical topology is optimised by reordering of processes in communicator. pg-123, 121- ppmpi-pacheco
	free_coords[0] = 0;		// means for new topology this coordinate will be fixed
	free_coords[1] = 1;		// this coordinate will vary
	// creates a communicator of new grid of only rows. Each row_grid is determined by fixed value of coord[0] and variable values of coord[1] ( i*j-> i is fixed and j varies ). All such row_comm have same name but sigify difference communicators ad collection of process !! This communicator created by paritioning row_comm communicator (The order of processes is same as that in grid_comm, no reordering here )
	// MPI_Cart_sub() works only when there is an existing MPI_Grid communicator
	MPI_Cart_sub(grid_comm, free_coords, &row_comm);
	if (coordinates[1] == 0)	// coordinate[] corresponding to grid_comm. starting of a new row_comm ( my_row_grid_rank= 0 for ith row corresponding to 0+(i*dim[1])th process in grid_comm )
		row_test = coordinates[0];		// row number
	else
		row_test = -1;
	int my_row_grid_rank;
	MPI_Comm_rank(row_comm, &my_row_grid_rank);
	// uderstand this concept!!
	MPI_Bcast(&row_test, 1, MPI_INT, 0, row_comm);			// bcast in row_comm corresponding to that process
	printf("Process %d > my_row_grid_rank: %d coords = (%d,%d), row_test = %d\n", my_rank, my_row_grid_rank, coordinates[0], coordinates[1], row_test);
	// same concept for column
	free_coords[0] = 1;
	free_coords[1] = 0;
	MPI_Cart_sub(grid_comm, free_coords, &col_comm);
	if (coordinates[0] == 0)
		col_test = coordinates[1];
	else
		col_test = -1;
	int my_col_grid_rank;
	MPI_Comm_rank(col_comm, &my_col_grid_rank);
	MPI_Bcast(&col_test, 1, MPI_INT, 0, col_comm);
	printf("Process %d > my_col_grid_rank: %d coords = (%d,%d), col_test = %d\n", my_rank, my_col_grid_rank, coordinates[0], coordinates[1], col_test);
	MPI_Finalize();
}

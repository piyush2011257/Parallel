#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include "rand.h"
#define ABORT ErrorExit(__LINE__)

void ErrorExit(int line)
{	printf("Error on line %d\n", line);
	exit(1);
}

int main(int argc, char *argv[])
{	double *actual_x /* Indexed by row number */, b_pivot, *computed_x/* Indexed by row number */, denominator, highest, multiplier, **my_a, *my_b, num, numerator, *pivot_elements, *pivot_row_elements, total;
	FILE       *fd, *result_file;
	int flops, *have_x/* Indexed by row number */, i, j, k, my_rank, num_processes, order, *perm_vector, pivot_row, *rp_map, *sent_yet, temp;
	MPI_Status status;
	flops = 0;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	/* Open the input file and get the order out. */
	fd = fopen("matrix.txt", "r");
	fscanf(fd, "%d", &order);
	/* Set up the permutation vector.  This tells us the index location, within my_a, that a given row in the matrix is stored at. Initially row 0 is at index 0, row 1 is at index 1 , etc... */
	perm_vector = (int *) malloc(sizeof(int) * order);
	if (perm_vector == NULL) ABORT;
	for (i = 0; i < order; i++) /* Rows */
		perm_vector[i] = i;
	/* Generate a mapping of indexes to store rows at to processes. Note this is conceptually NOT the same as mapping rows to processes!*/
	rp_map = (int *) malloc(sizeof(int) * order);
	if (rp_map == NULL) ABORT;
	for (i = 0; i < order; i++) /* Indexes to store rows at */
		rp_map[i] = i % num_processes;
	/* Allocate space for the parts of the coefficient matrix and result vector this process is responsible for storing.  The coefficient matrix will be stored as a staggered array.*/
	my_a = (double **) malloc(sizeof(double *) * order);
	if (my_a == NULL) ABORT;
	for (i = 0; i < order; i++) /* Indexes to store rows at */
	{	if (rp_map[i] == my_rank)
		{	my_a[i] = (double *) malloc(sizeof(double) * order);
			if (my_a[i] == NULL) ABORT;
		}
		else my_a[i] = NULL;
	}
	my_b = (double *)  malloc(sizeof(double) * order);
	if (my_b == NULL) ABORT;
	/* Read in my parts of the coefficient matrix. */
	for (i = 0; i < order; i++) /* Columns */
	{	for (j = 0; j < order; j++) /* Rows */
		{	fscanf(fd, "%lf", &num);
			if (rp_map[perm_vector[j]] == my_rank)
				my_a[perm_vector[j]][i] = num;
		}
	}
	/* Read in my parts of the result vector. */
	for (i = 0; i < order; i++) /* Rows */
	{	fscanf(fd, "%lf", &num);
		if (rp_map[perm_vector[i]] == my_rank)
			my_b[perm_vector[i]] = num;
	}
	/* Allocate space for the actual solution vector.  This will be stored by all processes, but will be used, for solution verification, only by the processor which ultimately ends up being responsible for row 0. */
	actual_x = (double *) malloc(sizeof(double) * order);
	if (actual_x == NULL) ABORT;
	/* Read in the actual solution vector. */
	for (i = 0; i < order; i++) /* Rows */
		fscanf(fd, "%lf", &actual_x[i]);
	fclose(fd);
	/* Perform Gaussian Elimination with partial pivoting. Allocate space for the pivot elements.  This array will be indexed by row #, not the index # a row is stored at. */
	pivot_elements = (double *) malloc(sizeof(double) * order);
	if (pivot_elements == NULL) ABORT;
	pivot_row_elements = (double *) malloc(sizeof(double) * order);
	if (pivot_row_elements == NULL) ABORT;	
	for (i = 0; i < order - 1; i++) /* Rows */
	{	/* Send the pivot elements I own to all other processors. */
		for (j = i; j < order; j++) /* Rows */
		{	if (rp_map[perm_vector[j]] == my_rank)
			{	for (k = 0; k < num_processes; k++) /* Processes */
				{	if (k != my_rank)
					{	 /* i used as columns here. */
						MPI_Send(&my_a[perm_vector[j]][i], 1, MPI_DOUBLE, k, j, MPI_COMM_WORLD);
					}
				}
			}
		}
		/* Fill in the pivot elements array. */
		for (j = i; j < order; j++) /* Rows */
		{	if (rp_map[perm_vector[j]] == my_rank)
			{	/* i used as columns here. */
				pivot_elements[j] = my_a[perm_vector[j]][i];
				if (pivot_elements[j] < 0.0) pivot_elements[j] *= -1.0;
			}
			else
			{	MPI_Recv(&pivot_elements[j], 1, MPI_DOUBLE, rp_map[perm_vector[j]], j, MPI_COMM_WORLD, &status);
				if (pivot_elements[j] < 0.0) pivot_elements[j] *= -1.0;
			}
		}
		/* Determine what the pivot row is. */
		highest = 0.0;
		for (j = i; j < order; j++) /* Rows */
		{	if (pivot_elements[j] > highest)
			{	highest = pivot_elements[j];
				pivot_row = j;
			}
		}
		/* Now, switch row i with row pivot_row. */
		temp = perm_vector[i];
		perm_vector[i] = perm_vector[pivot_row];
		perm_vector[pivot_row] = temp;
		/* If my process owns row i (the pivot row), send it to the other processes.  Otherwise, receive it.  Do the same for the result vector element on the pivot row. */
		if (rp_map[perm_vector[i]] == my_rank)
		{	for (j = 0; j < num_processes; j++) /* Processes */
			{	if (j != my_rank)
				{	MPI_Send(my_a[perm_vector[i]], order, MPI_DOUBLE, j, i, MPI_COMM_WORLD);
					MPI_Send(&my_b[perm_vector[i]], 1, MPI_DOUBLE, j, i, MPI_COMM_WORLD);
				}
			}
			/* Copy the pivot row into pivot_row_elements. */
			for (j = 0; j < order; j++) /* Columns */
				pivot_row_elements[j] = my_a[perm_vector[i]][j];
			b_pivot = my_b[perm_vector[i]];
		}
		else	
		{	MPI_Recv(pivot_row_elements, order, MPI_DOUBLE, rp_map[perm_vector[i]], i, MPI_COMM_WORLD, &status);
			MPI_Recv(&b_pivot, 1, MPI_DOUBLE, rp_map[perm_vector[i]], i, MPI_COMM_WORLD, &status);
		}
		/* Subtract a multiple of the pivot row from the rows below it which I own.  This should be done such that column i (of the rows I own) becomes 0 after the subtraction. */
		for (j = i + 1; j < order; j++) /* Rows */
		{	if (rp_map[perm_vector[j]] == my_rank)
			{	multiplier = my_a[perm_vector[j]][i] / pivot_row_elements[i];
				for (k = 0; k < order; k++)
				{	my_a[perm_vector[j]][k] -= multiplier * pivot_row_elements[k];
					flops += 2;
				}
				my_b[perm_vector[j]] -= multiplier * b_pivot;
				flops += 2;
			}
		}
	}
	/* We now have an upper triangular system.  This must now be solved using back substitution. */
	/* Allocate space for and initialize to false every entry in the array that indicates which x values we have.*/
	have_x = (int *) malloc(sizeof(int) * order);
	if (have_x == NULL) ABORT;
	for (i = 0; i < order; i++)
		have_x[i] = 0;
	/* Allocate space for the data structure which indicates which processes we've sent a computed x value to.*/
	sent_yet = (int *) malloc(sizeof(int) * num_processes);
	if (sent_yet == NULL) ABORT;
	/* Allocate space for the array which holds computed x values.*/
	computed_x = (double *) malloc(sizeof(double) * order);
	if (computed_x == NULL) ABORT;
	/* Perform back substitution. */
	for (i = order - 1; i >= 0; i--) /* Rows */
	{	if (rp_map[perm_vector[i]] == my_rank)
		{	total = 0.0;
			for (j = i + 1; j < order; j++) /* X values */
			{	if (!have_x[j])
				{	 MPI_Recv(&computed_x[j], 1, MPI_DOUBLE, rp_map[perm_vector[j]], j, MPI_COMM_WORLD, &status);
					have_x[j] = 1;
				}
				total += computed_x[j] * my_a[perm_vector[i]][j];
				flops += 2;
			}
			computed_x[i] = (my_b[perm_vector[i]] - total) / my_a[perm_vector[i]][i];
			flops++;
			have_x[i] = 1;
			/* Send this value to the other processors. */
			for (j = 0; j < num_processes; j++)
				sent_yet[j] = 0;
			for (j = i - 1; j >= 0; j--) /* Rows */
			{	if (rp_map[perm_vector[j]] != my_rank)
				{	if (!sent_yet[rp_map[perm_vector[j]]])
					{	MPI_Send(&computed_x[i], 1, MPI_DOUBLE, rp_map[perm_vector[j]], i, MPI_COMM_WORLD);
						sent_yet[rp_map[perm_vector[j]]] = 1;
					}
				}
			}
		}
	}
	/* Write the computed values of x to disk, if we are the process that ended up being responsible for row 0. Also, perform the solution 	verfication step. */
	numerator   = 0.0;
	denominator = 0.0;
	if (rp_map[perm_vector[0]] == my_rank)
	{	result_file = fopen("result.txt", "w");
		fprintf(result_file, "Computed solution vector:\n\n");
		for (i = 0; i < order; i++)
		{	fprintf(result_file, "%lf\n", computed_x[i]);
			numerator   += pow((actual_x[i] - computed_x[i]), 2.0);
			denominator += pow(actual_x[i], 2.0);
		}
		fclose(result_file);
		numerator   = sqrt(numerator);
		denominator = sqrt(denominator);
		printf("\n\nError in computed solution: %e\n\n", numerator / denominator);
	}
	printf("Process %d performed %d floating point operations.\n", my_rank, flops);
	MPI_Finalize();
}

#ifndef LINALG_H
#define LINALG_H

#include <vector>
#include <cmath>

namespace linalg {

	typedef std::vector<float> vector;				// vector of floats
	typedef std::vector<std::vector<float>> matrix;	// matrix

	vector make_vector(int length);
	matrix make_matrix(int n, int m);	// number of rows, mumber of columns

	// your standard matrix utilities
	matrix mult_matrix(matrix a, matrix b);
	matrix add_matrix(matrix a, matrix b);
	matrix scale_matrix(matrix m, float f);
	vector add_vector(vector a, vector b);
	vector scale_vector(vector v, float f);
	matrix as_row_matrix(vector v);
	matrix as_column_matrix(vector v);
	// flattens out a matrix (row by row first)
	vector as_vector(matrix m);
	vector apply(vector v, float (*func)(float));
}

#endif
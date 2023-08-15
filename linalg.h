#ifndef LINALG_H
#define LINALG_H

#include <vector>
#include <cmath>

namespace linalg {

	// this is a header file describing a namespace of linear algebra utilities
	
	const float EPS = 1e-5;

	typedef std::vector<float> vector;				// vector of floats
	typedef std::vector<std::vector<float>> matrix;	// matrix

	vector make_vector(int length);
	matrix make_matrix(int n, int m);	// number of rows, mumber of columns

	// your standard matrix utilities
	matrix mult_matrix(matrix a, matrix b);
	matrix add_matrix(matrix a, matrix b);
	matrix scale_matrix(matrix m, float f);
	float det_matrix(matrix m); 				//matrix determinant
	vector mult_vector(vector a, vector b); //dot product
	matrix outer_matrix(vector a, vector b); //outer product
	vector add_vector(vector a, vector b);
	vector scale_vector(vector v, float f);
	int argmax_vector(vector v);
	matrix as_row_matrix(vector v);
	matrix as_column_matrix(vector v);
	// flattens out a matrix (row by row first)
	vector as_vector(matrix m);
	vector apply(vector v, float (*func)(float));
	float vector_length(vector v);
	float vector_sum(vector v);
}

#endif
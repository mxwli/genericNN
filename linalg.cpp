#include "linalg.h"

#include <cassert>

#define DEBUG_LINALG

using namespace linalg;

vector linalg::make_vector(int length) {
    return vector(length);
}
matrix linalg::make_matrix(int n, int m) {
    return matrix(n, vector(m));
}

matrix linalg::mult_matrix(matrix a, matrix b) {
    #ifdef DEBUG_LINALG
        assert(a[0].size() == b.size()); // assert postcondition
    #endif
    matrix ret = matrix(a.size(), vector(b[0].size(), 0));
    for (int i = 0; i < a.size(); i++)
        for (int i3 = 0; i3 < b.size(); i3++)
            for (int i2 = 0; i2 < b[0].size(); i2++)
                ret[i][i2] += a[i][i3] * b[i3][i2];
    return ret;
}
matrix linalg::add_matrix(matrix a, matrix b) {
    #ifdef DEBUG_LINALG
        assert(a.size() == b.size() && a[0].size() == b[0].size());
    #endif
    matrix ret = a;
    for(int i = 0; i < a.size(); i++)
        for(int i2 = 0; i2 < a[i].size(); i2++)
        ret[i][i2] += b[i][i2];
    return ret;
}
matrix linalg::scale_matrix(matrix m, float f) {
    matrix ret = m;
    for(auto& row : ret)
        for(auto& col: row)
            col *= f;
    return ret;
}

float linalg::det_matrix(matrix m) {
	#ifdef DEBUG_LINALG
		assert(m.size() == m[0].size());
	#endif
	float ret = 1;
	for(std::size_t i = 0; i < m.size(); i++) {
		if(std::abs(m[i][i])<EPS) {
			bool foundflag = false;
			for(std::size_t i2 = i+1; i2 < m.size(); i2++) if(std::abs(m[i2][i]) > EPS) {
				swap(m[i], m[i2]);
				foundflag = true;
				break;
			}
			if(foundflag)
				ret *= -1; // swap operation negates determinant
			else
				return 0; // rows are linearly dependent
		}
		for(std::size_t i2 = i+1; i2 < m.size(); i2++)
			m[i2] = add_vector(m[i2], scale_vector(m[i], -m[i2][i]/m[i][i]));
	}
	for(std::size_t i = 0; i < m.size(); i++)
		ret *= m[i][i];
	return ret;
}
vector linalg::mult_vector(vector a, vector b) {
    #ifdef DEBUG_LINALG
        assert(a.size() == b.size());
    #endif
    vector ret = a;
    for(int i = 0; i < a.size(); i++) ret[i] = a[i]*b[i];
    return ret;
}
matrix linalg::outer_matrix(vector a, vector b) {
    matrix ret = make_matrix(a.size(), b.size());
    for(int i = 0; i < a.size(); i++)
        for(int i2 = 0; i2 < b.size(); i2++)
            ret[i][i2] = a[i]*b[i2];
    return ret;
}
vector linalg::add_vector(vector a, vector b) {
    #ifdef DEBUG_LINALG
        assert(a.size() == b.size());   // assert postcondition
    #endif
    vector ret(a.size());
    for(int i = 0; i < a.size(); i++) ret[i] = a[i] + b[i];
    return ret;
}
vector linalg::scale_vector(vector v, float f) {
    vector ret = v;
    for(auto& i: ret)
        i *= f;
    return ret;
}
int linalg::argmax_vector(vector v) {
    int ret = 0;
    for(int i = 0; i < v.size(); i++) if(v[i] > v[ret]) ret = i;
    return ret;
}

matrix linalg::as_row_matrix(vector v) {
	return matrix(1, v);
}
matrix linalg::as_column_matrix(vector v) {
	matrix ret = matrix(v.size(), vector(1,0));
    for(int i = 0; i < v.size(); i++) ret[i][0] = v[i];
	return ret;
}

vector linalg::as_vector(matrix m) {
    vector ret;
    for(const auto& row: m)
        ret.insert(ret.end(), row.begin(), row.end());
    return ret;
}

vector linalg::apply(vector v, float(*func)(float)) {
    vector ret(v.size());
    for(int i = 0; i < v.size(); i++) ret[i] = func(v[i]);
    return ret;
}

float linalg::vector_length(vector v) {
    float ret = 0;
    for(const auto& elem: v) ret += elem*elem;
    return std::sqrt(ret);
}
float linalg::vector_sum(vector v) {
    float ret = 0;
    for(const auto& elem: v) ret += elem;
    return ret;
}
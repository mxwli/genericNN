#include "NN.h"

#include <cassert>

#define DEBUG_NN

using namespace NN;

float NN::rand_unif(float L, float H) {
	std::uniform_real_distribution ureal(L, H);
	return ureal(mers_twister);
}
float NN::rand_gaus(float M, float STD) {
	std::normal_distribution norm(M, STD);
	return norm(mers_twister);
}

float NN::func_relu(float x) { return x < 0 ? 0 : x; }
float NN::func_relu_derivative(float x) {return x<0?0:1;}
float NN::func_logistic(float x) { return x / std::sqrt(x * x + 1); }
float NN::func_logistic_derivative(float x) {return 1/(x*x+1)/std::sqrt(x*x+1);}
float NN::func_linear(float x) {return x;}
float NN::func_linear_derivative(float x) {return 1;}

NN::layer::layer(int sze, activation_type act) {
	size = sze;
	activation = act;
}

NN::network::network(std::vector<layer> lyrs, double rate) {
	layers = lyrs;
	learning_rate = rate;
}

network_compiled NN::compile_network(network net) {
	network_compiled comp;
	comp.learning_rate = net.learning_rate;

	layer_compiled prev;

	for(const auto& i: net.layers) {
		layer_compiled i_comp;
		i_comp.layer = linalg::make_vector(i.size);
		i_comp.weights = linalg::make_matrix(i.size, prev.layer.size());
		i_comp.biases = linalg::make_vector(i.size);
		if(i.activation == activation_type::linear) {
			i_comp.activation = func_linear;
			i_comp.activation_derivative = func_linear_derivative;
			// no initialization for the specific elements, all set to zero
		}
		if(i.activation == activation_type::logistic) {
			i_comp.activation = func_logistic;
			i_comp.activation_derivative = func_logistic_derivative;
			// we use normalized xavier weight and bias initialization
			float bounds = std::sqrt(6.0/(i.size+prev.layer.size()));
			for(int x = 0; x < i.size; x++)
				for(int y = 0; y < prev.layer.size(); y++)
					i_comp.weights[x][y] = rand_unif(-bounds, bounds);
			for(int x = 0; x < i.size; x++)
				i_comp.biases[x] = rand_unif(-bounds, bounds);
		}
		if(i.activation == activation_type::relu) {
			i_comp.activation = func_relu;
			i_comp.activation_derivative = func_relu_derivative;
			// we use He for weight and bias initialization
			float stdeviation = std::sqrt(2.0/i.size);
			for(int x = 0; x < i.size; x++)
				for(int y = 0; y < prev.layer.size(); y++)
					i_comp.weights[x][y] = rand_gaus(0, stdeviation);
			for(int x = 0; x < i.size; x++)
				i_comp.biases[x] = rand_unif(0, stdeviation);
		}
		comp.layers.push_back(i_comp);
		prev = i_comp;
	}
	return comp;
}

linalg::vector NN::run_network(network_compiled& ret, linalg::vector input) {
	#ifdef DEBUG_NN
		assert(ret.layers[0].layer.size() == input.size());
	#endif
	for(int i = 0; i < input.size(); i++) {
		ret.layers[0].layer[i] = input[i];
	}
	ret.layers[0].layer = linalg::apply(
		linalg::add_vector(ret.layers[0].layer, ret.layers[0].biases),
		ret.layers[0].activation
	);
	for(int i = 1; i < ret.layers.size(); i++)
		ret.layers[i].layer = linalg::apply(
			linalg::add_vector(
				linalg::as_vector(linalg::mult_matrix(
					ret.layers[i].weights,
					linalg::as_column_matrix(ret.layers[i-1].layer)
				)),
				ret.layers[i].biases
			),
			ret.layers[i].activation
		);
	return ret.layers.back().layer;
}


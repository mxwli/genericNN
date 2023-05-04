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

float NN::func_relu(float x) { return x < 0 ? (0.01*x) : x; }
float NN::func_relu_derivative(float x) {return x<0?0.01:1;}
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

NN::gradient_layer::gradient_layer(layer_compiled of) {
	weight_grad = linalg::make_matrix(of.weights.size(), of.weights[0].size());
	bias_grad = linalg::make_vector(of.biases.size());
}
NN::gradient::gradient(network_compiled of) {
	grad_layers = std::vector<gradient_layer>(of.layers.size());
	for(int i = 0; i < of.layers.size(); i++)
		grad_layers[i] = gradient_layer(of.layers[i]);
}

gradient_layer NN::grad_layer_add(gradient_layer a, gradient_layer b) {
	#ifdef DEBUG_NN
		assert(a.weight_grad.size() == b.weight_grad.size());
		assert(a.weight_grad[0].size() == b.weight_grad[0].size());
		assert(a.bias_grad.size() == b.bias_grad.size());
	#endif
	gradient_layer ret = a;
	ret.weight_grad = linalg::add_matrix(a.weight_grad, b.weight_grad);
	ret.bias_grad = linalg::add_vector(a.bias_grad, b.bias_grad);
	return ret;
}
gradient_layer NN::grad_layer_scale(gradient_layer a, float f) {
	gradient_layer ret = a;
	ret.weight_grad = linalg::scale_matrix(ret.weight_grad, f);
	ret.bias_grad = linalg::scale_vector(ret.bias_grad, f);
	return ret;
}
gradient NN::grad_add(gradient a, gradient b) {
	#ifdef DEBUG_NN
		assert(a.grad_layers.size() == b.grad_layers.size());
	#endif
	gradient ret = a;
	for(int i = 0; i < ret.grad_layers.size(); i++)
		ret.grad_layers[i] = grad_layer_add(a.grad_layers[i], b.grad_layers[i]);
	return ret;
}
gradient NN::grad_scale(gradient a, float f) {
	gradient ret = a;
	for(int i = 0; i < ret.grad_layers.size(); i++)
		ret.grad_layers[i] = grad_layer_scale(a.grad_layers[i], f);
	return ret;
}

gradient NN::back_propagate(network_compiled net, linalg::vector desired) {
	#ifdef DEBUG_NN
		assert(net.layers.back().layer.size() == desired.size());
	#endif
	gradient ret(net);
	linalg::vector diff = linalg::scale_vector(
		linalg::add_vector(
			net.layers.back().layer,
			linalg::scale_vector(desired,-1)
		),
		2
	);
	for(int i = ((int)net.layers.size())-1; i > 0; i--) {
		diff = linalg::mult_vector(
			diff,
			linalg::apply(
				net.layers[i].layer,
				net.layers[i].activation_derivative
			)
		);
		ret.grad_layers[i].bias_grad = diff;
		ret.grad_layers[i].weight_grad = linalg::outer_matrix(
			diff,
			net.layers[i-1].layer
		);
		diff = linalg::as_vector(
			linalg::mult_matrix(
				linalg::as_row_matrix(diff),
				net.layers[i].weights
			)
		);
	}
	return ret;
}
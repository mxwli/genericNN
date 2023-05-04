#ifndef NN_H
#define NN_H

#include <vector>
#include <random>
#include "linalg.h"

namespace NN {

	inline std::mt19937_64 mers_twister;

	// generates a random number between two bounds
	float rand_unif(float L, float H);
	// generates a random number in a gaussian distr with a mean
	// and standard deviation
	float rand_gaus(float M, float STD);

	// your standard activation functions
	//note that this implementation is a leaky relu
	float func_relu(float x);
	float func_relu_derivative(float x);
	float func_logistic(float x);
	float func_logistic_derivative(float x);
	float func_linear(float x);
	float func_linear_derivative(float x);

	typedef enum activation_type {
		relu, logistic, linear
	} activation_type;

	typedef struct layer {
		int size;
		activation_type activation;
		layer(int sze, activation_type act);
		// makes layer with corresponding size and activation function
	} layer;

	typedef struct layer_compiled {
		linalg::vector layer;
		linalg::matrix weights;	//input weights
		linalg::vector biases;	//input biases
		float (*activation)(float);
		float (*activation_derivative)(float);
		//after inputs are calculated, the resulting layer is put
		//through this function to get the final layer state
	} layer_compiled;

	typedef struct network {
		std::vector<layer> layers;
		double learning_rate;	// rate at which weights/biases are modified
		network(std::vector<layer> lyrs, double rate);
	} network;

	typedef struct network_compiled {
		std::vector<layer_compiled> layers;
		double learning_rate;	// rate at which weights/biases are modified
	} network_compiled;

	network_compiled compile_network(network net);

	// activations are kept in net
	linalg::vector run_network(network_compiled& net, linalg::vector input);

	float mean_squared_error(linalg::vector output, linalg::vector desired);

	typedef struct gradient_layer {
		linalg::matrix weight_grad;
		linalg::vector bias_grad;
		gradient_layer(layer_compiled of);	// what layer is this a gradient of?
	} gradient_layer;

	typedef struct gradient {
		std::vector<gradient_layer> grad_layers;
		gradient(network_compiled of);		// what network is this a gradient of?
	} gradient;

	gradient_layer grad_layer_add(gradient_layer a, gradient_layer b);
	gradient_layer grad_layer_scale(gradient_layer a, float f);
	gradient grad_add(gradient a, gradient b);
	gradient grad_scale(gradient a, float f);

	gradient back_propagate(network_compiled net, linalg::vector desired);

	void apply_gradient(network_compiled& net, gradient grad);
}

#endif
#include <iostream>
#include "NN.h"
using namespace std;

int main() {
	vector<NN::layer> vec{
		NN::layer(5, NN::activation_type::linear),
		NN::layer(1, NN::activation_type::logistic),
	};
	NN::network net(vec, 0.001);
	NN::network_compiled net_comp = NN::compile_network(net);
	for(const auto layer: net_comp.layers) {
		cout << "weights:\n";
		for(const auto row: layer.weights) {
			for(const auto col: row) {
				cout << col << " ";
			}
			cout << "\n";
		}
		cout << "biases:\n";
		for(const auto col: layer.biases) {
			cout << col << "\n";
		}
		cout << "\n-----------\n";
	}
	cout << "network printed!\n\n";
	linalg::vector input{-2, -1, 0, 1, 2};
	linalg::vector output = NN::run_network(net_comp, input);
	for(int i = 0; i < output.size(); i++) {
		cout << output[i] << "\n";
	}
}
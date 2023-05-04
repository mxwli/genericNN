#include <iostream>
#include "NN.h"
using namespace std;

int main() {
	vector<NN::layer> vec{
		NN::layer(2, NN::activation_type::linear),
		NN::layer(3, NN::activation_type::relu),
		NN::layer(3, NN::activation_type::relu),
		NN::layer(1, NN::activation_type::logistic),
	};
	NN::network net(vec, 0.1);
	NN::network_compiled net_comp = NN::compile_network(net);
	vector<linalg::vector> train_data{
		linalg::vector{0,0},
		linalg::vector{0,1},
		linalg::vector{1,0},
		linalg::vector{1,1}
	};
	vector<linalg::vector> target_data{
		linalg::vector{0},
		linalg::vector{1},
		linalg::vector{1},
		linalg::vector{0}
	};
	for(int epoch = 0; epoch < 5000; epoch++) {
		NN::gradient total(net_comp);
		float MSE = 0;
		for(int i = 0; i < 4; i++) {
			linalg::vector output = NN::run_network(net_comp, train_data[i]);
			MSE += NN::mean_squared_error(target_data[i], output);
			total = NN::grad_add(
				total,
				NN::back_propagate(net_comp, target_data[i])
			);
		}
		total = NN::grad_scale(total, -net_comp.learning_rate/4);
		cout << MSE << "\n";
		NN::apply_gradient(net_comp, total);
	}
	while(true) {
		float a, b;
		cin >> a >> b;
		cout << (NN::run_network(net_comp, {a, b})[0]) << "\n";
	}
}
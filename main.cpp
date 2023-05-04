#include <iostream>
#include "NN.h"
#include <raylib.h>
using namespace std;

vector<NN::layer> vec{
	NN::layer(2, NN::activation_type::linear),
	NN::layer(10, NN::activation_type::relu),
	NN::layer(10, NN::activation_type::relu),
	NN::layer(1, NN::activation_type::logistic)
};
NN::network_compiled net_comp = NN::compile_network(NN::network(vec, 0.1));

int main() {
	InitWindow(500, 500, "network test");
	SetTargetFPS(60);
	while(!WindowShouldClose()) {
		if(IsKeyPressed(KEY_SPACE)) {
			net_comp = NN::compile_network(NN::network(vec, 0.1));
		}
	}
}
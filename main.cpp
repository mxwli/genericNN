#include <iostream>
#include "NN.h"
#include <raylib.h>
using namespace std;

vector<NN::layer> vec{
	NN::layer(2, NN::activation_type::linear),
	NN::layer(32, NN::activation_type::relu),
	NN::layer(32, NN::activation_type::relu),
	NN::layer(32, NN::activation_type::relu),
	NN::layer(1, NN::activation_type::logistic)
};
NN::network net(vec, 0.1);
NN::network_compiled net_comp = NN::compile_network(net);

int grid[10][10];

int main() {
	InitWindow(500, 500, "network test");
	SetTargetFPS(20);
	while(!WindowShouldClose()) {
		if(IsKeyPressed(KEY_SPACE)) {
			net_comp = NN::compile_network(net);
		}
		if(IsMouseButtonPressed(0)) {
			grid[GetMouseX()/50][GetMouseY()/50] = !grid[GetMouseX()/50][GetMouseY()/50];
		}
		vector<linalg::vector> X, y;
		for(int i = 0; i < 10; i++) {
			for(int i2 = 0; i2 < 10; i2++) {
				linalg::vector newX{(float)i, (float)i2};
				linalg::vector newY{(float)grid[i][i2]};
				X.push_back(newX);
				y.push_back(newY);
			}
		}
		NN::automatic_fit(net_comp, X, y, 1);
		BeginDrawing();
		for(int i = 0; i < 10; i++) {
			for(int i2 = 0; i2 < 10; i2++) {
				float predict = NN::run_network(net_comp, {(float)i, (float)i2})[0];
				if(grid[i][i2]) {
					if(predict>0.5)
						DrawRectangle(i*50,i2*50,50,50,DARKGREEN);
					else
						DrawRectangle(i*50,i2*50,50,50,BROWN);
				}
				else {
					if(predict>0.5)
						DrawRectangle(i*50,i2*50,50,50,ORANGE);
					else
						DrawRectangle(i*50,i2*50,50,50,GREEN);
				}
			}
		}
		DrawFPS(1,1);
		EndDrawing();
	}

	CloseWindow();
}
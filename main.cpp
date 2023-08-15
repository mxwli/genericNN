#include <iostream>
#include <raylib.h>
#include <raymath.h>
#include <box2d/box2d.h>
#include <fstream>
#include <sstream>
#include "game.h"
#include "NN.h"
#include "reinforcement.h"
using namespace std;
#define SCREENWIDTH 700
#define SCREENHEIGHT 700

NN::network configuration({
	NN::layer(7, NN::activation_type::linear),
	NN::layer(10, NN::activation_type::relu),
	NN::layer(3, NN::activation_type::logistic)
}, 0.1);
NN::network_compiled net = NN::compile_network(configuration);
void load_from_file(string file_name) {
	ifstream file_in(file_name,ifstream::in);
	net = NN::read_network_compiled(file_in);
	file_in.close();
}

int main() {

	int iterations = 100, batch_size = 150, N_ticks = 300;
	float decay = 0.995;
	stringstream sstream;
	sstream << "saves/C-" << iterations << "-" << batch_size << "-" << N_ticks << "-"
		<< decay << "-";
	string save_file;
	sstream >> save_file;

	load_from_file("saves/C-750-128-150-0.995-record");
	//reinforce::REINFORCE(net,iterations, batch_size, N_ticks, decay, save_file);
	//load_from_file(save_file);

	b2Vec2 gravity(0, 3);
	GS::game current_game(gravity, 0, 0);
	Vector2 camera_pos = {-50, -50};
	InitWindow(SCREENWIDTH, SCREENHEIGHT, "");
	SetTargetFPS(15);
	while(!WindowShouldClose()) {
		linalg::vector state = GS::get_state(current_game);
		state[2] = state[2]-IsKeyDown(KEY_D)+IsKeyDown(KEY_A);
		state[3] = state[3]-IsKeyDown(KEY_S)+IsKeyDown(KEY_W);
		linalg::vector network_response = NN::run_network(net, state);
		std::vector<int> interpreted_action(network_response.size());
		for(int i = 0; i < network_response.size(); i++)
			interpreted_action[i] = network_response[i]>NN::rand_unif(-1,1);
		interpreted_action = std::vector<int>{
			interpreted_action[0],
			interpreted_action[1],
			interpreted_action[1],
			interpreted_action[0],
			interpreted_action[2]
		};
		GS::game_step(current_game, interpreted_action);
		BeginDrawing();
		ClearBackground(BLACK);
		current_game.player.draw_drone(camera_pos, 7, WHITE);
		EndDrawing();
	}
	CloseWindow();
}
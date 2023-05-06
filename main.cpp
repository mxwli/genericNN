#include <iostream>
#include <raylib.h>
#include <raymath.h>
#include <box2d/box2d.h>
#include <fstream>
#include "game.h"
#include "NN.h"
#include "reinforcement.h"
using namespace std;
#define SCREENWIDTH 700
#define SCREENHEIGHT 700
#define GAMEWIDTH 10
#define GAMEHEIGHT 10
#define ZOOM 70

b2Vec2 gravity(0, 3);

NN::network config({
	NN::layer(4, NN::activation_type::linear),
	NN::layer(10, NN::activation_type::relu),
	NN::layer(6, NN::activation_type::logistic)
}, 0.001);
NN::network_compiled net = NN::compile_network(config);

float run_epoch(NN::network_compiled& net, int samples, int N_ticks) {
	reinforce::playthroughs current_playthroughs;
	float totalreward = 0;
	for(int i = 0; i < samples; i++) {
		current_playthroughs.new_trajectory();
		GS::game current_game(gravity);
		
		for(int tick = 0; tick < N_ticks; tick++) {
			linalg::vector state{
				current_game.player.body->GetLinearVelocity().x,
				current_game.player.body->GetLinearVelocity().y,
				current_game.player.body->GetAngularVelocity(),
				current_game.player.body->GetAngle()
			};
			linalg::vector action = NN::run_network(net, state);
			for(int k = 0; k < action.size(); k++)
				action[k] = action[k] > NN::rand_unif(-1, 1)?1:-1;
			
			std::vector<int> translated_action(action.size());
			for(int k = 0; k < action.size(); k++)
				translated_action[k] = action[k] > 0;
			current_game.player.apply_thrusters(translated_action);
			current_game.world.Step(1.0/30, 2, 2);

			float reward = 0;
			reward += 1.6-current_game.player.body->GetLinearVelocity().LengthSquared();
			reward += 0.6-current_game.player.body->GetAngularVelocity();
			current_playthroughs.append_SAR(state, action, reward);
			totalreward += reward;
		}
	}
	current_playthroughs.gradient_descent(net, 0.995);
	return totalreward/samples/N_ticks;
}

int main() {

	cout << "0 for train from scratch, 1 for train from save, 2 for display from save\n";
	int choice_number; cin >> choice_number;
	if(choice_number == 0 || choice_number == 1) {
		if(choice_number == 1) {
			ifstream file_in("saves/A-500-50-180.txt", ifstream::in);
			net = NN::read_network_compiled(file_in);
			file_in.close();
		}
		float maxreward = -100;
		for(int i = 0; i < 1000; i++) {
			cout << "epoch " << i << "\treward: ";
			float tot = run_epoch(net, 6, 150);
			cout << tot << "\n";
			if(tot > maxreward+0.1) {
				ofstream file_out("saves/A-500-50-180.txt", ofstream::out);
					NN::write_network_compiled(net, file_out);
				file_out.close();
				maxreward = tot;
			}
		}

		cout << "Press <Enter> to begin visuals\n";
		string s; cin >> s;
	}
	else {
		ifstream file_in("saves/A-500-50-180.txt", ifstream::in);
		net = NN::read_network_compiled(file_in);
		file_in.close();
	}


	GS::game frame(gravity);
	InitWindow(SCREENWIDTH, SCREENHEIGHT, "");
	int FPS = 30;
	int velocityIterations = 6;
	int positionIterations = 2;

	SetTargetFPS(FPS);

	while(!WindowShouldClose()) {
		linalg::vector state{
			frame.player.body->GetLinearVelocity().x,
			frame.player.body->GetLinearVelocity().y,
			frame.player.body->GetAngularVelocity(),
			frame.player.body->GetAngle()
		};
		linalg::vector action = NN::run_network(net, state);
		for(int k = 0; k < action.size(); k++)
			action[k] = action[k] > NN::rand_unif(-1, 1)?1:-1;
		
		std::vector<int> translated_action(action.size());
		for(int k = 0; k < action.size(); k++)
			translated_action[k] = action[k] > 0;
		frame.player.apply_thrusters(translated_action);
		frame.world.Step(1.0/FPS, velocityIterations, positionIterations);
		BeginDrawing();
		ClearBackground(BLACK);
		for(int i = 1; i < frame.player.ends.size(); i++) {
			DrawLineV(
				Vector2Scale(
					frame.player.local_to_global(frame.player.ends[i-1]),
					ZOOM
				),
				Vector2Scale(
					frame.player.local_to_global(frame.player.ends[i]),
					ZOOM
				),
				WHITE
			);

		}
		EndDrawing();
	}
}
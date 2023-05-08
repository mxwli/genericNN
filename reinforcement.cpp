#include "reinforcement.h"

#include <iostream>
#include <fstream>

#include "game.h"

using namespace reinforce;

void reinforce::playthroughs::new_trajectory() {
	list.push_back(trajectory());
}
void reinforce::playthroughs::append_SAR(
	linalg::vector state, linalg::vector action, float reward) {

	state_action_reward push;
	push.state = state; push.action = action; push.reward = reward;
	list.back().list.push_back(push);
}
void reinforce::playthroughs::gradient_descent(
	NN::network_compiled& net, float discount_factor) {
	
	NN::gradient total_weighted_grad(net);
	float attribution_mean = 0;
	float attribution_STD = 0;
	float attribution_N = 0;

	for(auto& traj: list) {
		float accumulation = 0;
		for(int i = (int)(traj.list.size())-1; i >= 0; i--) {
			// calculate state-action attributions and mean
			accumulation *= discount_factor;
			accumulation += traj.list[i].reward;
			traj.list[i].attribution = accumulation;
			attribution_mean += accumulation;
			attribution_N += 1;
		}
	}
	attribution_mean /= attribution_N;
	for(auto& traj: list) {
		for(int i = (int)(traj.list.size())-1; i >= 0; i--) {
			// calculate standard deviation
			attribution_STD +=
				(traj.list[i].attribution-attribution_mean)*
				(traj.list[i].attribution-attribution_mean);
		}
	}
	// calculate standard deviation
	attribution_STD =
		std::sqrt(attribution_STD/attribution_N);

	for(auto& traj: list) {
		for(int i = (int)(traj.list.size())-1; i >= 0; i--) {
			NN::run_network(net, traj.list[i].state);
			total_weighted_grad = NN::grad_add(
				total_weighted_grad,
				NN::grad_scale(
					NN::back_propagate(net, traj.list[i].action),
					//traj.list[i].attribution
					(traj.list[i].attribution-attribution_mean)/attribution_STD
				)
			); // this aims to make the action more or less likely
		}
	}
	total_weighted_grad = NN::grad_scale(
		total_weighted_grad,
		-net.learning_rate/attribution_N
	);
	NN::apply_gradient(net, total_weighted_grad);
}

/*
appraises a given state from GS::game_step or GS::get_state
*/
float appraisal(linalg::vector state) {
	float x_pos = state[2], y_pos = state[3];
	return 2-std::sqrt(x_pos*x_pos+y_pos*y_pos) - std::abs(state[6]);
}

void reinforce::REINFORCE(NN::network_compiled& net,
	int descent_iterations,
	int batch_size,
	int N_ticks,
	float discount_factor,
	std::string save_file) {
	float best_appraisal = -100000;
	for(int _A = 0; _A < descent_iterations; _A++) {
		std::cout << "iteration " << _A << "\n";
		float average_appraisal = 0;
		playthroughs current_playthroughs;

		for(int _B = 0; _B < batch_size; _B++) {
			current_playthroughs.new_trajectory();
			GS::game current_game(b2Vec2(0, 3), 0, 0);
			current_game.player.body->ApplyAngularImpulse(5,true);
			linalg::vector current_state = GS::get_state(current_game);

			for(int _C = 0; _C < N_ticks; _C++) {
				linalg::vector network_response = NN::run_network(net, current_state);
				std::vector<int> interpreted_action(network_response.size());
				for(int i = 0; i < network_response.size(); i++)
					interpreted_action[i] = network_response[i]>NN::rand_unif(-1,1);
				linalg::vector interpreted_action_f(interpreted_action.size());
				for(int i = 0; i < interpreted_action.size(); i++)
					interpreted_action_f[i] = interpreted_action[i]>0?1:-1;
				std::vector<int> interpreted_thrusters{
					interpreted_action[0],
					interpreted_action[1],
					interpreted_action[1],
					interpreted_action[0],
					interpreted_action[2]
				};
				linalg::vector new_state = GS::game_step(current_game, interpreted_thrusters);

				float reward = appraisal(new_state);
				current_playthroughs.append_SAR(current_state, interpreted_action_f, reward);
				current_state = new_state;
				average_appraisal += reward;
			}
		}
		average_appraisal /= N_ticks*batch_size;
		std::cout << "\taverage appraisal: " << average_appraisal << "\n";
		if(average_appraisal > best_appraisal) {
			best_appraisal = average_appraisal+0.0001;
			std::cout << "\t------updating save file------\n";
			std::ofstream file_out(save_file, std::ofstream::out);
			NN::write_network_compiled(net, file_out);
			file_out.close();
			std::cout << "\t------saved file!------\n";
		}
		current_playthroughs.gradient_descent(net, discount_factor);
	}
}
#include "reinforcement.h"

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
	int Nelements = 0;

	for(auto& traj: list) {
		float accumulation = 0;
		for(int i = (int)(traj.list.size())-1; i >= 0; i--) {
			// calculate state-action attributions
			accumulation *= discount_factor;
			accumulation += traj.list[i].reward;
			traj.list[i].attribution = accumulation;

			NN::run_network(net, traj.list[i].state);
			total_weighted_grad = NN::grad_add(
				total_weighted_grad,
				NN::grad_scale(
					NN::back_propagate(net, traj.list[i].action),
					traj.list[i].attribution
				)
			); // this aims to make the action "more likely"
				// in proportion to its attribution
			Nelements++;
		}
	}
	total_weighted_grad = NN::grad_scale(
		total_weighted_grad,
		-net.learning_rate/Nelements
	);
	NN::apply_gradient(net, total_weighted_grad);
}
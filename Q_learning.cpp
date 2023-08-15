#include "Q_learning.h"

#include <iostream>
#include <fstream>
#include <deque>

using namespace Q_learning;

/*
In the following implementation, the exploration policy is an
epsilon-greedy policy

net: passed by reference parameter network to be trained
dq_max: max size of deque before older data is popped
dq_sample: number of elements in dq that are sampled in the stochastic G descent
it_N: number of iterations
gpi: number of games per iteration
init_eps: initial epsilon value for epsilon-greedy policy
discount_factor: discount factor for Q-value computation
*/

linalg::vector Q_learning::get_custom_state(GS::game& world, linalg::vector target) {
	linalg::vector state = GS::get_state(world);
	target = linalg::add_vector(target, linalg::scale_vector({state[0], state[1]}, -1));
	float targabs = linalg::vector_length(target);
	float velabs = linalg::vector_length({state[2], state[3]});
	float velscale = 20, distancescale = 20;
	return {
		std::min(1.0f,state[2]/velscale), std::min(1.0f,state[3]/velscale), //velocities
		state[6], state[4], state[5], //angle_vel, sin(angle), cos(angle)
		std::min(1.0f,targabs/distancescale), // target distance
		linalg::det_matrix({{state[2],state[3]},target})
		/targabs/velabs, // sin(target_delta_theta)
		linalg::vector_sum(linalg::mult_vector({state[2],state[3]},target))
		/targabs/velabs // cos(target_delta_theta)
	};
}

int policy_decision(GS::game& world, NN::network_compiled& net, linalg::vector target, float eps) {
	// epsilon-greedy policy
	if(NN::rand_unif(0, 1) < eps) {
		return rand()%8;
	}
	else {
		linalg::vector result = NN::run_network(net, get_custom_state(world, target));
		return linalg::argmax_vector(result);
	}
}

void Q_learning::train(
	NN::network_compiled& net,
	int dq_max, int dq_sample, int it_N, int gpi, int max_ticks,
	float init_eps, float discount_factor
) {
	std::deque<SARR> learning_buf;
	for(int _it = 1; _it <= it_N; _it++) {
		for(int _game = 1; _game <= gpi; _game++) {
			GS::game world(b2Vec2(0, 3), 0, 0);
			linalg::vector target;
			
		}
		while(learning_buf.size() > dq_max) learning_buf.pop_front();

	}
}
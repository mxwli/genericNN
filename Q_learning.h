#ifndef Q_LEARNING_H
#define Q_LEARNING_H

#include "NN.h"
#include "game.h"

namespace Q_learning {

	typedef struct state_action_reward_resultant {
		// a binding of three types:
		// a state, an action, the reward corresponding with the two
		// and the resulting state
		linalg::vector state;
		linalg::vector action;
		float reward;
		linalg::vector resultant;
	} SARR;
	
	// here, we attempt to implement deep Q-learning.
	// the input of the network is the state and objective
	// the output of the network is "how desirable" a particular
	// action is. Note that this means that there is an output for
	// every single unique action (meaning 3 thrusters would mean 2^3=8 outputs)

	/*
	This time, the game is more complex than what was learned in reinforce.h
	the goal of the game is to fly within a certain distance of
	a series of points in space as quickly as possible.
	the input vector will be of the following form
	[
		min(1, vel_X/20), vel_Y/20, angle_vel, sin(angle), cos(angle), //(standard inputs)
		min(1, target_distance/20),
		sin(target_delta_theta), cos(target_delta_theta)
						// objective inputs
	]
	the output of the network will be an 8 length vector indexed by a bitmask
	of the thruster activations. The content of the vector is the Q-value of the
	state action pair
	*/

	linalg::vector get_custom_state(GS::game& world, linalg::vector target);
	// we're using a custom state to fit our unique situation
	
	void train(
		NN::network_compiled& net,
		int deque_max_size, int dq_sample, // buffer parameters
		int iteration_N, int games_per_it, int max_ticks, // game parameters
		float init_epsilon, float discount_factor // descent parameters
	);

}

#endif
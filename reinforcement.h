#ifndef REINFORCEMENT_H
#define REINFORCEMENT_H

#include <string>

#include "NN.h"

namespace reinforce {

	// a header defining a namespace of reinforcement utilities

	typedef struct state_action_reward {
		// a binding of three types:
		// a state, an action, and the reward corresponding with the two
		linalg::vector state;
		linalg::vector action;
		float reward;
		float attribution;
	} state_action_reward;

	typedef struct trajectory {
		// a list of state_action_rewards, representing the trajectory
		// of the agent in a single playthrough
		std::vector<state_action_reward> list;
	} trajectory;

	typedef struct playthroughs {
		// a list of trajectories, each representing a separate playthough
		std::vector<trajectory> list;
		void new_trajectory();
		void append_SAR(linalg::vector state, linalg::vector action, float reward);
		void gradient_descent(NN::network_compiled& net, float discount_factor);
		// the goal of playthroughs is  to "blindly" perform reinforcement actions
		// (ie without knowledge of the policy or game)
		// so that the struct acts as a library to train the network
	} playthroughs;

	void REINFORCE(NN::network_compiled& net,
		int descent_iterations,		//number of times gradient descent is applied
		int batch_size,				//number of playthroughs per descent
		int N_ticks,				//number of simulation ticks
		float discount_factor,		//commonly referred to as gamma
		std::string save_file		//where do we save the best network?
	);
	// The goal of the above is to run a common variant of Ronald Williams'
	// REINFORCE algorithms, As described in page 621 of hands
	// on machine learning with scikit learn, keras & tensorflow
	// (Aurelien Geron)
	// the input of the network is GS::get_state, and the output of the network
	// is a number x in [-1, 1] indicating the probability of a thruster
	// being fired (ie. x=0 means 50% chance)
}
#endif
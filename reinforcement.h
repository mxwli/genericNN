#ifndef REINFORCEMENT_H
#define REINFORCEMENT_H

#include "NN.h"

namespace reinforce {

	// a header defining a namespace of reinforcement utilities

	typedef struct state_action_reward {
		// a binding of three types:
		// a state, an action, and the reward corresponding with the two
		linalg::vector state;
		linalg::vector action;
		float reward;
	} state_action_reward;

	typedef struct SAR_list {
		// a list of state_action_rewards, representing the trajectory
		// of the agent in a single playthrough
		std::vector<state_action_reward> list;
	} SAR_list;

	// The goal of the above is to run a common variant of Ronald Williams'
	// REINFORCE algorithms, As described in page 621 of hands
	// on machine learning with scikit learn, keras & tensorflow
	// (Aurelien Geron)
}
#endif
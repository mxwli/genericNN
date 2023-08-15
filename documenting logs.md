# Box2D notes:

### setup:
```cpp
b2Vec2 gravity(0, -10);
b2World world(gravity);
```

### creating a static body:
```cpp
// Define the body
b2BodyDef groundBodyDef;
groundBodyDef.position.Set(0, -10);
b2Body* groundBody = world.CreateBody(&groundBodyDef);

// Define the polygon
b2PolygonShape groundBox;
groundBox.SetAsBox(width, height);

// Add the fixture (properties of body)
groundBody->CreateFixture(&groundBox, 0);
```

### creating a dynamic body
```cpp
// Define the body
b2BodyDef bodyDef;
bodyDef.type = b2_dynamicBody;
bodyDef.position.Set(0, 4);
b2Body* body = world.CreateBody(&bodyDef);

// Define the polygon
b2PolygonShape dynamicBox;
dynamicBox.SetAsBox(1, 1);

//Define the fixture (properties)
b2FixtureDef fixtureDef;
fixtureDef.shape = &dynamicBox;
fixtureDef.density = 1.0f;
fixtureDef.friction = 0.3f;
fixtureDef.restitution = 1.0f;

// Add the fixture (properties of the body)
body->CreateFixture(&fixtureDef);
```

### Simulating the world
```cpp
// setup
float timeStep = 1.0f/6.0f;
int velocityIterations = 6;
int positionIterations = 2;

// loop
for(...) {
    world.Step(timeStep, velocityIterations, positionIterations);
    // perform insight here
}

// resulting world
```

### Gradient Descent:
to increase the output of a neuron:
increase its input bias
increase each weight in proportion to the correspodning neuron value
increase each neuron value in proportion to the corresponding weight (this is done via another layer of gradient descent)

### Using the network (XOR example)
```cpp
#include <iostream>
#include "NN.h"
using namespace std;

int main() {
	vector<NN::layer> vec{
		NN::layer(2, NN::activation_type::linear),
		NN::layer(3, NN::activation_type::relu),
		NN::layer(1, NN::activation_type::logistic),
	};
	NN::network net(vec, 0.1);
	NN::network_compiled net_comp = NN::compile_network(net);
	vector<linalg::vector> train_data{
		linalg::vector{0,0},
		linalg::vector{0,1},
		linalg::vector{1,0},
		linalg::vector{1,1}
	};
	vector<linalg::vector> target_data{
		linalg::vector{0},
		linalg::vector{1},
		linalg::vector{1},
		linalg::vector{0}
	};
	for(int epoch = 0; epoch < 5000; epoch++) {
		NN::gradient total(net_comp);
		float MSE = 0;
		for(int i = 0; i < 4; i++) {
			linalg::vector output = NN::run_network(net_comp, train_data[i]);
			MSE += NN::mean_squared_error(target_data[i], output);
			total = NN::grad_add(
				total,
				NN::back_propagate(net_comp, target_data[i])
			);
		}	// REINFORCE algorithms, As described in page 621 of hands
	// on machine learning with scikit learn, keras & tensorflow
	// (Aurelien Geron)
		total = NN::grad_scale(total, -net_comp.learning_rate/4);
		NN::apply_gradient(net_comp, total);
	}
	while(true) {
		float a, b;
		cin >> a >> b;
		cout << (NN::run_network(net_comp, {a, b})[0]) << "\n";
	}
}
```

# reinforcement learning:
```cpp
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

	cout << "0 for train, 1 for testing\n";
	int choice_number; cin >> choice_number;
	if(choice_number == 0) {
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
		linalg::vector state{	// REINFORCE algorithms, As described in page 621 of hands
	// on machine learning with scikit learn, keras & tensorflow
	// (Aurelien Geron)GetAngularVelocity(),
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
```

### Observations:
higher epochs generally increase accuracy under the right hyperparameters.
lower samples allows for faster learning and less of a chance to converge to a bad minima (around 6 is optimal).
N_ticks is untested.
discount_factor is untested.
learning_rate is optimal at around 0.001, give or take 0.0005.
optimal layer configuration is 4 (linear), 10 (relu), 6(logistic).

### Reinforcement learning (rewards adjusted)
```cpp
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
```

### observations
with this style of reinforcement learning, the following hyperparameters seem to work
iterations: ~750 to train a network to a satisfactory level
batch_size: a high number (32+) is necessary for the network to converge
N_ticks: Should be 75 at the bare minimum (5 seconds in simulation time)
decay: 0.995; alternative decay values have been untested so far

the 7:10(relu):3(logistic) network model appears to work just fine.

However, in preparation for training for more advanced games, a second
relu layer may be added in the future

Moving on from basic reinforce algorithms, the next step is to implement
deep Q-learning
Rewrite of old game I wrote in highschool


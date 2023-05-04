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
		}
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
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


#ifndef GAME_H
#define GAME_H
#include <box2d/box2d.h>
#include <vector>
#include "raymath.h" //used for its vector math

namespace GS {

	// The objective of the game is to stabilize the
	// position and velocity of a drone. The drone is
	// defined to be a 1.5x0.5 box with 6 thrusters, 3 on either end
	// the 3 thrusters are offset 90 degrees from each other
	// and the length of the drone, making a sort of ++ shape.
	// specifically, the thrusters are at positions (0.75, 0), (-0.75, 0)
	
	// For now, the simplicity of the game is designed to simply test
	// elementary reinforcement algorithms. As such, the objective is
	// to make the velocity and angular velocity as close to zero as possible

	typedef struct thruster {
		Vector2 local_pos;
		Vector2 local_thrust;
		thruster(float x, float y, float fx, float fy);
	} thrust;

	typedef struct drone {
		b2PolygonShape box;
		b2BodyDef body_def;
		b2FixtureDef fixture_def;
		b2Body* body;
		std::vector<Vector2> ends;
		std::vector<thruster> thrusters;
		Vector2 local_to_global(Vector2 v);
		// converts a point in the local coordinate to a point
		// in the global coordinate system
		void apply_thrusters(std::vector<int> thruster_mask);
		// if an element in the mask is greater than zero, then
		// the corresponding thruster is fired and a force is imparted
		drone(b2World& world);

	} drone;

	typedef struct game {
		b2World world;
		drone player;
		game(b2Vec2 gravity);
	} game;
}

#endif
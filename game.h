#ifndef GAME_H
#define GAME_H
#include <box2d/box2d.h>
#include <vector>
#include "linalg.h"
#include "raylib.h"
#include "raymath.h" //used for its vector math

namespace GS {

	// The objective of the game is to stabilize the
	// position and velocity of a drone. The drone is
	// defined to be a box with various thrusters
	
	// For now, the simplicity of the game is designed to simply test
	// elementary reinforcement algorithms

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
		std::vector<int> last_activation;
		Vector2 local_to_global(Vector2 v);
		// converts a point in the local coordinate to a point
		// in the global coordinate system
		void apply_thrusters(std::vector<int> thruster_mask);
		// if an element in the mask is greater than zero, then
		// the corresponding thruster is fired and a force is imparted
		void draw_drone(Vector2 camera_pos, float zoom, Color draw_color);
		// assumes we're using raylib and draws drone onto the screen
		// with a given zoom and top left coordinate of camera

		drone(float posx, float posy, b2World& world);

	} drone;

	typedef struct game {
		b2World world;
		drone player;
		game(b2Vec2 gravity, float posx, float posy);
	} game;

	// for training purposes:
	// Takes in a game and an action, performs the action
	// (player.apply_thrusters) and steps the game,
	// returns a linalg::vector of the following form:
	// [X-position, Y-position,
	// X-velocity, Y-velocity,
	// sin(angle), cos(angle), angular velocity]
	linalg::vector game_step(game& game, std::vector<int> action);
	// the following function does nothing and returns the save as the above
	linalg::vector get_state(game& game);
}

#endif
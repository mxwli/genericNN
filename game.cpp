#include "game.h"

#include <iostream>

using namespace GS;

#define DEBUG_GS

GS::thruster::thruster(float x, float y, float fx, float fy) {
	local_pos = {x, y};
	local_thrust = {fx, fy};
}

Vector2 GS::drone::local_to_global(Vector2 v) {
	Vector2 body_global_pos = {body->GetPosition().x, body->GetPosition().y};
	return Vector2Add(
		body_global_pos,
		Vector2Rotate(
			v,
			body->GetAngle()
		)
	);
}

void GS::drone::apply_thrusters(std::vector<int> thruster_mask) {
	#ifdef DEBUG_GS
		assert(thruster_mask.size() == thrusters.size());
	#endif
	Vector2 body_global_pos = {body->GetPosition().x, body->GetPosition().y};
	float body_global_angle = body->GetAngle();
	for(int i = 0; i < thrusters.size(); i++) if(thruster_mask[i]>0) {
		Vector2 global_pos = Vector2Add(
			body_global_pos,
			Vector2Rotate(
				thrusters[i].local_pos,
				body_global_angle
			)
		);
		Vector2 global_thrust = Vector2Rotate(
			thrusters[i].local_thrust,
			body_global_angle
		);
		body->ApplyForce(
			b2Vec2(global_thrust.x,global_thrust.y),
			b2Vec2(global_pos.x,global_pos.y),
			true
		);
	}
}

GS::drone::drone(b2World& world) {
	body_def.type = b2_dynamicBody;
	body_def.position.Set(5, 5);
	body = world.CreateBody(&body_def);

	box.SetAsBox(1.5f, 0.75f);
	
	fixture_def.shape = &box;
	fixture_def.density = 1.0f;
	fixture_def.friction = 1.0f;
	fixture_def.restitution = 1.0f;

	body->CreateFixture(&fixture_def);

	ends = {{0.75, 0.25}, {0.75, -0.25},
		{-0.75, -0.25}, {-0.75, 0.25}, {0.75, 0.25}};

	thrusters = {
		thruster(0.75, 0, 0, -15),
		thruster(0.75, 0, -15, 0),
		thruster(0.75, 0, 0, 15),
		thruster(-0.75, 0, 0, -15),
		thruster(-0.75, 0, 15, 0),
		thruster(-0.75, 0, 0, 15),
	};
}

GS::game::game(b2Vec2 gravity): world(gravity), player(world) {
	
}
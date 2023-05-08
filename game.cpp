#include "game.h"

#include <iostream>
#include "raylib.h"

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
	last_activation = thruster_mask;
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
void GS::drone::draw_drone(Vector2 camera_pos, float zoom, Color draw_color) {
	DrawLineV(
		Vector2Scale(
			Vector2Subtract(local_to_global({0, 0}), camera_pos),
			zoom
		),
		Vector2Add(
			Vector2Scale(
				Vector2Subtract(local_to_global({0, 0}), camera_pos),
				zoom
			),
			{body->GetLinearVelocity().x, body->GetLinearVelocity().y}
		),
		RED
	);
	for(int i = 1; i < ends.size(); i++) {
		DrawLineV(
			Vector2Scale(
				Vector2Subtract(local_to_global(ends[i-1]), camera_pos),
				zoom
			),
			Vector2Scale(
				Vector2Subtract(local_to_global(ends[i]), camera_pos),
				zoom
			),
			draw_color
		);
	}
	for(int i = 0; i < thrusters.size(); i++) if(last_activation[i]) {
		DrawCircleV(
			Vector2Scale(
				Vector2Subtract(
					local_to_global(thrusters[i].local_pos),
					camera_pos),
				zoom
			),
			0.01*Vector2Length(thrusters[i].local_thrust)*zoom,
			RED
		);
	}
}

GS::drone::drone(float posx, float posy, b2World& world) {
	body_def.type = b2_dynamicBody;
	body_def.position.Set(posx, posy);
	body = world.CreateBody(&body_def);

	float width = 1, height = 1;

	box.SetAsBox(width, height);
	
	fixture_def.shape = &box;
	fixture_def.density = 1.0f;
	fixture_def.friction = 1.0f;
	fixture_def.restitution = 1.0f;

	body->CreateFixture(&fixture_def);

	ends = {{width/2, height/2}, {width/2, -height/2},
		{-width/2, -height/2}, {-width/2, height/2}, {width/2, height/2}};

	thrusters = {
		thruster(width/2, height/4, 0, -10),
		thruster(width/2, -height/4, 0, 10),
		thruster(-width/2, height/4, 0, -10),
		thruster(-width/2, -height/4, 0, 10),
		thruster(0, height/2, 0, -40)
	};

	last_activation = std::vector<int>(thrusters.size());
}

GS::game::game(b2Vec2 gravity, float posx, float posy):
	world(gravity), player(posx, posy, world) {
	
}

linalg::vector GS::game_step(game& game, std::vector<int> action) {
	game.player.apply_thrusters(action);
	game.world.Step(1.0/15, 2, 2);
	linalg::vector ret{
		game.player.body->GetPosition().x,
		game.player.body->GetPosition().y,
		game.player.body->GetLinearVelocity().x,
		game.player.body->GetLinearVelocity().y,
		std::sin(game.player.body->GetAngle()),
		std::cos(game.player.body->GetAngle()),
		game.player.body->GetAngularVelocity()
	};
	return ret;
}
linalg::vector GS::get_state(game& game) {
	linalg::vector ret{
		game.player.body->GetPosition().x,
		game.player.body->GetPosition().y,
		game.player.body->GetLinearVelocity().x,
		game.player.body->GetLinearVelocity().y,
		std::sin(game.player.body->GetAngle()),
		std::cos(game.player.body->GetAngle()),
		game.player.body->GetAngularVelocity()
	};
	return ret;
}
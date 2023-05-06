#include <iostream>
#include <raylib.h>
#include <raymath.h>
#include <box2d/box2d.h>
#include "game.h"
using namespace std;
#define SCREENWIDTH 700
#define SCREENHEIGHT 700
#define GAMEWIDTH 10
#define GAMEHEIGHT 10
#define ZOOM 70


int main() {
	GS::game frame(b2Vec2(0, 0));
	InitWindow(SCREENWIDTH, SCREENHEIGHT, "");
	int FPS = 30;
	int velocityIterations = 6;
	int positionIterations = 2;

	SetTargetFPS(FPS);

	while(!WindowShouldClose()) {
		vector<int> mask{
			IsKeyDown(KEY_Q),
			IsKeyDown(KEY_A),
			IsKeyDown(KEY_Z),
			IsKeyDown(KEY_E),
			IsKeyDown(KEY_D),
			IsKeyDown(KEY_C)
		};
		frame.player.apply_thrusters(mask);
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
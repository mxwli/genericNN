#include <iostream>
#include "NN.h"
#include <raylib.h>
#include <raymath.h>
#include <box2d/box2d.h>
using namespace std;
#define SCREENWIDTH 700
#define SCREENHEIGHT 700
#define GAMEWIDTH 10
#define GAMEHEIGHT 10
#define ZOOM 70


int main() {
	InitWindow(SCREENWIDTH, SCREENHEIGHT, "");
	int FPS = 30;
	int velocityIterations = 6;
	int positionIterations = 2;

	SetTargetFPS(FPS);

	while(!WindowShouldClose()) {
		BeginDrawing();
		ClearBackground(BLACK);
		EndDrawing();
	}
}
#include <assert.h>
#include <stdio.h>
#include "raylib.h"
#include <math.h>

const size_t sw = 1680;
const size_t sh = 1024;

int main(){
    while (!WindowShouldClose())
    {
        BeginDrawing();
        ClearBackground(BLACK);


        DrawLine(0, 0, sw, sh, WHITE);

        EndDrawing();
    }

    CloseWindow();

    return 0;
}

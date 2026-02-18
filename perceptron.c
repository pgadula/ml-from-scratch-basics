#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "raylib.h"
#include "raymath.h"

#define NOB_IMPLEMENTATION
#include "nob.h"

#include "anim.h"

#define WIDTH 1920
#define HEIGHT 1024

#define POINT_COUNT 512
#define CLASS_COUNT 4
#define POINT_RADIUS 0.1

#define BACKGROUND_COLOR (Color){0, 2, 8, 255}
#define COLOR_GRAY       (Color){80, 80, 80 ,255}
#define COLOR_BLUE       (Color){88, 196, 221, 255}
#define COLOR_RED        (Color){255, 85, 85, 255}
#define COLOR_GREEN      (Color){130, 255, 100, 255}

TweenEngine te;

typedef enum {
    VIEW_2D = 0,
    VIEW_3D = 1
} VIEW_MODE;

typedef enum {
    UNKNOWN = 0,
    SETOSA = 1,
    VIRGINICA = 2,
    VERSICOLOR = 3
} IRIS_LABEL;


IRIS_LABEL map_label(const char *input) {
    if (strcmp(input, "Setosa") == 0)
        return SETOSA;
    if (strcmp(input, "Virginica") == 0)
        return VIRGINICA;
    if (strcmp(input, "Versicolor") == 0)
        return VERSICOLOR;
    return UNKNOWN;
}

typedef enum {
    EUC_2D = 0,
    EUC_3D = 1,
} DIST_METRIC;

VIEW_MODE view_mode = VIEW_2D;

typedef struct {
    float *w;
    float b;
} Perceptron;


typedef struct{
    float v0;
    float label;
}Sample;

typedef struct {
    int count;
    int capacity;
    Sample *items;
} TrainingSet;


float randf(float min, float max)
{
    return min + (float)rand() / RAND_MAX * (max - min);
}

typedef struct{
    Vector3 from;
    Vector3 to;
    Color color;
} ArrowData;

void draw_arrow(float t, ArrowData *arrow){
    if(view_mode == VIEW_2D){
        arrow->from.y = 0;
        arrow->to.y = 0;
    }

    DrawLine3D(
        arrow->from,
        lerp_vec3(arrow->from, arrow->to, t),
        lerp_color(YELLOW, arrow->color, t)
    );
}


float axes_len = 0.0f; 

void draw_axes(VIEW_MODE view_mode) {
    float len = axes_len;
    if (len < 0.01f) return;
    
    DrawLine3D((Vector3){-len, 0, 0}, (Vector3){ len, 0, 0}, RED);
    if (view_mode == VIEW_3D)
        DrawLine3D((Vector3){0, -len, 0}, (Vector3){0, len, 0}, GREEN);
    DrawLine3D((Vector3){0, 0, -len}, (Vector3){0, 0, len}, BLUE);

    int ticks = (int)len;
    for (int i = -ticks; i <= ticks; i++) {
        float t = 0.1f;
        DrawLine3D((Vector3){i, -t, 0}, (Vector3){i, t, 0}, COLOR_RED);
        if (view_mode == VIEW_3D)
            DrawLine3D((Vector3){-t, i, 0}, (Vector3){t, i, 0}, COLOR_GREEN);
        DrawLine3D((Vector3){0, -t, i}, (Vector3){0, t, i}, COLOR_BLUE);
    }
}

Vector3 random_vec3(){
    return (Vector3){ .x = randf(-10, 10), .y = randf(-10, 10), .z = randf(-10, 10)};
}


Color z_axes_labels =       (Color){88, 196, 221, 0};
Color x_axes_labels =        (Color){255, 85, 85, 0};
Color y_axes_labels =      (Color){130, 255, 100, 0};

void draw_axis_labels(const Camera *camera, VIEW_MODE view_mode) {
    float len = 6.2f;

    Vector2 x_pos = GetWorldToScreen((Vector3){ len, 0, 0}, *camera);
    Vector2 y_pos = GetWorldToScreen((Vector3){ 0, len, 0}, *camera);
    Vector2 z_pos = GetWorldToScreen((Vector3){ 0, 0, len}, *camera);

    DrawText("X petal width",  (int)x_pos.x, (int)x_pos.y, 24, x_axes_labels); 
    if (view_mode == VIEW_3D)
        DrawText("Y sepal width",  (int)y_pos.x, (int)y_pos.y, 24, y_axes_labels); 
    DrawText("Z petal length", (int)z_pos.x, (int)z_pos.y, 24, z_axes_labels); 

}

typedef struct {
    Camera camera;

    //animation props
    Vector3 desire_target;
    Vector3 desire_pos;
    float desire_fovy;
} AnimCamera;


void cam_look_at(Camera *cam, Vector3 target){
    tween_vec3(&te, &cam->target, target, 1); 
}

Tween *cam_move(Camera *cam, Vector3 target){
    return tween_vec3(&te, &cam->position, target, 1); 
}

void cam_fovy(Camera *cam, float target){
    tween_float(&te, &cam->fovy, target, 2); 
}


void draw_perceptron(const Perceptron *perceptron){

}

void train(Perceptron *perceptron, TrainingSet *dataset){
   int weights_count = sizeof(perceptron->w) / sizeof(perceptron->w[0]);
   assert(weights_count == dataset->count); 
   
}

float cost(){
    return 0;
}
int main()
{
    srand(time(NULL));

    te = (TweenEngine){0};

    TrainingSet training_set = {0};

    BoundingBox ground = { (Vector3){ -100, 0, -100 }, (Vector3){100, 0, 100} };

    Camera camera = { 0 };
    camera.position = (Vector3){ -10.0f, 0.0f, 0.5f };
    camera.target = (Vector3){ 0.0f, -1.0f, 1.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    Tween *tw = tween_float(&te, &axes_len, 6.0f, 2.0f);
    tw->ease = EASE_OUT_BOUNCE;
    tw->elapsed = -0.5;

    //labels animation
   Tween *l1 = tween_alpha(&te, &x_axes_labels, 0, 255, 2);
   Tween *l2 = tween_alpha(&te, &y_axes_labels, 0, 255, 2);
   Tween *l3 = tween_alpha(&te, &z_axes_labels, 0, 255, 2);

   l1->elapsed = -1.5;
   l2->elapsed = -1.5;
   l3->elapsed = -1.5;

   InitWindow(WIDTH, HEIGHT, "Perceptron");
   SetTargetFPS(60);

    SetMousePosition(WIDTH/2, HEIGHT/2);

    while (!WindowShouldClose())
    {
        float dt = GetFrameTime(); 
  //      if (view_mode == VIEW_3D)
  //         UpdateCamera(&camera, CAMERA_FREE);
  //      else{
  //          float scroll = GetMouseWheelMove();
  //          if (scroll != 0){
  //              tween_float(&te, &camera.fovy, 
  //              Clamp(camera.fovy - scroll * 3.0f, 10.0f, 90.0f), 0.3f);
  //          }
  //      }
        /* Input */
        BeginDrawing();
            ClearBackground(BACKGROUND_COLOR);
            BeginMode3D(camera);

  //              tween_update(&te, dt);
  //              if(view_mode == VIEW_2D)
  //                  DrawGrid(10, 1);        // Draw a grid
  //              draw_axes(view_mode);
  //          EndMode3D();
  //              draw_axis_labels(&camera, view_mode);
  //              DrawText("SPACE - regenerate points", 20, 20, 20, GRAY);
        EndDrawing();
    }
    CloseWindow();
    return 0;
}


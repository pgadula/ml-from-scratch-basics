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
#include "iris.h"

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

float lr;
bool is_training;

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

VIEW_MODE view_mode = VIEW_3D;

typedef struct {
    Vector3 pos;
    Color color;
    float radius;
} Visual;

typedef struct {
    float x;
    float y;
    float z;
    IRIS_LABEL label;
    int class;
    Visual vis;
} Sample;

typedef struct {
    size_t capacity;
    size_t count;
    Sample *items;
} Dataset;

typedef struct{
   float b; 
   float w1;  
   float w2;  
} SVM;

Color FEATURES_COLORS[CLASS_COUNT] = {
    COLOR_GRAY,
    COLOR_BLUE,
    COLOR_RED,
    COLOR_GREEN
};

Color CLASSIFIED_COLORS[CLASS_COUNT] = {
    GRAY,
    BLUE,
    RED,
    GREEN
};

float randf(float min, float max)
{
    return min + (float)rand() / RAND_MAX * (max - min);
}

void toggle_view(VIEW_MODE* view_mode)
{
    *view_mode ^= VIEW_3D;
}

void reset_points(Dataset *dataset)
{
    dataset->count = 0;
}

float get_dist(DIST_METRIC metric, Vector3 a, Vector3 b) {
    float dx = a.x - b.x;
    float dz = a.z - b.z;
    float dy = a.y - b.y;

    switch (metric) {
        case EUC_2D:
            return dx*dx + dz*dz;
        case EUC_3D: 
            return dx*dx + dy*dy + dz*dz;
        default:
            return 0.0f;
    }
}

typedef struct{
    Vector3 from;
    Vector3 to;
    Color color;
} ArrowData;


void generate_points(Dataset *dataset)
{
    reset_points(dataset);
    Sample* points = dataset->items;
    for (int i = 0; i < dataset->capacity; i++)
    {
        Sample pt = (Sample){.x = randf(0, WIDTH), .y = randf(0, HEIGHT), .z = randf(0, WIDTH), .label = UNKNOWN};
        da_append(dataset, pt);
    }
}

void draw_axes(VIEW_MODE view_mode) {
    float len = 6.0f;
    float label_offset = 0.3f;
    
    DrawLine3D(
        (Vector3){-len, 0, 0}, 
        (Vector3){ len, 0, 0}, 
        RED
    );
    if(view_mode == VIEW_3D)
        DrawLine3D(
                (Vector3){0, -len, 0},
                (Vector3){0,  len, 0}, 
                GREEN
                );
    DrawLine3D(
        (Vector3){0, 0, -len}, 
        (Vector3){0, 0,  len}, 
        BLUE
    );

    for (int i = -5; i <= 5; i++) {
        float t = 0.1f;
        DrawLine3D(
            (Vector3){i, -t, 0}, 
            (Vector3){i,  t, 0}, 
            COLOR_RED
        );

        if(view_mode == VIEW_3D)
            DrawLine3D(
                    (Vector3){-t, i, 0}, 
                    (Vector3){ t, i, 0}, 
                    COLOR_GREEN
                    );
        DrawLine3D(
            (Vector3){0, -t, i}, 
            (Vector3){0,  t, i}, 
            COLOR_BLUE
        );
    }
}

void draw_dataset(const Dataset *td, float dt, bool is_training_set){
    for (int i = 0; i < td->count; i++){
        Sample entry = td->items[i];
        Visual vis = entry.vis; 
        Vector3 pos = vis.pos; 
        float r = vis.radius; 
        Color color = vis.color; 
        if (view_mode == VIEW_2D)
            pos.y = 0;

        if (is_training_set) {
            DrawSphere(pos, r, color);
        } else {
            float size = r * 1.2f;
            DrawCube(pos, size, size, size, color);
            if (entry.label == UNKNOWN) {
                float pulse = 1.0f + 0.2f * sinf(GetTime() * 4.0f);
                float ps = size * 1.5f * pulse;
                DrawCube(pos, ps, ps, ps, (Color){255, 255, 255, 60});
            }
        }
    }
}

Vector3 random_vec3(){
    return (Vector3){ .x = randf(-10, 10), .y = randf(-10, 10), .z = randf(-10, 10)};
}


void prepare_training_dataset(Dataset *td, TweenEngine *e){
    float max_sepal_length = IRIS.data[0].sepal_length;
    float max_sepal_width = IRIS.data[0].sepal_width;
    float max_petal_length = IRIS.data[0].petal_length;
    float max_petal_width = IRIS.data[0].petal_width;

    for(int i = 0; i < IRIS.count; i++){
        Row data = IRIS.data[i];
        if(max_sepal_length < data.sepal_length)
            max_sepal_length = data.sepal_length;
        if(max_sepal_width < data.sepal_width)
            max_sepal_width = data.sepal_width;
        if(max_petal_length < data.petal_length)
            max_petal_length = data.petal_length;
        if(max_petal_width < data.petal_width)
            max_petal_width = data.petal_width;
    }

    for(int i = 0; i < IRIS.count; i++){
        Row row = IRIS.data[i];
        IRIS_LABEL label = map_label(row.variety);
        if (label == VERSICOLOR) continue;
        int class = label == SETOSA ? 1 : -1;

        float s_l = (row.sepal_length / max_sepal_length) / 10;
        float s_w = (row.sepal_width  / max_sepal_width ) * 10.0f - 5.0f;
        float p_l = (row.petal_length / max_petal_length) * 10.0f - 5.0f;
        float p_w = (row.petal_width  / max_petal_width ) * 10.0f - 5.0f;

        Sample sample = (Sample){ 
            .x = p_w, .y = p_l, .z = s_w,
            .label = label, 
            .class = class, 
            .vis = { 
                .pos = random_vec3(),
                .radius = 0, 
                .color = WHITE
            }
        };
        da_append(td, sample);

        int idx = td->count - 1;
        float dur = 1.0;
        Color color = FEATURES_COLORS[label];
        tween_vec3(e, &td->items[idx].vis.pos, 
                (Vector3){ .x = p_w, .y = p_l, .z = s_w }, dur);
        tween_float(e, &td->items[idx].vis.radius, s_l, dur);
        tween_color(e, &td->items[idx].vis.color, color, dur);
    }
}

void draw_classes(){
    DrawText("SETOSA", WIDTH-150, 20, 20, FEATURES_COLORS[SETOSA]);
    DrawText("VIRGINICA", WIDTH-150, 40, 20, FEATURES_COLORS[VIRGINICA]);
}

void draw_axis_labels(const Camera *camera, VIEW_MODE view_mode) {
    float len = 6.2f;
    Vector2 x_pos = GetWorldToScreen((Vector3){ len, 0, 0}, *camera);
    Vector2 y_pos = GetWorldToScreen((Vector3){ 0, len, 0}, *camera);
    Vector2 z_pos = GetWorldToScreen((Vector3){ 0, 0, len}, *camera);

    DrawText("X petal width",  (int)x_pos.x, (int)x_pos.y, 24, COLOR_RED);
    if (view_mode == VIEW_3D)
        DrawText("Y sepal width",  (int)y_pos.x, (int)y_pos.y, 24, COLOR_GREEN);
    DrawText("Z petal length", (int)z_pos.x, (int)z_pos.y, 24, COLOR_BLUE);
}

typedef struct {
    Camera camera;

    //animation props
    Vector3 desire_target;
    Vector3 desire_pos;
    float desire_fovy;
} AnimCamera;


void cam_look_at(TweenEngine *e, Camera *cam, Vector3 target){
    tween_vec3(e, &cam->target, target, 1); 
}

Tween *cam_move(TweenEngine *e, Camera *cam, Vector3 target){
    return tween_vec3(e, &cam->position, target, 1); 
}

void cam_fovy(TweenEngine *e, Camera *cam, float target){
    tween_float(e, &cam->fovy, target, 2); 
}

void toggle_view_anim(TweenEngine *e, Dataset *ds, Camera *camera, VIEW_MODE *view_mode) {
    *view_mode ^= VIEW_3D;

    cam_look_at(e, camera, (Vector3){ 0, 0, 0 });
    if (*view_mode == VIEW_3D) {
        for (int i = 0; i < ds->count; i++) {
            tween_vec3(e, &ds->items[i].vis.pos, 
                    (Vector3){ ds->items[i].x, ds->items[i].y, ds->items[i].z }, 1.0f);
        }
        cam_look_at(e, camera, (Vector3){ 0, 0, 0 });
        cam_move(e, camera, (Vector3){ 10, 10, 10 });
    } else {
        for (int i = 0; i < ds->count; i++) {
            tween_vec3(e, &ds->items[i].vis.pos,
                    (Vector3){ ds->items[i].x, 0, ds->items[i].z }, 1.0f);
        }
        cam_move(e, camera, (Vector3){ 0.0, 20, 0.01 });
        cam_look_at(e, camera, (Vector3){ 0, 0, 0 });
    }
}

void train(Dataset *dataset, SVM *svm){
    float C = 1;
    float w1, w2, b;
    w1 = svm->w1;
    w2 = svm->w2;
    b = svm->b;

    int epochs = 1;
    
    for(int e = 0; e < epochs; e++){
        for(int i = 0; i < dataset->count; i++){
            Sample s = dataset->items[i];
            float x1 = s.x;
            float x2 = s.z;
            float yi = s.class;

            float margin = yi*(w1*x1 + w2*x2 + b);

            if (margin >= 1){
                w1 -= lr * w1;
                w2 -= lr * w2;
            }
            else{
                w1 -= lr * (w1 - C*yi*x1);
                w2 -= lr * (w2 - C*yi*x2);
                b  -= lr * (-C*yi);
            }
        }
    }

    //printf("w1:%f w2:%f b:%f\n", w1, w2, b);
    svm->w1 = w1;
    svm->w2 = w2;
    svm->b = b;
}

void draw_svm(const SVM *svm, VIEW_MODE view_mode){
    float w1 = svm->w1;
    float w2 = svm->w2;
    float b  = svm->b;

    if (fabsf(w2) < 0.0001f) return;

    float x0 = -5.0f, x1 = 5.0f;

    if (view_mode == VIEW_3D) {
        float y0 = -5.0f, y1 = 5.0f;
        Color margin_fill = (Color){88, 196, 221, 20};

        Vector3 mp1_a = {x0, y0, -(w1*x0 + b - 1.0f) / w2};
        Vector3 mp1_b = {x1, y0, -(w1*x1 + b - 1.0f) / w2};
        Vector3 mp1_c = {x1, y1, -(w1*x1 + b - 1.0f) / w2};
        Vector3 mp1_d = {x0, y1, -(w1*x0 + b - 1.0f) / w2};
        DrawTriangle3D(mp1_a, mp1_b, mp1_c, margin_fill);
        DrawTriangle3D(mp1_a, mp1_c, mp1_d, margin_fill);
        DrawTriangle3D(mp1_c, mp1_b, mp1_a, margin_fill);
        DrawTriangle3D(mp1_d, mp1_c, mp1_a, margin_fill);

        Vector3 mm1_a = {x0, y0, -(w1*x0 + b + 1.0f) / w2};
        Vector3 mm1_b = {x1, y0, -(w1*x1 + b + 1.0f) / w2};
        Vector3 mm1_c = {x1, y1, -(w1*x1 + b + 1.0f) / w2};
        Vector3 mm1_d = {x0, y1, -(w1*x0 + b + 1.0f) / w2};
        DrawTriangle3D(mm1_a, mm1_b, mm1_c, margin_fill);
        DrawTriangle3D(mm1_a, mm1_c, mm1_d, margin_fill);
        DrawTriangle3D(mm1_c, mm1_b, mm1_a, margin_fill);
        DrawTriangle3D(mm1_d, mm1_c, mm1_a, margin_fill);
    } else {
        DrawLine3D(
                (Vector3){x0, 0, -(w1*x0 + b) / w2},
                (Vector3){x1, 0, -(w1*x1 + b) / w2}, WHITE);

        DrawLine3D(
                (Vector3){x0, 0, -(w1*x0 + b - 1.0f) / w2},
                (Vector3){x1, 0, -(w1*x1 + b - 1.0f) / w2}, BLUE);
        DrawLine3D(
                (Vector3){x0, 0, -(w1*x0 + b + 1.0f) / w2},
                (Vector3){x1, 0, -(w1*x1 + b + 1.0f) / w2}, BLUE);
    }
}

void svm_icr_b(SVM *svm, float delta){
    svm->b += delta;
}

void svm_icr_w1(SVM *svm, float delta){
    svm->w1 += delta;
}

void svm_icr_w2(SVM *svm, float delta){
    svm->w2 += delta;
}


int main()
{
    lr = 0.0001;
    is_training = false;
    float delta = 0.01;
    srand(time(NULL));
    TweenEngine te = {0};

    da_reserve(&te, 1024);

    Dataset dataset = {0};
    Dataset training_set = {0};

    prepare_training_dataset(&training_set, &te);

    BoundingBox ground = { (Vector3){ -100, 0, -100 }, (Vector3){100, 0, 100} };

    SVM svm = { .b = 0.1, .w1 = 0.1, .w2 = 0.1 };
    Camera camera = { 0 };
    camera.position = (Vector3){ -10.0f, 0.0f, 0.5f };
    camera.target = (Vector3){ 0.0f, -1.0f, 1.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    toggle_view_anim(&te, &training_set, &camera, &view_mode);
    InitWindow(WIDTH, HEIGHT, "SVM");
    SetTargetFPS(60);
    
    SetMousePosition(WIDTH/2, HEIGHT/2);

    while (!WindowShouldClose())
    {
        float dt = GetFrameTime(); 
        if (view_mode == VIEW_3D)
           UpdateCamera(&camera, CAMERA_FREE);
        else{
            float scroll = GetMouseWheelMove();
            if (scroll != 0){
                tween_float(&te, &camera.fovy, 
                Clamp(camera.fovy - scroll * 3.0f, 10.0f, 90.0f), 0.3f);
            }
        }
        /* Input */
        if (IsKeyPressed(KEY_T))
            toggle_view_anim(&te, &training_set, &camera, &view_mode);

        if (IsKeyPressed(KEY_Q))
            is_training = !is_training;
        if (IsKeyPressed(KEY_I))
            svm_icr_w2(&svm, delta);
        if (IsKeyPressed(KEY_O))
            svm_icr_w1(&svm, delta);
        if (IsKeyPressed(KEY_P))
            svm_icr_b(&svm, delta);

        if(is_training)
            train(&training_set, &svm);

        BeginDrawing();
            ClearBackground(BACKGROUND_COLOR);
            BeginMode3D(camera);

                tween_update(&te, dt);
                if(view_mode == VIEW_2D)
                    DrawGrid(10, 1);        // Draw a grid
                draw_axes(view_mode);
                draw_dataset(&training_set, dt, true);
                draw_dataset(&dataset, dt, false);
                draw_svm(&svm, view_mode);
            EndMode3D();
                draw_axis_labels(&camera, view_mode);
                
                draw_classes();
        EndDrawing();
    }
    CloseWindow();
    return 0;
}


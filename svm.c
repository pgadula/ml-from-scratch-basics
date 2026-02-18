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

TweenEngine te;

typedef struct{
   float b; 
   float w1;  
   float w2;  
} SVM;

float lr;
bool is_training;
SVM svm_visual = {0};

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


void prepare_training_dataset(Dataset *td){
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
        tween_vec3(&te, &td->items[idx].vis.pos, 
                (Vector3){ .x = p_w, .y = p_l, .z = s_w }, dur);
        tween_float(&te, &td->items[idx].vis.radius, s_l, dur);
        tween_color(&te, &td->items[idx].vis.color, color, dur);
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


void cam_look_at(Camera *cam, Vector3 target){
    tween_vec3(&te, &cam->target, target, 1); 
}

Tween *cam_move(Camera *cam, Vector3 target){
    return tween_vec3(&te, &cam->position, target, 1); 
}

void cam_fovy(Camera *cam, float target){
    tween_float(&te, &cam->fovy, target, 2); 
}

void toggle_view_anim(Dataset *ds, Camera *camera, VIEW_MODE *view_mode) {
    *view_mode ^= VIEW_3D;

    cam_look_at(camera, (Vector3){ 0, 0, 0 });
    if (*view_mode == VIEW_3D) {
        for (int i = 0; i < ds->count; i++) {
            tween_vec3(&te, &ds->items[i].vis.pos, 
                    (Vector3){ ds->items[i].x, ds->items[i].y, ds->items[i].z }, 1.0f);
        }
        cam_look_at(camera, (Vector3){ 0, 0, 0 });
        cam_move(camera, (Vector3){ 10, 10, 10 });
    } else {
        for (int i = 0; i < ds->count; i++) {
            tween_vec3(&te, &ds->items[i].vis.pos,
                    (Vector3){ ds->items[i].x, 0, ds->items[i].z }, 1.0f);
        }
        cam_move(camera, (Vector3){ 0.0, 18, 0.01 });
        cam_look_at(camera, (Vector3){ 0, 0, 0 });
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

    svm->w1 = w1;
    svm->w2 = w2;
    svm->b = b;
}

void draw_svm(const SVM *svm, VIEW_MODE view_mode) {
    float w1 = svm->w1;
    float w2 = svm->w2;
    float b  = svm->b;

    if (fabsf(w1) < 0.0001f && fabsf(w2) < 0.0001f) return;
    if (fabsf(w2) < 0.0001f) return;

    float x0 = -5.0f, x1 = 5.0f;

    if (view_mode == VIEW_3D) {
        float y0 = -5.0f, y1 = 5.0f;
        Color margin_fill = (Color){88, 196, 221, 20};
        Color decision_fill = (Color){255, 255, 255, 12};
        Color edge = (Color){88, 196, 221, 80};
        Color edge_w = (Color){255, 255, 255, 50};

        // decision plane (w1*x + w2*z + b = 0)
        Vector3 d_a = {x0, y0, -(w1*x0 + b) / w2};
        Vector3 d_b = {x1, y0, -(w1*x1 + b) / w2};
        Vector3 d_c = {x1, y1, -(w1*x1 + b) / w2};
        Vector3 d_d = {x0, y1, -(w1*x0 + b) / w2};
        DrawTriangle3D(d_a, d_b, d_c, decision_fill);
        DrawTriangle3D(d_a, d_c, d_d, decision_fill);
        DrawTriangle3D(d_c, d_b, d_a, decision_fill);
        DrawTriangle3D(d_d, d_c, d_a, decision_fill);
        DrawLine3D(d_a, d_b, edge_w);
        DrawLine3D(d_b, d_c, edge_w);
        DrawLine3D(d_c, d_d, edge_w);
        DrawLine3D(d_d, d_a, edge_w);

        // margin +1
        Vector3 mp_a = {x0, y0, -(w1*x0 + b - 1.0f) / w2};
        Vector3 mp_b = {x1, y0, -(w1*x1 + b - 1.0f) / w2};
        Vector3 mp_c = {x1, y1, -(w1*x1 + b - 1.0f) / w2};
        Vector3 mp_d = {x0, y1, -(w1*x0 + b - 1.0f) / w2};
        DrawTriangle3D(mp_a, mp_b, mp_c, margin_fill);
        DrawTriangle3D(mp_a, mp_c, mp_d, margin_fill);
        DrawTriangle3D(mp_c, mp_b, mp_a, margin_fill);
        DrawTriangle3D(mp_d, mp_c, mp_a, margin_fill);
        DrawLine3D(mp_a, mp_b, edge);
        DrawLine3D(mp_b, mp_c, edge);
        DrawLine3D(mp_c, mp_d, edge);
        DrawLine3D(mp_d, mp_a, edge);

        // margin -1
        Vector3 mm_a = {x0, y0, -(w1*x0 + b + 1.0f) / w2};
        Vector3 mm_b = {x1, y0, -(w1*x1 + b + 1.0f) / w2};
        Vector3 mm_c = {x1, y1, -(w1*x1 + b + 1.0f) / w2};
        Vector3 mm_d = {x0, y1, -(w1*x0 + b + 1.0f) / w2};
        DrawTriangle3D(mm_a, mm_b, mm_c, margin_fill);
        DrawTriangle3D(mm_a, mm_c, mm_d, margin_fill);
        DrawTriangle3D(mm_c, mm_b, mm_a, margin_fill);
        DrawTriangle3D(mm_d, mm_c, mm_a, margin_fill);
        DrawLine3D(mm_a, mm_b, edge);
        DrawLine3D(mm_b, mm_c, edge);
        DrawLine3D(mm_c, mm_d, edge);
        DrawLine3D(mm_d, mm_a, edge);

    } else {
        // decision line
        DrawLine3D(
            (Vector3){x0, 0, -(w1*x0 + b) / w2},
            (Vector3){x1, 0, -(w1*x1 + b) / w2}, WHITE);
        // margin lines
        DrawLine3D(
            (Vector3){x0, 0, -(w1*x0 + b - 1.0f) / w2},
            (Vector3){x1, 0, -(w1*x1 + b - 1.0f) / w2}, COLOR_BLUE);
        DrawLine3D(
            (Vector3){x0, 0, -(w1*x0 + b + 1.0f) / w2},
            (Vector3){x1, 0, -(w1*x1 + b + 1.0f) / w2}, COLOR_BLUE);

        // margin fill
        int steps = 50;
        Color fill = (Color){255, 255, 255, 15};
        for (int i = 0; i < steps; i++) {
            float xi = x0 + (x1 - x0) * i / steps;
            float xn = x0 + (x1 - x0) * (i + 1) / steps;
            float z_p  = -(w1*xi + b + 1.0f) / w2;
            float z_m  = -(w1*xi + b - 1.0f) / w2;
            float z_pn = -(w1*xn + b + 1.0f) / w2;
            float z_mn = -(w1*xn + b - 1.0f) / w2;
            DrawTriangle3D((Vector3){xi,0,z_p}, (Vector3){xn,0,z_pn}, (Vector3){xn,0,z_mn}, fill);
            DrawTriangle3D((Vector3){xi,0,z_p}, (Vector3){xn,0,z_mn}, (Vector3){xi,0,z_m},  fill);
            DrawTriangle3D((Vector3){xn,0,z_mn}, (Vector3){xn,0,z_pn}, (Vector3){xi,0,z_p}, fill);
            DrawTriangle3D((Vector3){xi,0,z_m},  (Vector3){xn,0,z_mn}, (Vector3){xi,0,z_p}, fill);
        }
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

float compute_accuracy(const Dataset *ds, const SVM *svm) {
    int correct = 0;
    for (int i = 0; i < ds->count; i++) {
        Sample s = ds->items[i];
        float prediction = svm->w1 * s.x + svm->w2 * s.z + svm->b;
        int predicted_class = prediction >= 0 ? 1 : -1;
        if (predicted_class == s.class) correct++;
    }
    return (float)correct / ds->count;
}

float compute_loss(const Dataset *ds, const SVM *svm) {
    float C = 1.0f;
    float loss = 0.5f * (svm->w1 * svm->w1 + svm->w2 * svm->w2); // regularization
    for (int i = 0; i < ds->count; i++) {
        Sample s = ds->items[i];
        float margin = s.class * (svm->w1 * s.x + svm->w2 * s.z + svm->b);
        float hinge = 1.0f - margin;
        if (hinge > 0) loss += C * hinge;
    }
    return loss / ds->count;
}


int main()
{
    lr = 0.0001;
    is_training = false;
    float delta = 0.01;
    srand(time(NULL));
    te = (TweenEngine){0};

    da_reserve(&te, 1024);

    Dataset dataset = {0};
    Dataset training_set = {0};

    prepare_training_dataset(&training_set);

    BoundingBox ground = { (Vector3){ -100, 0, -100 }, (Vector3){100, 0, 100} };

    SVM svm = {0};
    Camera camera = { 0 };
    camera.position = (Vector3){ -10.0f, 0.0f, 0.5f };
    camera.target = (Vector3){ 0.0f, -1.0f, 1.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    toggle_view_anim(&training_set, &camera, &view_mode);
    InitWindow(WIDTH, HEIGHT, "SVM");
    SetTargetFPS(60);
    HideCursor(); 
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
            toggle_view_anim(&training_set, &camera, &view_mode);

        if (IsKeyPressed(KEY_Q))
            is_training = !is_training;
        if (IsKeyPressed(KEY_I))
            svm_icr_w2(&svm, IsKeyDown(KEY_LEFT_SHIFT) ? -delta : delta);
        if (IsKeyPressed(KEY_O))
            svm_icr_w1(&svm, IsKeyDown(KEY_LEFT_SHIFT) ? -delta : delta);
        if (IsKeyPressed(KEY_P))
            svm_icr_b(&svm, IsKeyDown(KEY_LEFT_SHIFT) ? -delta : delta);

        if (is_training)
            train(&training_set, &svm);

        float loss = compute_loss(&training_set, &svm);
        float acc = compute_accuracy(&training_set, &svm);

        DrawText(TextFormat("Loss: %.4f | Accuracy: %.1f%%", loss, acc * 100.0f),
    WIDTH - 400, HEIGHT - 55, 20, GRAY);
        float smooth = 8.0f * dt;
        svm_visual.w1 = lerpf(svm_visual.w1, svm.w1, smooth);
        svm_visual.w2 = lerpf(svm_visual.w2, svm.w2, smooth);
        svm_visual.b  = lerpf(svm_visual.b,  svm.b,  smooth);

        BeginDrawing();
            ClearBackground(BACKGROUND_COLOR);
            BeginMode3D(camera);

                tween_update(&te, dt);
                if(view_mode == VIEW_2D)
                    DrawGrid(10, 1);        // Draw a grid
                draw_axes(view_mode);
                draw_dataset(&training_set, dt, true);
                draw_dataset(&dataset, dt, false);
                draw_svm(&svm_visual, view_mode);
            EndMode3D();
            // po EndMode3D:
            DrawText(TextFormat("LR: %.5f | W1: %.3f W2: %.3f B: %.3f | Loss: %.4f",
                        lr, svm.w1, svm.w2, svm.b, loss), 20, HEIGHT - 30, 20, GRAY);

            DrawText(is_training ? "TRAINING..." : "PAUSED [Q to train]", 20, HEIGHT - 55, 20, 
                    is_training ? COLOR_GREEN : COLOR_RED);
            DrawText("[T] Toggle view  [I/O/P] +w2/w1/b  [Shift+I/O/P] -w2/w1/b", 20, 20, 18, GRAY);
                draw_axis_labels(&camera, view_mode);
                
                draw_classes();
        EndDrawing();
    }
    CloseWindow();
    return 0;
}


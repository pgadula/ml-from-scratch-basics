#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "raylib.h"
#include "raymath.h"

#define NOB_IMPLEMENTATION
#include "nob.h"
#include "anim.h"

#if defined(PLATFORM_WEB)
#include <emscripten.h>
#endif

#define WIDTH  1920
#define HEIGHT 1024

#define POINT_RADIUS 0.1

#define BACKGROUND_COLOR (Color){0, 2, 8, 255}
#define COLOR_GRAY       (Color){80, 80, 80, 255}
#define COLOR_BLUE       (Color){88, 196, 221, 255}
#define COLOR_RED        (Color){255, 85, 85, 255}
#define COLOR_GREEN      (Color){130, 255, 100, 255}

#ifndef PI
#define PI 3.14159265358979323846f
#endif

TweenEngine te;

typedef enum {
    VIEW_2D = 0,
    VIEW_3D = 1
} VIEW_MODE;

typedef enum {
    CLASS_INNER = 1,  // blue - inner circle
    CLASS_OUTER = 2,  // red  - outer ring
} CIRCLE_LABEL;

VIEW_MODE view_mode = VIEW_2D;
bool kernel_applied = false;

typedef struct {
    Vector3 pos;
    Color color;
    float radius;
} Visual;

typedef struct {
    float x;
    float y;  // kernel dimension: x^2 + z^2
    float z;
    int label;
    Visual vis;
} Sample;

typedef struct {
    size_t capacity;
    size_t count;
    Sample *items;
} Dataset;

float randf(float min, float max) {
    return min + (float)rand() / RAND_MAX * (max - min);
}

void reset_points(Dataset *dataset) {
    dataset->count = 0;
}

Vector3 random_vec3(void) {
    return (Vector3){
        .x = randf(-10, 10),
        .y = randf(-10, 10),
        .z = randf(-10, 10)
    };
}

float axes_len = 0.0f;

void draw_axes(VIEW_MODE vm) {
    float len = axes_len;
    if (len < 0.01f) return;

    DrawLine3D((Vector3){-len, 0, 0}, (Vector3){ len, 0, 0}, RED);
    if (vm == VIEW_3D)
        DrawLine3D((Vector3){0, -len, 0}, (Vector3){0, len, 0}, GREEN);
    DrawLine3D((Vector3){0, 0, -len}, (Vector3){0, 0, len}, BLUE);

    int ticks = (int)len;
    for (int i = -ticks; i <= ticks; i++) {
        float t = 0.1f;
        DrawLine3D((Vector3){(float)i, -t, 0}, (Vector3){(float)i, t, 0}, COLOR_RED);
        if (vm == VIEW_3D)
            DrawLine3D((Vector3){-t, (float)i, 0}, (Vector3){t, (float)i, 0}, COLOR_GREEN);
        DrawLine3D((Vector3){0, -t, (float)i}, (Vector3){0, t, (float)i}, COLOR_BLUE);
    }
}



Color x_axes_labels = (Color){255, 85, 85, 0};
Color y_axes_labels = (Color){130, 255, 100, 0};
Color z_axes_labels = (Color){88, 196, 221, 0};

void draw_axis_labels(const Camera *camera, VIEW_MODE vm) {
    float len = 6.2f;
    Vector2 x_pos = GetWorldToScreen((Vector3){ len, 0, 0}, *camera);
    Vector2 y_pos = GetWorldToScreen((Vector3){ 0, len, 0}, *camera);
    Vector2 z_pos = GetWorldToScreen((Vector3){ 0, 0, len}, *camera);

    DrawText("X", (int)x_pos.x, (int)x_pos.y, 24, x_axes_labels);
    if (vm == VIEW_3D)
        DrawText("Y  (x\xc2\xb2 + z\xc2\xb2)", (int)y_pos.x, (int)y_pos.y, 24, y_axes_labels);
    DrawText("Z", (int)z_pos.x, (int)z_pos.y, 24, z_axes_labels);
}

/* ─── drawing helpers ─── */

void draw_dataset(const Dataset *td, bool is_training) {
    for (size_t i = 0; i < td->count; i++) {
        Sample entry = td->items[i];
        Visual vis = entry.vis;
        Vector3 pos = vis.pos;
        float r = vis.radius;
        Color color = vis.color;

        if (view_mode == VIEW_2D) pos.y = 0;

        if (is_training) {
            DrawSphere(pos, r, color);
        } else {
            float size = r * 1.2f;
            DrawCube(pos, size, size, size, color);
            if (entry.label == 0) {
                float pulse = 1.0f + 0.2f * sinf((float)GetTime() * 4.0f);
                float ps = size * 1.5f * pulse;
                DrawCube(pos, ps, ps, ps, (Color){255, 255, 255, 60});
            }
        }
    }
}

/* ─── separating plane ─── */

#define SEP_PLANE_HEIGHT 2.5f 
#define SEP_PLANE_SIZE   5.0f

float sep_plane_alpha = 0.0f;   /* 0 = hidden, animates to ~50 */
float sep_plane_y     = 30.0f;  /* current Y — starts above, sweeps down */
bool  sep_plane_shown = false;

void draw_separating_plane(void) {
    if (sep_plane_alpha < 1.0f) return;

    float h = sep_plane_y;
    float s = SEP_PLANE_SIZE;
    unsigned char a = (unsigned char)sep_plane_alpha;

    Color plane_color = (Color){255, 255, 0, a};

    /* front face */
    DrawTriangle3D(
        (Vector3){-s, h, -s}, (Vector3){ s, h, -s}, (Vector3){ s, h,  s}, plane_color);
    DrawTriangle3D(
        (Vector3){-s, h, -s}, (Vector3){ s, h,  s}, (Vector3){-s, h,  s}, plane_color);
    /* back face */
    DrawTriangle3D(
        (Vector3){ s, h,  s}, (Vector3){ s, h, -s}, (Vector3){-s, h, -s}, plane_color);
    DrawTriangle3D(
        (Vector3){-s, h,  s}, (Vector3){ s, h,  s}, (Vector3){-s, h, -s}, plane_color);

    /* edge lines for visibility */
    Color edge = (Color){255, 255, 0, (unsigned char)(a * 2 > 255 ? 255 : a * 2)};
    DrawLine3D((Vector3){-s, h, -s}, (Vector3){ s, h, -s}, edge);
    DrawLine3D((Vector3){ s, h, -s}, (Vector3){ s, h,  s}, edge);
    DrawLine3D((Vector3){ s, h,  s}, (Vector3){-s, h,  s}, edge);
    DrawLine3D((Vector3){-s, h,  s}, (Vector3){-s, h, -s}, edge);
}

void classify_by_plane(Dataset *ds) {
    for (size_t i = 0; i < ds->count; i++) {
        /* above plane → outer (red), below → inner (blue) */
        Color target;
        if (ds->items[i].y > SEP_PLANE_HEIGHT) {
            target = COLOR_RED;
        } else {
            target = COLOR_BLUE;
        }
        /* animate a brief white flash then to classified color */
        Tween *tc = tween_color(&te, &ds->items[i].vis.color, WHITE, 0.2f);
        (void)tc;
        Tween *tc2 = tween_color(&te, &ds->items[i].vis.color, target, 0.6f);
        tc2->elapsed = -0.3f;
    }
}

void restore_original_colors(Dataset *ds) {
    for (size_t i = 0; i < ds->count; i++) {
        Color original = (ds->items[i].label == CLASS_INNER) ? COLOR_BLUE : COLOR_RED;
        tween_color(&te, &ds->items[i].vis.color, original, 0.5f);
    }
}

void toggle_separating_plane(Dataset *ds) {
    if (!kernel_applied) return;  /* plane only makes sense in kernel space */

    if (!sep_plane_shown) {
        sep_plane_shown = true;

        sep_plane_y = 30.0f;
        tween_float(&te, &sep_plane_alpha, 50.0f, 0.5f);
        Tween *tw = tween_float(&te, &sep_plane_y, SEP_PLANE_HEIGHT, 2.0f);
        tw->ease = EASE_OUT_BOUNCE;

        classify_by_plane(ds);
    } else {
        sep_plane_shown = false;

        Tween *tw = tween_float(&te, &sep_plane_y, 30.0f, 1.0f);
        (void)tw;
        Tween *ta = tween_float(&te, &sep_plane_alpha, 0.0f, 1.0f);
        ta->elapsed = -0.5f;

        restore_original_colors(ds);
    }
}

void prepare_kernel_dataset(Dataset *td) {
    int n = 128; // points per class
    da_reserve(td, n * 2);

    for (int i = 0; i < n * 2; i++) {
        float angle = randf(0, 2.0f * PI);
        float r;
        int lab;

        if (i < n) {
            r = randf(0.0f, 2.0f);
            lab = CLASS_INNER;
        } else {
            r = randf(3.0f, 5.0f);
            lab = CLASS_OUTER;
        }

        float x = r * cosf(angle);
        float z = r * sinf(angle);
        float y_kernel = x * x + z * z;  // phi(x,z)

        Color color = (lab == CLASS_INNER) ? COLOR_BLUE : COLOR_RED;

        Sample sample = (Sample){
            .x = x,
            .y = y_kernel,
            .z = z,
            .label = lab,
            .vis = {
                .pos = random_vec3(),
                .radius = 0,
                .color = WHITE
            }
        };
        da_append(td, sample);

        float dur = 1.0f;
        Tween *t_v = tween_vec3(&te, &td->items[i].vis.pos,
                                (Vector3){ x, 0, z }, dur);
        Tween *t_r = tween_float(&te, &td->items[i].vis.radius, POINT_RADIUS, dur);
        Tween *t_c = tween_color(&te, &td->items[i].vis.color, color, dur);

        float delay = -randf(0.5f, 2.5f);
        t_v->elapsed = delay;
        t_r->elapsed = delay;
        t_r->ease    = EASE_OUT_BOUNCE;
        t_c->elapsed = delay;
    }
}


void cam_look_at(Camera *cam, Vector3 target) {
    tween_vec3(&te, &cam->target, target, 1);
}

Tween *cam_move(Camera *cam, Vector3 target) {
    return tween_vec3(&te, &cam->position, target, 1);
}

void cam_fovy(Camera *cam, float target) {
    tween_float(&te, &cam->fovy, target, 2);
}


void kernel_trick_toggle(Dataset *ds, Camera *camera, VIEW_MODE *vm) {
    if (!kernel_applied) {
        kernel_applied = true;
        *vm = VIEW_3D;

        for (size_t i = 0; i < ds->count; i++) {
            Tween *tw = tween_vec3(&te, &ds->items[i].vis.pos,
                (Vector3){ ds->items[i].x, ds->items[i].y, ds->items[i].z }, 1.5f);
            tw->ease = EASE_OUT_BOUNCE;
        }

        cam_move(camera, (Vector3){ 12, 18, 12 });
        cam_look_at(camera, (Vector3){ 0, 6, 0 });

    } else {
        kernel_applied = false;
        *vm = VIEW_2D;

        if (sep_plane_shown) {
            sep_plane_shown = false;
            tween_float(&te, &sep_plane_alpha, 0.0f, 0.5f);
            tween_float(&te, &sep_plane_y, 30.0f, 0.5f);
        }

        for (size_t i = 0; i < ds->count; i++) {
            tween_vec3(&te, &ds->items[i].vis.pos,
                (Vector3){ ds->items[i].x, 0, ds->items[i].z }, 1.5f);
        }
        restore_original_colors(ds);

        cam_move(camera, (Vector3){ 0.0f, 15.0f, 0.01f });
        cam_look_at(camera, (Vector3){ 0, 0, 0 });
    }
}


void toggle_view_anim(Dataset *ds, Camera *camera, VIEW_MODE *vm) {
    *vm ^= VIEW_3D;
    cam_look_at(camera, (Vector3){ 0, 0, 0 });

    if (*vm == VIEW_3D) {
        for (size_t i = 0; i < ds->count; i++) {
            float target_y = kernel_applied ? ds->items[i].y : 0.0f;
            Tween *tw = tween_vec3(&te, &ds->items[i].vis.pos,
                (Vector3){ ds->items[i].x, target_y, ds->items[i].z }, 1.0f);
            tw->ease = EASE_OUT_BOUNCE;
        }
        cam_move(camera, (Vector3){ 10, 10, 10 });
    } else {
        for (size_t i = 0; i < ds->count; i++) {
            tween_vec3(&te, &ds->items[i].vis.pos,
                (Vector3){ ds->items[i].x, 0, ds->items[i].z }, 1.0f);
        }
        cam_move(camera, (Vector3){ 0.0f, 15.0f, 0.01f });
    }
}

void draw_classes(void) {
    DrawText("INNER (blue)", WIDTH - 200, 20, 20, COLOR_BLUE);
    DrawText("OUTER (red)",  WIDTH - 200, 44, 20, COLOR_RED);
}

void draw_controls(void) {
    const int x = 20, y = 20, fs = 20, lh = fs + 6;
    int i = 0;
    DrawText("Controls",                                  x, y + lh * i++, fs + 6, RAYWHITE);
    DrawText("K - kernel trick (lift / flatten)",         x, y + lh * i++, fs, GRAY);
    DrawText("Q - separating plane (needs kernel)",       x, y + lh * i++, fs, GRAY);
    DrawText("T - toggle 2D / 3D view",                  x, y + lh * i++, fs, GRAY);
    DrawText("2D: Mouse Wheel - zoom",                    x, y + lh * i++, fs, GRAY);
    DrawText("3D: Free camera - WASD / Mouse",            x, y + lh * i++, fs, GRAY);
}

void draw_kernel_status(void) {
    const char *status = kernel_applied ? "KERNEL: ON  (phi = x^2 + z^2)" : "KERNEL: OFF (flat 2D)";
    Color col = kernel_applied ? YELLOW : GRAY;
    DrawText(status, 20, HEIGHT - 40, 28, col);
}

/* ─── globals ─── */

Dataset dataset       = {0}; 
Dataset training_set  = {0}; 
Camera  camera        = {0};
BoundingBox ground    = { (Vector3){-100, 0, -100}, (Vector3){100, 0, 100} };


void update_frame(void) {
    float dt = GetFrameTime();

    if (view_mode == VIEW_3D)
        UpdateCamera(&camera, CAMERA_FREE);
    else {
        float scroll = GetMouseWheelMove();
        if (scroll != 0) {
            tween_float(&te, &camera.fovy,
                        Clamp(camera.fovy - scroll * 3.0f, 10.0f, 90.0f), 0.3f);
        }
    }

    if (IsKeyPressed(KEY_K))
        kernel_trick_toggle(&training_set, &camera, &view_mode);

    if (IsKeyPressed(KEY_Q))
        toggle_separating_plane(&training_set);


    BeginDrawing();
    ClearBackground(BACKGROUND_COLOR);
    BeginMode3D(camera);

    tween_update(&te, dt);

    if (view_mode == VIEW_2D) DrawGrid(10, 1);

    draw_axes(view_mode);
    draw_separating_plane();
    draw_dataset(&training_set, true);

    EndMode3D();

    draw_axis_labels(&camera, view_mode);
    draw_controls();
    draw_kernel_status();
    draw_classes();

    EndDrawing();
}


int main(void) {
    srand((unsigned)time(NULL));

    te = (TweenEngine){0};
    da_reserve(&te, 1024);

    prepare_kernel_dataset(&training_set);

    camera.position   = (Vector3){ 0.0f, 15.0f, 0.01f };
    camera.target     = (Vector3){ 0.0f, 0.0f, 0.0f };
    camera.up         = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy       = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    /* axes entrance animation */
    Tween *tw = tween_float(&te, &axes_len, 6.0f, 2.0f);
    tw->ease    = EASE_OUT_BOUNCE;
    tw->elapsed = -0.5f;

    /* labels fade in */
    Tween *l1 = tween_alpha(&te, &x_axes_labels, 0, 255, 2);
    Tween *l2 = tween_alpha(&te, &y_axes_labels, 0, 255, 2);
    Tween *l3 = tween_alpha(&te, &z_axes_labels, 0, 255, 2);
    l1->elapsed = -1.5f;
    l2->elapsed = -1.5f;
    l3->elapsed = -1.5f;

    InitWindow(WIDTH, HEIGHT, "Kernel Trick Visualization");
    SetTargetFPS(60);
    SetMousePosition(WIDTH / 2, HEIGHT / 2);

#if defined(PLATFORM_WEB)
    emscripten_set_main_loop(update_frame, 0, 1);
#else
    while (!WindowShouldClose()) {
        update_frame();
    }
#endif

    CloseWindow();
    return 0;
}

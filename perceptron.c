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
#if defined(PLATFORM_WEB)
    #include <emscripten/emscripten.h>
#endif

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

float lr = 0.0001;
TweenEngine te;

typedef enum {
    VIEW_2D = 0,
    VIEW_3D = 1
} VIEW_MODE;


VIEW_MODE view_mode = VIEW_3D;

typedef struct {
    float *w;
    int num_weights;
    float b;
} Perceptron;


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


void cam_look_at(Camera *cam, Vector3 target){
    tween_vec3(&te, &cam->target, target, 1); 
}

Tween *cam_move(Camera *cam, Vector3 target){
    return tween_vec3(&te, &cam->position, target, 1); 
}

void cam_fovy(Camera *cam, float target){
    tween_float(&te, &cam->fovy, target, 2); 
}


//float activation_fn(float sum){
//    return sum > 0 ? 1 : 0;
//}

float activation_fn(float sum){
    return 1.0f / (1.0f + expf(-sum));
}

void train(Perceptron *p,  int sample_count, int input_count, float dataset[sample_count][input_count
], float *errors){
    float results = 0.0;
    *errors = 0;

    for(int i = 0; i < sample_count; i++){
        float *weights = p->w;
        float *data = dataset[i];
        float expected = data[input_count-1];
        float sum = 0;
        for(int j = 0; j < input_count-1; j++){
            sum += weights[j] * data[j]; 
        }
        sum += p->b;
        float output = activation_fn(sum);
        float error = output - expected;
        *errors += error * error;
        for(int j = 0; j < input_count-1; j++){
            p->w[j] -= lr * error * data[j];
        }

        p->b -= lr * error;
    }
    *errors =  *errors / (float)sample_count;
}

float cost(){
    return 0;
}

//training sets
float Base_Dataset[4][2] = {
    { 1, 2},
    { 2, 4},
    { 3, 6},
    { 4, 8}
};

float OR_Dataset[4][3] = {
    { 0, 1, 1},
    { 1, 0, 1},
    { 0, 0, 0},
    { 1, 1, 1}
};

float AND_Dataset[4][3] = {
    { 1, 0, 0},
    { 0, 1, 0},
    { 0, 0, 0},
    { 1, 1, 1}
};


float NAND_Dataset[4][3] = {
    { 0, 1, 1},
    { 1, 0, 1},
    { 0, 0, 1},
    { 1, 1, 0}
};

float XOR_Dataset[4][3] = {
    { 0, 1, 1},
    { 1, 0, 1},
    { 0, 0, 0},
    { 1, 1, 1}
};

float predict(const Perceptron *p, float *inputs) {
    float sum = 0;
    for (int j = 0; j < p->num_weights; j++)
        sum += p->w[j] * inputs[j];
    sum += p->b;
    return activation_fn(sum);
}

void init_perceptron(Perceptron *p, int input_size){
    p->w = malloc(sizeof(float) * input_size);
    p->num_weights = input_size;
    for(int i = 0; i < input_size; i++){
       p->w[i] = randf(0, 1); 
    }
}

void draw_dataset_points(int sample_count, int input_count, float dataset[][input_count], const Camera *camera) {
    for (int i = 0; i < sample_count; i++) {
        float x = dataset[i][0];
        float z = dataset[i][1] * i;
        float label = dataset[i][input_count - 1];

        Vector3 pos = (Vector3){ x, z, 0 };

        Color c = label > 0.5f ? COLOR_RED : COLOR_BLUE;
        DrawSphere(pos, POINT_RADIUS, c);
    }
}

void toggle_view_anim(Camera *camera, VIEW_MODE *view_mode) {
    *view_mode ^= VIEW_3D;

    cam_look_at(camera, (Vector3){ 0, 0, 0 });
    Tween *tw;
    if (*view_mode == VIEW_3D) {
        cam_look_at(camera, (Vector3){ 0, 0, 0 });
        cam_move(camera, (Vector3){ 10, 10, 10 });
    } else {
        cam_move(camera, (Vector3){ 0.0, 15, 0.01 });
        cam_look_at(camera, (Vector3){ 0, 0, 0 });
    }
}

float perc_anim_t = 0.0f;        // główny progress animacji struktury
float pulse_t = 0.0f;            // pulsowanie neuronu
float signal_t[3] = {0};         // animacja sygnału po połączeniach (max 3 inputy)
float output_signal_t = 0.0f;    // sygnał wyjściowy
bool perc_anim_started = false;

void start_perceptron_anim() {
    if (perc_anim_started) return;
    perc_anim_started = true;

    // Fade in struktury
    Tween *tw = tween_float(&te, &perc_anim_t, 1.0f, 1.5f);
    tw->ease = EASE_OUT_QUAD;

    // Sygnały po połączeniach z delayem
    for (int i = 0; i < 3; i++) {
        signal_t[i] = 0;
        Tween *s = tween_float(&te, &signal_t[i], 1.0f, 0.6f);
        s->ease = EASE_IN_OUT_QUAD;
        s->elapsed = -1.5f - i * 0.2f; // delay per connection
    }

    // Output signal
    output_signal_t = 0;
    Tween *out = tween_float(&te, &output_signal_t, 1.0f, 0.5f);
    out->ease = EASE_OUT_QUAD;
    out->elapsed = -2.5f;

    // Pulse neuron
    Tween *p = tween_float(&te, &pulse_t, 1.0f, 0.4f);
    p->ease = EASE_OUT_BOUNCE;
    p->elapsed = -2.3f;
}

void draw_perceptron_structure(const Perceptron *p, float *inputs, float output, float expected) {
    int x_start = WIDTH * 0.10;
    int y_start = HEIGHT * 0.30;
    int base_radius = 30;
    int spacing_y = 120;
    int layer_gap = 250;
    int n = p->num_weights;

    float alpha = perc_anim_t * 255;
    if (alpha < 1) return;

    int y_offset = y_start + 50;

    // === INPUT NODES ===
    Vector2 input_pos[n];
    for (int i = 0; i < n; i++) {
        input_pos[i] = (Vector2){ x_start, y_offset + i * spacing_y };

        // Pulse on signal arrival
        float r = base_radius + (signal_t[i] > 0.5f ? sinf(signal_t[i] * PI) * 5 : 0);

        Color col = COLOR_BLUE;
        col.a = (unsigned char)alpha;
        DrawCircleLines(input_pos[i].x, input_pos[i].y, r, col);

        // Glow when signal passes
        if (signal_t[i] > 0.01f && signal_t[i] < 0.99f) {
            Color glow = COLOR_BLUE;
            glow.a = (unsigned char)(sinf(signal_t[i] * PI) * 100);
            DrawCircle(input_pos[i].x, input_pos[i].y, r + 8, glow);
        }

      // New:
        char label[32];
        snprintf(label, sizeof(label), "x%d=%.2f", i, inputs[i]);
        Color label_col = COLOR_BLUE;
        label_col.a = (unsigned char)alpha;
        int tw = MeasureText(label, 20);
        DrawText(label, input_pos[i].x - tw/2, input_pos[i].y - 8, 20, label_col); 
    }

    Vector2 bias_pos = { x_start, y_offset - spacing_y };
    {
        Color col = COLOR_GRAY;
        col.a = (unsigned char)alpha;
        DrawCircleLines(bias_pos.x, bias_pos.y, base_radius, col);
        DrawText("b", bias_pos.x - 5, bias_pos.y - 8, 20, col);
    }

    // === NEURON (output node) ===
    float center_y = y_offset + (n - 1) * spacing_y / 2.0f;
    Vector2 neuron_pos = { x_start + layer_gap, center_y };
    {
        // Neuron pulses when activated
        float neuron_r = base_radius + pulse_t * sinf(GetTime() * 4) * 4;

        // Fill color interpolates based on output
        Color fill = lerp_color(COLOR_BLUE, COLOR_RED, output);
        fill.a = (unsigned char)(alpha * 0.8f);
        DrawCircle(neuron_pos.x, neuron_pos.y, neuron_r, fill);

        Color outline = WHITE;
        outline.a = (unsigned char)alpha;
        DrawCircleLines(neuron_pos.x, neuron_pos.y, neuron_r, outline);

        // Output value fades in
        if (output_signal_t > 0.1f) {
            char out_buf[32];
            snprintf(out_buf, sizeof(out_buf), "%.2f", output);
            Color txt = BLACK;
            txt.a = (unsigned char)(output_signal_t * 255);
            DrawText(out_buf, neuron_pos.x - 15, neuron_pos.y - 8, 18, txt);
        }
    }

    // === CONNECTIONS: input -> neuron ===
    for (int i = 0; i < n; i++) {
        float w = p->w[i];
        float thickness = fminf(fabsf(w) * 3.0f + 1.0f, 6.0f);
        Color c = w >= 0 ? COLOR_RED : COLOR_BLUE;
        c.a = (unsigned char)(alpha * 0.7f);

        Vector2 from = { input_pos[i].x + base_radius, input_pos[i].y };
        Vector2 to   = { neuron_pos.x - base_radius, neuron_pos.y };

        // Line grows in with perc_anim_t
        Vector2 current_to = {
            lerpf(from.x, to.x, perc_anim_t),
            lerpf(from.y, to.y, perc_anim_t),
        };
        DrawLineEx(from, current_to, thickness, c);

        // Animated signal dot traveling along connection
        if (signal_t[i] > 0.01f && signal_t[i] < 0.99f) {
            float st = signal_t[i];
            Vector2 dot = { lerpf(from.x, to.x, st), lerpf(from.y, to.y, st) };
            Color dot_col = YELLOW;
            dot_col.a = (unsigned char)(sinf(st * PI) * 255);
            DrawCircle(dot.x, dot.y, 6, dot_col);
        }

        // Weight label
        char w_buf[32];
        snprintf(w_buf, sizeof(w_buf), "%.3f", w);
        int mid_x = (input_pos[i].x + neuron_pos.x) / 2;
        int mid_y = (input_pos[i].y + neuron_pos.y) / 2 - 15;
        Color wc = c;
        wc.a = (unsigned char)alpha;
        DrawText(w_buf, mid_x, mid_y, 16, wc);
    }

    // === BIAS CONNECTION ===
    {
        float thickness = fminf(fabsf(p->b) * 3.0f + 1.0f, 6.0f);
        Color col = COLOR_GRAY;
        col.a = (unsigned char)(alpha * 0.7f);

        Vector2 from = { bias_pos.x + base_radius, bias_pos.y };
        Vector2 to   = { neuron_pos.x - base_radius, neuron_pos.y };
        Vector2 current_to = {
            lerpf(from.x, to.x, perc_anim_t),
            lerpf(from.y, to.y, perc_anim_t),
        };
        DrawLineEx(from, current_to, thickness, col);

        char b_buf[32];
        snprintf(b_buf, sizeof(b_buf), "%.3f", p->b);
        int mid_x = (bias_pos.x + neuron_pos.x) / 2;
        int mid_y = (bias_pos.y + neuron_pos.y) / 2 - 15;
        DrawText(b_buf, mid_x, mid_y, 16, col);
    }

    // === OUTPUT ARROW ===
    if (output_signal_t > 0.01f) {
        Vector2 from = { neuron_pos.x + base_radius, neuron_pos.y };
        Vector2 to   = { neuron_pos.x + layer_gap - 50, neuron_pos.y };
        Vector2 current_to = {
            lerpf(from.x, to.x, output_signal_t),
            lerpf(from.y, to.y, output_signal_t),
        };

        Color lc = WHITE;
        lc.a = (unsigned char)(output_signal_t * 255);
        DrawLineEx(from, current_to, 2, lc);

        if (output_signal_t > 0.8f) {
            char exp_buf[64];
            snprintf(exp_buf, sizeof(exp_buf), "out=%.2f exp=%.0f", output, expected);
            Color tc = WHITE;
            tc.a = (unsigned char)((output_signal_t - 0.8f) * 5.0f * 255);
            DrawText(exp_buf, to.x + 10, to.y - 10, 18, tc);
        }
    }

    // Title
    Color title = WHITE;
    title.a = (unsigned char)alpha;
    DrawText("Perceptron", x_start + 80, y_start - 50, 24, title);
}

void trigger_signal_anim() {
    for (int i = 0; i < 3; i++) {
        signal_t[i] = 0;
        Tween *s = tween_float(&te, &signal_t[i], 1.0f, 0.6f);
        s->ease = EASE_IN_OUT_QUAD;
        s->elapsed = -0.1f - i * 0.15f;
    }
    output_signal_t = 0;
    Tween *out = tween_float(&te, &output_signal_t, 1.0f, 0.5f);
    out->ease = EASE_OUT_QUAD;
    out->elapsed = -0.8f;

    pulse_t = 0;
    Tween *pt = tween_float(&te, &pulse_t, 1.0f, 0.4f);
    pt->ease = EASE_OUT_BOUNCE;
    pt->elapsed = -0.7f;
}


bool is_training_run = false;
float errors = 99.9;
Perceptron perceptron = {0};
int epochs = 1000;

Camera camera = { 0 };

void update_frame(){
        float dt = GetFrameTime(); 
        tween_update(&te, dt);
        if (IsKeyPressed(KEY_SPACE) || is_training_run) {
            for(int e = 0; e < epochs; e++)
                train(&perceptron, 4, 3, AND_Dataset, &errors);
            trigger_signal_anim();
        }
        if (IsKeyPressed(KEY_Q)) {
            is_training_run = !is_training_run;
        }

        if (IsKeyPressed(KEY_T)) {
            toggle_view_anim(&camera, &view_mode);
        }

        BeginDrawing();
            ClearBackground(BACKGROUND_COLOR);
            float test_input[] = {1, 1};
            float out = predict(&perceptron, test_input);
            draw_perceptron_structure(&perceptron, test_input, out, 1.0f);

            char buf[256];
            snprintf(buf, sizeof(buf), "w0=%.4f w1=%.4f b=%.4f error=%.4f", perceptron.w[0], perceptron.w[1], perceptron.b, errors);
            DrawText(buf, 20, 50, 20, COLOR_GREEN);
        EndDrawing();
}
int main()
{
    
    srand(time(NULL));

    te = (TweenEngine){0};
    da_reserve(&te, 1024);
    init_perceptron(&perceptron, 2);

    camera.position = (Vector3){ -10.0f, 0.0f, 0.5f };
    camera.target = (Vector3){ 0.0f, -1.0f, 1.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    Tween *tw = tween_float(&te, &axes_len, 6.0f, 2.0f);
    tw->ease = EASE_OUT_BOUNCE;
    tw->elapsed = -0.5;

    InitWindow(WIDTH, HEIGHT, "Perceptron");
    SetTargetFPS(60);

    SetMousePosition(WIDTH/2, HEIGHT/2);
    start_perceptron_anim();

#if defined(PLATFORM_WEB)
    emscripten_set_main_loop(update_frame, 0, 1);
#else
    while (!WindowShouldClose())
    {
        update_frame();
    }
#endif
    CloseWindow();
    return 0;
}


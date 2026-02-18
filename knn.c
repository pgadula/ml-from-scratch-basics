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
    int index;
    float d;
    int label;
    Vector3 pos;
} KNN_Entry;

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

int compare_entry(const void *a, const void *b) {
    KNN_Entry a1 = *(const KNN_Entry*)a;
    KNN_Entry a2 = *(const KNN_Entry*)b;
    float x = a1.d;
    float y = a2.d;
    return (x > y) - (x < y);  // avoids overflow
}

int compare_i(const void *a, const void *b) {
    int x = *(const int*)a;
    int y = *(const int*)b;
    return (x > y) - (x < y);
}

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

void knn_anim(int k, DIST_METRIC metric,  Dataset *ds, const Dataset *t)
{
    KNN_Entry neighbors[t->count];
    for (int i = 0; i < ds->count; i++){
        int voting[CLASS_COUNT] = {0};
        Sample curr = ds->items[i];
        Vector3 c_pos = curr.vis.pos;
        for (int j = 0; j < t->count; j++){
            Sample entry = t->items[j];
            Vector3 n_pos = entry.vis.pos; 
            float d = get_dist(metric, c_pos, n_pos);
            KNN_Entry knn_entry = { .index = j, .d = d, .label = entry.label, .pos = n_pos};
            neighbors[j] = knn_entry;
        }
        qsort(neighbors, t->count, sizeof(KNN_Entry), compare_entry);

        for (int n = 0; n < k; n++){
            KNN_Entry entry = neighbors[n];
            voting[(int)entry.label] += 1;

            ArrowData *ad = malloc(sizeof(ArrowData));
            *ad = (ArrowData){ .from = c_pos, .to = entry.pos, .color = FEATURES_COLORS[entry.label]};
            Tween *tw = tween_draw(&te, draw_arrow, 3.0, ad);
            tw->elapsed = -(0.5 * n);
            tw->owns_data = true;
            tw->hold = 2.0;
        }
       int best_class = 0;
       int max_votes = voting[0];

       for (int c = 1; c < CLASS_COUNT; c++) {
           if (voting[c] > max_votes) {
               max_votes = voting[c];
               best_class = c;
           }
       }

      ds->items[i].label = best_class;
      tween_color(&te, &ds->items[i].vis.color, CLASSIFIED_COLORS[best_class], 2.0); 
    }
}

void knn(int k, DIST_METRIC metric,  Dataset *ds, const Dataset *t)
{
    for (int i = 0; i < ds->count; i++){
        int voting[CLASS_COUNT] = {0};
        KNN_Entry neighbors[t->count];
        Sample curr = ds->items[i];
        Vector3 c_pos = (Vector3){curr.x, curr.y, curr.z};
        for (int j = 0; j < t->count; j++){
            Sample entry = t->items[j];
            Vector3 n_pos = (Vector3){entry.x, entry.y, entry.z};
            float d = get_dist(metric, c_pos, n_pos);
            neighbors[j] = (KNN_Entry){ .index = j, .d = d, .label = entry.label};
        }
        qsort(neighbors, t->count, sizeof(KNN_Entry), compare_entry);

        for (int n = 0; n < k; n++){
            KNN_Entry entry = neighbors[n];
            voting[(int)entry.label] += 1;
        }

       int best_class = 0;
       int max_votes = voting[0];

       for (int c = 1; c < CLASS_COUNT; c++) {
           if (voting[c] > max_votes) {
               max_votes = voting[c];
               best_class = c;
           }
       }

       ds->items[i].label = best_class;
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
    da_reserve(td, IRIS.count);

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

    td->count = IRIS.count;
    for(int i = 0; i < IRIS.count; i++){
        Row row = IRIS.data[i];
        float s_l = (row.sepal_length / max_sepal_length) / 12;
        float s_w = (row.sepal_width  / max_sepal_width ) * 10.0f - 5.0f;
        float p_l = (row.petal_length / max_petal_length) * 10.0f - 5.0f;
        float p_w = (row.petal_width  / max_petal_width ) * 10.0f - 5.0f;
        IRIS_LABEL label = map_label(row.variety);
        td->items[i] = (Sample){ 
            .x = p_w, .y = s_w, .z = p_l,
                .label = label, 
                .vis ={ 
                    .pos = random_vec3(),
                    .radius = 0, 
                    .color = WHITE           }
        };

        //animation
        float dur = 1.0;
        Color color = FEATURES_COLORS[map_label(row.variety)] ;
        Tween *t_v = tween_vec3(&te, &td->items[i].vis.pos, 
                (Vector3){ 
                .x = p_l, .y = s_w, .z = p_w
                }, dur
                );
        Tween *t_r = tween_float(&te, &td->items[i].vis.radius, s_l, dur);
        Tween *t_c = tween_color(&te, &td->items[i].vis.color, color, dur);

        t_v->elapsed = - 2;
        t_r->elapsed = - 2;
        t_r->ease = EASE_OUT_BOUNCE;
        t_c->elapsed = - 2;
    }
}



void draw_classes(){
    DrawText("SETOSA", WIDTH-150, 20, 20, FEATURES_COLORS[SETOSA]);
    DrawText("VIRGINICA", WIDTH-150, 40, 20, FEATURES_COLORS[VIRGINICA]);
    DrawText("VERSICOLOR", WIDTH-150, 60, 20, FEATURES_COLORS[VERSICOLOR]);
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

void animate_labels(){
}

void toggle_view_anim(Dataset *ds, Camera *camera, VIEW_MODE *view_mode) {
    *view_mode ^= VIEW_3D;

    cam_look_at(camera, (Vector3){ 0, 0, 0 });
    Tween *tw;
    if (*view_mode == VIEW_3D) {
        for (int i = 0; i < ds->count; i++) {
            tw = tween_vec3(&te, &ds->items[i].vis.pos, 
                    (Vector3){ ds->items[i].x, ds->items[i].y, ds->items[i].z }, 1.0f);
            tw->ease = EASE_OUT_BOUNCE;
        }
        cam_look_at(camera, (Vector3){ 0, 0, 0 });
        cam_move(camera, (Vector3){ 10, 10, 10 });
    } else {
        for (int i = 0; i < ds->count; i++) {
            tween_vec3(&te, &ds->items[i].vis.pos,
                    (Vector3){ ds->items[i].x, 0, ds->items[i].z }, 1.0f);
        }
        cam_move(camera, (Vector3){ 0.0, 15, 0.01 });
        cam_look_at(camera, (Vector3){ 0, 0, 0 });
    }
}

int main()
{
    srand(time(NULL));

    te = (TweenEngine){0};
    da_reserve(&te, 1024);

    Dataset dataset = {0};
    Dataset training_set = {0};

    prepare_training_dataset(&training_set);

    BoundingBox ground = { (Vector3){ -100, 0, -100 }, (Vector3){100, 0, 100} };

    Camera camera = { 0 };
    camera.position = (Vector3){ -10.0f, 0.0f, 0.5f };
    camera.target = (Vector3){ 0.0f, -1.0f, 1.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    toggle_view_anim(&training_set, &camera, &view_mode);
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

   InitWindow(WIDTH, HEIGHT, "KNN Playground");
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
            toggle_view_anim(&training_set, &camera, &view_mode);
        if (IsKeyPressed(KEY_K))
            knn_anim(5, view_mode, &dataset, &training_set);
        if (IsKeyPressed(KEY_R))
            reset_points(&dataset);
        if (IsKeyPressed(KEY_P))
            generate_points(&dataset);
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) || IsKeyPressed(KEY_ENTER)){
            Ray ray = GetMouseRay(GetMousePosition(), camera);
            RayCollision hit = GetRayCollisionBox(ray, ground);
            if(hit.hit)
            {
                Vector3 p = hit.point;
                Vector3 sp = {.x = p.x, .y = randf(-5, 5), .z = p.z};
                Sample sample = { .x = sp.x, .y = sp.y, .z = sp.z, 
                    .label = 0, 
                    .vis = { 
                        .pos = { .x = sp.x, .y = sp.y, .z = sp.z},
                        .radius = 0,
                        .color = WHITE }
                    };

                da_append(&dataset, sample);
                Sample *entry = &da_last(&dataset);
                tween_float(&te, &entry->vis.radius, POINT_RADIUS, 2.0) ;
            }
        }

        BeginDrawing();
            ClearBackground(BACKGROUND_COLOR);
            BeginMode3D(camera);

                tween_update(&te, dt);
                if(view_mode == VIEW_2D)
                    DrawGrid(10, 1);        // Draw a grid
                draw_axes(view_mode);
                draw_dataset(&training_set, dt, true);
                draw_dataset(&dataset, dt, false);
            EndMode3D();
                draw_axis_labels(&camera, view_mode);
                DrawText("SPACE - regenerate points", 20, 20, 20, GRAY);
                draw_classes();
        EndDrawing();
    }
    CloseWindow();
    return 0;
}


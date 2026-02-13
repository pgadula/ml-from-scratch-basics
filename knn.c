#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "raylib.h"
#define NOB_IMPLEMENTATION
#include "nob.h"
#include "raymath.h"
#include "iris.h"

#define WIDTH 1024
#define HEIGHT 1024

#define POINT_COUNT 512
#define CLASS_COUNT 4
#define POINT_RADIUS 0.1

#define BACKGROUND_COLOR (Color){20, 22, 28, 255}
#define COLOR_GRAY       (Color){80, 80, 80 ,255}
#define COLOR_BLUE       (Color){88, 196, 221, 255}
#define COLOR_RED        (Color){255, 85, 85, 255}
#define COLOR_GREEN      (Color){130, 255, 100, 255}


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
} KNN_Entry;

typedef struct {
    float x;
    float y;
    float z;
    IRIS_LABEL label;
} Sample;

typedef struct {
    size_t capacity;
    size_t count;
    Sample *items;
} Dataset;

Color CLASS_COLORS[CLASS_COUNT] = {
    COLOR_GRAY,
    COLOR_BLUE,
    COLOR_RED,
    COLOR_GREEN
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
        Sample pt = (Sample){.x = randf(0, WIDTH), .y = randf(0, WIDTH), .z = randf(0, WIDTH), .label = UNKNOWN};
        da_append(dataset, pt);
    }
}

void draw_axes(void) {
    float len = 6.0f;
    float label_offset = 0.3f;
    
    DrawLine3D(
        (Vector3){-len, 0, 0}, 
        (Vector3){ len, 0, 0}, 
        COLOR_RED
    );
    DrawLine3D(
        (Vector3){0, -len, 0}, 
        (Vector3){0,  len, 0}, 
        COLOR_GREEN
    );
    DrawLine3D(
        (Vector3){0, 0, -len}, 
        (Vector3){0, 0,  len}, 
        COLOR_BLUE
    );

    for (int i = -5; i <= 5; i++) {
        float t = 0.1f;
        DrawLine3D(
            (Vector3){i, -t, 0}, 
            (Vector3){i,  t, 0}, 
            COLOR_RED
        );
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

void draw_dataset(const Dataset *td){
    for (int i = 0; i < td->count; i++){
        Sample entry = td->items[i];
        Vector3 pos = { entry.x, entry.y, entry.z };
        if (view_mode == VIEW_2D)
            pos.y = 0;

        DrawSphere(
                pos,
                POINT_RADIUS,
                CLASS_COLORS[(int)entry.label]
                );
    }
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
        float s_l = (row.sepal_length / max_sepal_length) * 10.0f - 5.0f;
        float s_w = (row.sepal_width  / max_sepal_width ) * 10.0f - 5.0f;
        float p_l = (row.petal_length / max_petal_length) * 10.0f - 5.0f;
        float p_w = (row.petal_width  / max_petal_width ) * 10.0f - 5.0f;
        td->items[i] = (Sample){ .x = p_w, .y = s_w, .z = p_l,
            .label = map_label(row.variety)};
    }
}

void draw_classes(){
    DrawText("SETOSA", WIDTH-150, 20, 20, CLASS_COLORS[SETOSA]);
    DrawText("VIRGINICA", WIDTH-150, 40, 20, CLASS_COLORS[VIRGINICA]);
    DrawText("VERSICOLOR", WIDTH-150, 60, 20, CLASS_COLORS[VERSICOLOR]);
}

void draw_axis_labels(Camera camera) {
    float len = 16.2f;

    Vector2 x_pos = GetWorldToScreen((Vector3){ len, 0, 0}, camera);
    Vector2 y_pos = GetWorldToScreen((Vector3){ 0, len, 0}, camera);
    Vector2 z_pos = GetWorldToScreen((Vector3){ 0, 0, len}, camera);

    DrawText("petal_width",  (int)x_pos.x, (int)x_pos.y, 16, COLOR_RED);
    DrawText("sepal_width",  (int)y_pos.x, (int)y_pos.y, 16, COLOR_GREEN);
    DrawText("petal_length", (int)z_pos.x, (int)z_pos.y, 16, COLOR_BLUE);
}

int main()
{
    srand(time(NULL));

    Dataset dataset = {0};
    Dataset training_set = {0};

    prepare_training_dataset(&training_set);

    BoundingBox ground = { (Vector3){ -100, 0, -100 }, (Vector3){100, 0, 100} };

    // Define the camera to look into our 3d world
    Camera camera = { 0 };
    camera.position = (Vector3){ 0.0f, 20.0f, 0.5f };
    camera.target = (Vector3){ 0.0f, -1.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 60.0f;
    camera.projection = CAMERA_PERSPECTIVE;


    InitWindow(WIDTH, HEIGHT, "KNN Playground");
    SetTargetFPS(60);

    SetMousePosition(WIDTH/2, HEIGHT/2);
    while (!WindowShouldClose())
    {
        UpdateCamera(&camera, CAMERA_FREE);
        /* Input */
        if (IsKeyPressed(KEY_T))
            toggle_view(&view_mode);
        if (IsKeyPressed(KEY_K)){
            printf("press!");
            knn(7, view_mode, &dataset, &training_set);
        }
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
                Sample sample = { .x = p.x, .y = randf(-5, 5), .z = p.z, .label = 0 };
                da_append(&dataset, sample);
            }
        }

        /* Draw */
        BeginDrawing();
        ClearBackground(BACKGROUND_COLOR);

        BeginMode3D(camera);
        
            draw_axes();
            draw_axis_labels(camera);
            draw_dataset(&training_set);
            draw_dataset(&dataset);
           // DrawGrid(20, 1);        // Draw a grid
        EndMode3D();
            DrawText("SPACE - regenerate points", 20, 20, 20, GRAY);
            draw_classes();

        EndDrawing();
    }
    CloseWindow();
    return 0;
}


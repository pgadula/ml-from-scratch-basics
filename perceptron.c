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

#define POINT_RADIUS 0.1

#define BACKGROUND_COLOR (Color){0, 2, 8, 255}
#define COLOR_GRAY       (Color){80, 80, 80, 255}
#define COLOR_BLUE       (Color){88, 196, 221, 255}
#define COLOR_RED        (Color){255, 85, 85, 255}
#define COLOR_GREEN      (Color){130, 255, 100, 255}
#define COLOR_YELLOW     (Color){240, 192, 64, 255}
#define COLOR_SURFACE    (Color){13, 17, 23, 255}
#define COLOR_BORDER     (Color){26, 32, 48, 255}
#define COLOR_DIM        (Color){110, 122, 138, 255}

float lr = 0.1f;
TweenEngine te;

// ── Perceptron ──────────────────────────────────────────────

typedef struct {
    float *w;
    int num_weights;
    float b;
} Perceptron;

float activation_fn(float sum) {
    return 1.0f / (1.0f + expf(-sum));
}

float predict(const Perceptron *p, float *inputs) {
    float sum = 0;
    for (int j = 0; j < p->num_weights; j++)
        sum += p->w[j] * inputs[j];
    sum += p->b;
    return activation_fn(sum);
}

void init_perceptron(Perceptron *p, int input_size) {
    p->w = malloc(sizeof(float) * input_size);
    p->num_weights = input_size;
    p->b = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < input_size; i++)
        p->w[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

void reset_perceptron(Perceptron *p) {
    p->b = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < p->num_weights; i++)
        p->w[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

void train_step(Perceptron *p, int sample_count, int input_count,
                float dataset[][input_count], float *mse) {
    float total_error = 0;
    for (int i = 0; i < sample_count; i++) {
        float *data = dataset[i];
        float expected = data[input_count - 1];
        float sum = p->b;
        for (int j = 0; j < input_count - 1; j++)
            sum += p->w[j] * data[j];
        float output = activation_fn(sum);
        float error = output - expected;
        total_error += error * error;
        for (int j = 0; j < input_count - 1; j++)
            p->w[j] -= lr * error * data[j];
        p->b -= lr * error;
    }
    *mse = total_error / (float)sample_count;
}

// ── Datasets ────────────────────────────────────────────────

float AND_Dataset[4][3] = {
    {0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}
};
float OR_Dataset[4][3] = {
    {0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}
};
float NAND_Dataset[4][3] = {
    {0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}
};
float XOR_Dataset[4][3] = {
    {0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}
};

typedef struct {
    const char *name;
    float (*data)[3];
    int count;
} DatasetInfo;

DatasetInfo datasets[] = {
    {"AND",  AND_Dataset,  4},
    {"OR",   OR_Dataset,   4},
    {"NAND", NAND_Dataset, 4},
    {"XOR",  XOR_Dataset,  4},
};
int num_datasets = 4;
int current_dataset = 0;

// ── Error history ───────────────────────────────────────────

#define ERROR_HISTORY_MAX 500
float error_history[ERROR_HISTORY_MAX] = {0};
int error_history_count = 0;

void push_error(float e) {
    if (error_history_count < ERROR_HISTORY_MAX) {
        error_history[error_history_count++] = e;
    } else {
        memmove(error_history, error_history + 1, (ERROR_HISTORY_MAX - 1) * sizeof(float));
        error_history[ERROR_HISTORY_MAX - 1] = e;
    }
}

void reset_error_history(void) {
    error_history_count = 0;
    memset(error_history, 0, sizeof(error_history));
}

// ── Weight history (for animation) ─────────────────────────

typedef struct {
    float w0, w1, b;
} WeightSnapshot;

#define WEIGHT_HISTORY_MAX 200
WeightSnapshot weight_history[WEIGHT_HISTORY_MAX] = {0};
int weight_history_count = 0;

void push_weights(const Perceptron *p) {
    WeightSnapshot s = {p->w[0], p->w[1], p->b};
    if (weight_history_count < WEIGHT_HISTORY_MAX) {
        weight_history[weight_history_count++] = s;
    } else {
        memmove(weight_history, weight_history + 1, (WEIGHT_HISTORY_MAX - 1) * sizeof(WeightSnapshot));
        weight_history[WEIGHT_HISTORY_MAX - 1] = s;
    }
}

void reset_weight_history(void) {
    weight_history_count = 0;
}

// ── Perceptron animation state ──────────────────────────────

float perc_anim_t = 0.0f;
float pulse_t = 0.0f;
float signal_t[2] = {0};
float output_signal_t = 0.0f;

void start_perceptron_anim(void) {
    Tween *tw = tween_float(&te, &perc_anim_t, 1.0f, 1.5f);
    tw->ease = EASE_OUT_QUAD;

    for (int i = 0; i < 2; i++) {
        signal_t[i] = 0;
        Tween *s = tween_float(&te, &signal_t[i], 1.0f, 0.6f);
        s->ease = EASE_IN_OUT_QUAD;
        s->elapsed = -1.5f - i * 0.2f;
    }

    output_signal_t = 0;
    Tween *out = tween_float(&te, &output_signal_t, 1.0f, 0.5f);
    out->ease = EASE_OUT_QUAD;
    out->elapsed = -2.5f;

    pulse_t = 0;
    Tween *p = tween_float(&te, &pulse_t, 1.0f, 0.4f);
    p->ease = EASE_OUT_BOUNCE;
    p->elapsed = -2.3f;
}

void trigger_signal_anim(void) {
    for (int i = 0; i < 2; i++) {
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

// ── Drawing helpers ─────────────────────────────────────────

float lerpf_local(float a, float b, float t) {
    return a + (b - a) * t;
}

void draw_panel(int x, int y, int w, int h, const char *title) {
    DrawRectangle(x, y, w, h, COLOR_SURFACE);
    DrawRectangleLines(x, y, w, h, COLOR_BORDER);
    if (title) {
        DrawText(title, x + 10, y + 8, 18, COLOR_BLUE);
        DrawLine(x, y + 30, x + w, y + 30, COLOR_BORDER);
    }
}

float conn_pulse_phase = 0.0f;

void draw_perceptron_structure(const Perceptron *p, float *inputs,
                                float output, float expected,
                                int ox, int oy, int panel_w, int panel_h) {
    draw_panel(ox, oy, panel_w, panel_h, "PERCEPTRON");

    int n = p->num_weights;

    // Scale everything to panel size
    int pad_x = panel_w * 0.08f;
    int pad_y = 90;
    int usable_w = panel_w - pad_x * 2;
    int usable_h = panel_h - pad_y - 40;

    int base_radius = panel_h * 0.07f;
    if (base_radius < 20) base_radius = 20;
    if (base_radius > 60) base_radius = 60;

    int neuron_radius = base_radius * 1.6f;
    int spacing_y = usable_h * 0.35f;
    int layer_gap = usable_w * 0.55f;

    int x_start = ox + pad_x + base_radius + 40;
    int y_center = oy + pad_y + usable_h / 2;

    float alpha = perc_anim_t * 255;
    if (alpha < 1) return;

    int font_big = panel_h > 500 ? 22 : 16;
    int font_med = panel_h > 500 ? 18 : 14;
    int font_sm  = panel_h > 500 ? 15 : 12;

    // ── Input nodes ──
    Vector2 input_pos[2];
    for (int i = 0; i < n; i++) {
        input_pos[i] = (Vector2){x_start, y_center + (i - 0.5f) * spacing_y};
        float r = base_radius;

        // Pulse on signal
        float pulse = 0;
        if (signal_t[i] > 0.01f && signal_t[i] < 0.99f)
            pulse = sinf(signal_t[i] * PI) * base_radius * 0.15f;
        r += pulse;

        // Outer ring
        Color col = COLOR_BLUE;
        col.a = (unsigned char)alpha;
        DrawCircleLines(input_pos[i].x, input_pos[i].y, r, col);
        DrawCircleLines(input_pos[i].x, input_pos[i].y, r - 1, col);

        // Inner glow on signal
        if (signal_t[i] > 0.01f && signal_t[i] < 0.99f) {
            float glow_a = sinf(signal_t[i] * PI);
            Color glow = COLOR_BLUE;
            glow.a = (unsigned char)(glow_a * 60);
            DrawCircle(input_pos[i].x, input_pos[i].y, r + base_radius * 0.3f, glow);
            glow.a = (unsigned char)(glow_a * 30);
            DrawCircle(input_pos[i].x, input_pos[i].y, r + base_radius * 0.6f, glow);
        }

        // Fill subtle
        Color fill = COLOR_BLUE;
        fill.a = (unsigned char)(alpha * 0.1f);
        DrawCircle(input_pos[i].x, input_pos[i].y, r - 2, fill);

        // Label
        char label[32];
        snprintf(label, sizeof(label), "x%d = %.0f", i, inputs[i]);
        Color lc = COLOR_BLUE;
        lc.a = (unsigned char)alpha;
        int tw = MeasureText(label, font_med);
        DrawText(label, input_pos[i].x - tw / 2, input_pos[i].y - font_med / 2, font_med, lc);
    }

    // ── Bias node ──
    float bias_r = base_radius * 0.7f;
    Vector2 bias_pos = {x_start, y_center - spacing_y * 0.5f - spacing_y * 0.6f};
    {
        Color col = COLOR_GRAY;
        col.a = (unsigned char)alpha;
        DrawCircleLines(bias_pos.x, bias_pos.y, bias_r, col);
        Color fill = COLOR_GRAY;
        fill.a = (unsigned char)(alpha * 0.08f);
        DrawCircle(bias_pos.x, bias_pos.y, bias_r - 2, fill);

        char bb[16];
        snprintf(bb, sizeof(bb), "bias");
        int tw = MeasureText(bb, font_sm);
        DrawText(bb, bias_pos.x - tw / 2, bias_pos.y - font_sm / 2, font_sm, col);
    }

    // ── Neuron (big, central) ──
    Vector2 neuron_pos = {x_start + layer_gap, y_center};
    {
        // Activation glow rings
        float act = output; // 0..1
        int num_rings = 3;
        for (int r = num_rings; r >= 1; r--) {
            float ring_r = neuron_radius + r * neuron_radius * 0.25f;
            float ring_alpha = act * 40.0f / r;
            // Pulse effect
            float pulse_mod = 1.0f + pulse_t * sinf(GetTime() * 5.0f + r) * 0.08f;
            ring_r *= pulse_mod;
            Color glow = (Color){255, 255, 255, (unsigned char)ring_alpha};
            DrawCircleLines(neuron_pos.x, neuron_pos.y, ring_r, glow);
        }

        float nr = neuron_radius + pulse_t * sinf(GetTime() * 4) * neuron_radius * 0.05f;
        unsigned char v = (unsigned char)(act * 255);
        Color fill = {v, v, v, (unsigned char)(alpha * 0.95f)};
        DrawCircle(neuron_pos.x, neuron_pos.y, nr, fill);

        if (act > 0.1f) {
            unsigned char v2 = (unsigned char)fminf(v * 1.3f, 255);
            Color inner = {v2, v2, v2, (unsigned char)(alpha * 0.5f)};
            DrawCircle(neuron_pos.x, neuron_pos.y, nr * 0.5f, inner);
        }

        // Outline
        Color outline = WHITE;
        outline.a = (unsigned char)(alpha * 0.8f);
        DrawCircleLines(neuron_pos.x, neuron_pos.y, nr, outline);
        DrawCircleLines(neuron_pos.x, neuron_pos.y, nr - 1, outline);

        // Sigma symbol
        Color sigma_col = act > 0.5f ? BLACK : WHITE;
        sigma_col.a = (unsigned char)(alpha * 0.6f);
        int sigma_fs = neuron_radius * 0.5f;
        const char *sigma = "\xcf\x83"; // σ in UTF-8
        int stw = MeasureText(sigma, sigma_fs);
        DrawText(sigma, neuron_pos.x - stw / 2, neuron_pos.y - sigma_fs * 0.7f, sigma_fs, sigma_col);

        // Output value below sigma
        if (output_signal_t > 0.1f) {
            char out_buf[16];
            snprintf(out_buf, sizeof(out_buf), "%.3f", output);
            Color txt = act > 0.5f ? BLACK : WHITE;
            txt.a = (unsigned char)(output_signal_t * 255);
            int otw = MeasureText(out_buf, font_med);
            DrawText(out_buf, neuron_pos.x - otw / 2, neuron_pos.y + sigma_fs * 0.3f, font_med, txt);
        }
    }

    for (int i = 0; i < n; i++) {
        float w = p->w[i];
        float abs_w = fabsf(w);
        float thickness = fminf(abs_w * 4.0f + 1.5f, 8.0f);
        Color c = w >= 0 ? COLOR_RED : COLOR_BLUE;
        c.a = (unsigned char)(alpha * 0.5f);

        Vector2 from = {input_pos[i].x + base_radius + 2, input_pos[i].y};
        Vector2 to = {neuron_pos.x - neuron_radius - 2, neuron_pos.y};
        Vector2 cur = {lerpf_local(from.x, to.x, perc_anim_t),
                       lerpf_local(from.y, to.y, perc_anim_t)};
        DrawLineEx(from, cur, thickness, c);

        // Continuous pulsing particles along connection
        int num_particles = 3;
        float conn_len = sqrtf((to.x - from.x) * (to.x - from.x) + (to.y - from.y) * (to.y - from.y));
        for (int pp = 0; pp < num_particles; pp++) {
            float phase = fmodf(conn_pulse_phase + (float)pp / num_particles + i * 0.3f, 1.0f);
            Vector2 dot = {lerpf_local(from.x, to.x, phase),
                           lerpf_local(from.y, to.y, phase)};
            float dot_alpha = sinf(phase * PI) * abs_w * 0.8f;
            if (dot_alpha > 1.0f) dot_alpha = 1.0f;
            Color dc = c;
            dc.a = (unsigned char)(dot_alpha * 180);
            float dot_r = 3.0f + abs_w * 2.0f;
            DrawCircle(dot.x, dot.y, dot_r, dc);
        }

        if (signal_t[i] > 0.01f && signal_t[i] < 0.99f) {
            float st = signal_t[i];
            Vector2 dot = {lerpf_local(from.x, to.x, st), lerpf_local(from.y, to.y, st)};
            float flash_a = sinf(st * PI);
            Color dc = COLOR_YELLOW;
            dc.a = (unsigned char)(flash_a * 255);
            DrawCircle(dot.x, dot.y, 8, dc);
            // Trail glow
            Color trail = COLOR_YELLOW;
            trail.a = (unsigned char)(flash_a * 80);
            DrawCircle(dot.x, dot.y, 16, trail);
        }

        char w_buf[16];
        snprintf(w_buf, sizeof(w_buf), "w%d=%.3f", i, w);
        float label_t = 0.35f;
        int mx = lerpf_local(from.x, to.x, label_t);
        int my = lerpf_local(from.y, to.y, label_t) - 18;
        Color wc = c;
        wc.a = (unsigned char)alpha;
        DrawText(w_buf, mx, my, font_sm, wc);
    }

    {
        float abs_b = fabsf(p->b);
        float thickness = fminf(abs_b * 3.0f + 1.0f, 6.0f);
        Color col = COLOR_GRAY;
        col.a = (unsigned char)(alpha * 0.4f);
        Vector2 from = {bias_pos.x + bias_r + 2, bias_pos.y};
        Vector2 to = {neuron_pos.x - neuron_radius - 2, neuron_pos.y};
        Vector2 cur = {lerpf_local(from.x, to.x, perc_anim_t),
                       lerpf_local(from.y, to.y, perc_anim_t)};
        DrawLineEx(from, cur, thickness, col);

        // Particles
        for (int pp = 0; pp < 2; pp++) {
            float phase = fmodf(conn_pulse_phase * 0.7f + (float)pp / 2.0f, 1.0f);
            Vector2 dot = {lerpf_local(from.x, to.x, phase),
                           lerpf_local(from.y, to.y, phase)};
            float dot_alpha = sinf(phase * PI) * abs_b * 0.6f;
            if (dot_alpha > 1.0f) dot_alpha = 1.0f;
            Color dc = COLOR_GRAY;
            dc.a = (unsigned char)(dot_alpha * 120);
            DrawCircle(dot.x, dot.y, 2.5f + abs_b * 1.5f, dc);
        }

        char b_buf[16];
        snprintf(b_buf, sizeof(b_buf), "b=%.3f", p->b);
        float label_t = 0.35f;
        int mx = lerpf_local(from.x, to.x, label_t);
        int my = lerpf_local(from.y, to.y, label_t) - 18;
        DrawText(b_buf, mx, my, font_sm, col);
    }

    // ── Output arrow ──
    if (output_signal_t > 0.01f) {
        Vector2 from = {neuron_pos.x + neuron_radius + 2, neuron_pos.y};
        Vector2 to = {neuron_pos.x + neuron_radius + 120, neuron_pos.y};
        Vector2 cur = {lerpf_local(from.x, to.x, output_signal_t), from.y};
        Color lc = WHITE;
        lc.a = (unsigned char)(output_signal_t * 200);
        DrawLineEx(from, cur, 2.5f, lc);

        // Arrow head
        if (output_signal_t > 0.5f) {
            float ah = 8;
            DrawTriangle(
                (Vector2){cur.x, cur.y},
                (Vector2){cur.x - ah, cur.y - ah * 0.5f},
                (Vector2){cur.x - ah, cur.y + ah * 0.5f},
                lc
            );
        }

        if (output_signal_t > 0.7f) {
            float fade = (output_signal_t - 0.7f) * 3.33f;
            char result[32];
            snprintf(result, sizeof(result), "%.2f", output);
            Color rc = output > 0.5f ? COLOR_GREEN : COLOR_RED;
            rc.a = (unsigned char)(fade * 255);
            DrawText(result, to.x + 10, to.y - font_big / 2, font_big, rc);

            char eb[32];
            snprintf(eb, sizeof(eb), "(expected %.0f)", expected);
            Color ec = COLOR_DIM;
            ec.a = (unsigned char)(fade * 180);
            DrawText(eb, to.x + 10, to.y + font_big / 2 + 4, font_sm, ec);
        }
    }

    // ── Activation function label ──
    {
        Color ac = COLOR_DIM;
        ac.a = (unsigned char)(alpha * 0.5f);
        DrawText("sigmoid", neuron_pos.x - MeasureText("sigmoid", font_sm) / 2,
                 neuron_pos.y + neuron_radius + 8, font_sm, ac);
    }
}

// ── Draw: Error chart ───────────────────────────────────────

void draw_error_chart(int ox, int oy, int w, int h) {
    draw_panel(ox, oy, w, h, "ERROR (MSE)");

    int pad = 12;
    int chart_x = ox + pad;
    int chart_y = oy + 38;
    int chart_w = w - pad * 2;
    int chart_h = h - 50;

    if (error_history_count < 2) return;

    // Find max for scaling
    float max_e = 0.001f;
    for (int i = 0; i < error_history_count; i++)
        if (error_history[i] > max_e) max_e = error_history[i];

    // Grid lines
    for (int i = 0; i <= 4; i++) {
        float y_frac = (float)i / 4.0f;
        int yy = chart_y + (int)(y_frac * chart_h);
        DrawLine(chart_x, yy, chart_x + chart_w, yy, (Color){30, 40, 55, 255});
        char lbl[16];
        snprintf(lbl, sizeof(lbl), "%.3f", max_e * (1.0f - y_frac));
        DrawText(lbl, chart_x + 2, yy - 12, 12, COLOR_DIM);
    }

    // Draw line
    int count = error_history_count;
    int start = count > chart_w ? count - chart_w : 0;
    int visible = count - start;

    for (int i = 1; i < visible; i++) {
        float v0 = error_history[start + i - 1] / max_e;
        float v1 = error_history[start + i] / max_e;
        int x0 = chart_x + (i - 1) * chart_w / visible;
        int x1 = chart_x + i * chart_w / visible;
        int y0 = chart_y + chart_h - (int)(v0 * chart_h);
        int y1 = chart_y + chart_h - (int)(v1 * chart_h);
        DrawLine(x0, y0, x1, y1, COLOR_RED);
    }

    // Current value
    if (count > 0) {
        char cur[32];
        snprintf(cur, sizeof(cur), "%.6f", error_history[count - 1]);
        DrawText(cur, chart_x + chart_w - 100, chart_y + 4, 14, COLOR_YELLOW);
    }
}

// ── Draw: Prediction table ──────────────────────────────────

void draw_prediction_table(const Perceptron *p, float dataset[][3],
                           int count, int ox, int oy, int w, int h) {
    draw_panel(ox, oy, w, h, "PREDICTIONS");

    int row_h = 28;
    int y = oy + 38;
    int col_w = w / 4;

    // Header
    DrawText("x0", ox + 10, y, 16, COLOR_DIM);
    DrawText("x1", ox + col_w, y, 16, COLOR_DIM);
    DrawText("exp", ox + col_w * 2, y, 16, COLOR_DIM);
    DrawText("out", ox + col_w * 3, y, 16, COLOR_DIM);
    y += row_h;
    DrawLine(ox, y - 4, ox + w, y - 4, COLOR_BORDER);

    for (int i = 0; i < count; i++) {
        float inp[2] = {dataset[i][0], dataset[i][1]};
        float expected = dataset[i][2];
        float out = predict(p, inp);
        float err = fabsf(out - expected);

        char b0[8], b1[8], be[8], bo[8];
        snprintf(b0, sizeof(b0), "%.0f", inp[0]);
        snprintf(b1, sizeof(b1), "%.0f", inp[1]);
        snprintf(be, sizeof(be), "%.0f", expected);
        snprintf(bo, sizeof(bo), "%.2f", out);

        Color row_col = err > 0.3f ? COLOR_RED : COLOR_GREEN;

        DrawText(b0, ox + 10, y, 16, COLOR_DIM);
        DrawText(b1, ox + col_w, y, 16, COLOR_DIM);
        DrawText(be, ox + col_w * 2, y, 16, WHITE);
        DrawText(bo, ox + col_w * 3, y, 16, row_col);

        y += row_h;
    }
}

// ── Draw: Heatmap matrix ────────────────────────────────────

#define HEATMAP_RES 32

void draw_heatmap(const Perceptron *p, int ox, int oy, int w, int h) {
    draw_panel(ox, oy, w, h, "PREDICTION SPACE");

    int pad = 12;
    int map_x = ox + pad + 20;
    int map_y = oy + 42;
    int map_w = w - pad * 2 - 30;
    int map_h = h - 60;

    float cell_w = (float)map_w / HEATMAP_RES;
    float cell_h = (float)map_h / HEATMAP_RES;

    for (int iy = 0; iy < HEATMAP_RES; iy++) {
        for (int ix = 0; ix < HEATMAP_RES; ix++) {
            float x_in = (float)ix / (HEATMAP_RES - 1);
            float y_in = 1.0f - (float)iy / (HEATMAP_RES - 1);
            float inp[2] = {x_in, y_in};
            float out = predict(p, inp);

            unsigned char v = (unsigned char)(out * 255);
            Color c = {v, v, v, 255};
            DrawRectangle(map_x + (int)(ix * cell_w),
                          map_y + (int)(iy * cell_h),
                          (int)cell_w + 1, (int)cell_h + 1, c);
        }
    }

    // Axis labels
    DrawText("0", map_x - 16, map_y + map_h - 10, 14, COLOR_DIM);
    DrawText("1", map_x - 16, map_y - 2, 14, COLOR_DIM);
    DrawText("0", map_x - 2, map_y + map_h + 4, 14, COLOR_DIM);
    DrawText("1", map_x + map_w - 6, map_y + map_h + 4, 14, COLOR_DIM);
    DrawText("x0", map_x + map_w / 2 - 8, map_y + map_h + 4, 14, COLOR_DIM);
    DrawText("x1", map_x - 20, map_y + map_h / 2 - 6, 14, COLOR_DIM);

    // Draw dataset points on heatmap
    DatasetInfo *ds = &datasets[current_dataset];
    for (int i = 0; i < ds->count; i++) {
        float px = ds->data[i][0];
        float py = ds->data[i][1];
        float label = ds->data[i][2];

        int sx = map_x + (int)(px * map_w);
        int sy = map_y + (int)((1.0f - py) * map_h);

        Color dot_c = label > 0.5f ? COLOR_GREEN : COLOR_RED;
        DrawCircle(sx, sy, 6, dot_c);
        DrawCircleLines(sx, sy, 6, WHITE);
    }
}


void draw_controls(int ox, int oy, bool is_training) {
    DrawText("CONTROLS", ox, oy, 18, COLOR_BLUE);
    oy += 26;

    Color kc = COLOR_YELLOW;
    Color tc = COLOR_DIM;
    int fs = 16;
    int gap = 22;

    DrawText("[1-4]", ox, oy, fs, kc);
    DrawText("Select dataset", ox + 60, oy, fs, tc);
    oy += gap;

    DrawText("[SPACE]", ox, oy, fs, kc);
    DrawText("Train step", ox + 80, oy, fs, tc);
    oy += gap;

    DrawText("[Q]", ox, oy, fs, kc);
    char tbuf[32];
    snprintf(tbuf, sizeof(tbuf), "Auto-train: %s", is_training ? "ON" : "OFF");
    DrawText(tbuf, ox + 40, oy, fs, is_training ? COLOR_GREEN : tc);
    oy += gap;

    DrawText("[+/-]", ox, oy, fs, kc);
    char lrbuf[32];
    snprintf(lrbuf, sizeof(lrbuf), "LR: %.4f", lr);
    DrawText(lrbuf, ox + 60, oy, fs, tc);
    oy += gap;

    DrawText("[R]", ox, oy, fs, kc);
    DrawText("Reset weights", ox + 40, oy, fs, tc);
    oy += gap + 8;

    // Dataset indicator
    DrawText("DATASET:", ox, oy, 16, COLOR_BLUE);
    oy += 22;
    for (int i = 0; i < num_datasets; i++) {
        Color c = (i == current_dataset) ? COLOR_GREEN : COLOR_DIM;
        char db[16];
        snprintf(db, sizeof(db), "[%d] %s", i + 1, datasets[i].name);
        DrawText(db, ox, oy, 16, c);
        oy += 20;
    }
}

// ── Main ────────────────────────────────────────────────────

bool is_training_run = false;
float current_error = 1.0f;
Perceptron perceptron = {0};
int epochs_per_frame = 100;
int total_epochs = 0;

void switch_dataset(int idx) {
    current_dataset = idx;
    reset_perceptron(&perceptron);
    reset_error_history();
    reset_weight_history();
    total_epochs = 0;
    current_error = 1.0f;
    is_training_run = false;
}

void update_frame(void) {
    float dt = GetFrameTime();
    tween_update(&te, dt);

    // Input
    if (IsKeyPressed(KEY_ONE))   switch_dataset(0);
    if (IsKeyPressed(KEY_TWO))   switch_dataset(1);
    if (IsKeyPressed(KEY_THREE)) switch_dataset(2);
    if (IsKeyPressed(KEY_FOUR))  switch_dataset(3);

    if (IsKeyPressed(KEY_Q)) is_training_run = !is_training_run;

    if (IsKeyPressed(KEY_R)) {
        reset_perceptron(&perceptron);
        reset_error_history();
        reset_weight_history();
        total_epochs = 0;
        current_error = 1.0f;
    }

    if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) {
        lr *= 2.0f;
        if (lr > 10.0f) lr = 10.0f;
    }
    if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) {
        lr *= 0.5f;
        if (lr < 0.00001f) lr = 0.00001f;
    }

    bool do_train = IsKeyPressed(KEY_SPACE) || is_training_run;
    if (do_train) {
        DatasetInfo *ds = &datasets[current_dataset];
        for (int e = 0; e < epochs_per_frame; e++) {
            train_step(&perceptron, ds->count, 3, ds->data, &current_error);
            total_epochs++;
        }
        push_error(current_error);
        trigger_signal_anim();
    }

    // ── Layout: 2 columns ─────────────────────────────────
    // Left: big perceptron structure (half screen)
    // Right top: heatmap, right mid: error chart, right bottom: table + controls

    int margin = 16;
    int top_y = margin + 36;
    int content_h = HEIGHT - top_y - margin;

    int col1_x = margin;
    int col1_w = WIDTH / 2 - margin;

    int col2_x = WIDTH / 2 + margin;
    int col2_w = WIDTH - col2_x - margin;

    // Update connection pulse phase
    conn_pulse_phase = fmodf(conn_pulse_phase + GetFrameTime() * 0.4f, 1.0f);

    BeginDrawing();
    ClearBackground(BACKGROUND_COLOR);

    // Title bar
    char title[128];
    snprintf(title, sizeof(title), "Perceptron — %s — Epoch: %d",
             datasets[current_dataset].name, total_epochs);
    DrawText(title, margin, margin, 24, WHITE);

    // Col 1: Big perceptron structure (full left half)
    float test_inputs[2] = {1, 1};
    float test_out = predict(&perceptron, test_inputs);
    draw_perceptron_structure(&perceptron, test_inputs, test_out,
                              datasets[current_dataset].data[3][2],
                              col1_x, top_y, col1_w, content_h);

    // Col 2 top: Heatmap
    int heatmap_h = col2_w; // square-ish
    if (heatmap_h > content_h * 0.48f) heatmap_h = content_h * 0.48f;
    draw_heatmap(&perceptron, col2_x, top_y, col2_w, heatmap_h);

    // Col 2 mid: Error chart
    int remaining = content_h - heatmap_h - margin;
    int error_h = remaining * 0.45f;
    draw_error_chart(col2_x, top_y + heatmap_h + margin, col2_w, error_h);

    // Col 2 bottom: Prediction table + Controls side by side
    int bottom_y = top_y + heatmap_h + margin + error_h + margin;
    int bottom_h = HEIGHT - bottom_y - margin;
    int table_w = col2_w * 0.5f;
    int ctrl_w = col2_w - table_w - margin;

    draw_prediction_table(&perceptron, datasets[current_dataset].data,
                          datasets[current_dataset].count,
                          col2_x, bottom_y, table_w, bottom_h);

    draw_controls(col2_x + table_w + margin + 10, bottom_y + 10, is_training_run);

    EndDrawing();
}

int main(void) {
    srand(time(NULL));

    te = (TweenEngine){0};
    da_reserve(&te, 1024);
    init_perceptron(&perceptron, 2);

    InitWindow(WIDTH, HEIGHT, "Perceptron");
    SetTargetFPS(60);
    SetMousePosition(WIDTH / 2, HEIGHT / 2);

    start_perceptron_anim();

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

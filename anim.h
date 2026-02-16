#include <stdlib.h>
#include "raylib.h"
#include "raymath.h"

typedef enum {
    TWEEN_FLOAT,
    TWEEN_DRAW,
    TWEEN_VEC3,
    TWEEN_COLOR,
} TweenType;

typedef enum {
    EASE_LINEAR,
    EASE_IN_QUAD,
    EASE_OUT_QUAD,
    EASE_IN_OUT_QUAD,
    EASE_OUT_BOUNCE,
} EaseFunc;

typedef struct {
    bool active;
    bool completed;
    bool owns_data;
    TweenType type;
    EaseFunc ease;
    void (*draw)(float t, void *userdata);
    void *target;
    
    union {
        struct { float from, to; } f;
        struct { Vector3 from, to; } v3;
        struct { Color from, to; } col;
    };

    float duration;
    float hold;
    float elapsed;
    void (*on_complete)(void *userdata);
    void *userdata;
} Tween;

typedef struct {
    int count;
    int capacity;
    Tween *items;
}   TweenEngine;

float ease(EaseFunc f, float t) {
    switch (f) {
        case EASE_LINEAR: return t;
        case EASE_IN_QUAD: return t * t;
        case EASE_OUT_QUAD: return t * (2 - t);
        case EASE_IN_OUT_QUAD:
            return t < 0.5f 
                ? 2 * t * t 
                : -1 + (4 - 2 * t) * t;
        case EASE_OUT_BOUNCE: {
            if (t < 1/2.75f)
                return 7.5625f * t * t;
            else if (t < 2/2.75f)
                return 7.5625f * (t -= 1.5f/2.75f) * t + 0.75f;
            else if (t < 2.5f/2.75f)
                return 7.5625f * (t -= 2.25f/2.75f) * t + 0.9375f;
            else
                return 7.5625f * (t -= 2.625f/2.75f) * t + 0.984375f;
        }
        default: return t;
    }
}

float lerpf(float a, float b, float t) { return a + (b - a) * t; }

Vector3 lerp_vec3(Vector3 a, Vector3 b, float t) {
    return (Vector3){
        lerpf(a.x, b.x, t),
        lerpf(a.y, b.y, t),
        lerpf(a.z, b.z, t),
    };
}

Color lerp_color(Color a, Color b, float t) {
    return (Color){
        (unsigned char)lerpf(a.r, b.r, t),
        (unsigned char)lerpf(a.g, b.g, t),
        (unsigned char)lerpf(a.b, b.b, t),
        (unsigned char)lerpf(a.a, b.a, t),
    };
}

Tween* tween_add(TweenEngine *e) {
    if (e->count >= e->capacity) return NULL;
    Tween *tw = &e->items[e->count++]; 
    *tw = (Tween){0};
    tw->active = true;
    tw->hold = 0;
    tw->ease = EASE_OUT_QUAD;
    return tw;
}


Tween *tween_draw(TweenEngine *e, void* draw, float duration, void *userdata) {
    Tween *tw = tween_add(e);
    if (!tw) return tw;
    tw->type = TWEEN_DRAW;
    tw->draw = draw;
    tw->userdata = userdata;
    tw->duration = duration;
    return tw;
}

Tween *tween_float(TweenEngine *e, float *target, float to, float duration) {
    Tween *tw = tween_add(e);
    if (!tw) return tw;
    tw->type = TWEEN_FLOAT;
    tw->target = target;
    tw->f.from = *target;
    tw->f.to = to;
    tw->duration = duration;
    return tw;
}

Tween *tween_vec3(TweenEngine *e, Vector3 *target, Vector3 to, float duration) {
    Tween *tw = tween_add(e);
    if (!tw) return tw;
    tw->type = TWEEN_VEC3;
    tw->target = target;
    tw->v3.from = *target;
    tw->v3.to = to;
    tw->duration = duration;
    return tw;
}

Tween *tween_color(TweenEngine *e, Color *target, Color to, float duration) {
    Tween *tw = tween_add(e);
    if (!tw) return tw;
    tw->type = TWEEN_COLOR;
    tw->target = target;
    tw->col.from = *target;
    tw->col.to = to;
    tw->duration = duration;
    return tw;
}

void tween_float_ex(TweenEngine *e, float *target, float to, 
                     float duration, EaseFunc ef, 
                     void (*on_complete)(void*), void *userdata) {
    Tween *tw = tween_add(e);
    if (!tw) return;
    tw->type = TWEEN_FLOAT;
    tw->target = target;
    tw->f.from = *target;
    tw->f.to = to;
    tw->duration = duration;
    tw->ease = ef;
    tw->on_complete = on_complete;
    tw->userdata = userdata;
}

void tween_update(TweenEngine *e, float dt) {
    for (int i = e->count - 1; i >= 0; i--) {
        Tween *tw = &e->items[i];
        if (!tw->active) continue;

        tw->elapsed += dt;
        if(tw->elapsed < 0) continue;
        float t = tw->elapsed / tw->duration;
        if (t >= 1.0f) t = 1.0f;
        
        float et = ease(tw->ease, t);

        switch (tw->type) {
            case TWEEN_DRAW:
                tw->draw(et, tw->userdata); 
                break;
            case TWEEN_FLOAT:
                *(float*)tw->target = lerpf(tw->f.from, tw->f.to, et);
                break;
            case TWEEN_VEC3:
                *(Vector3*)tw->target = lerp_vec3(tw->v3.from, tw->v3.to, et);
                break;
            case TWEEN_COLOR:
                *(Color*)tw->target = lerp_color(tw->col.from, tw->col.to, et);
                break;
        }

        if (tw->elapsed >= tw->duration + tw->hold) {
            if (tw->on_complete) tw->on_complete(tw->userdata);
            if(tw->owns_data && tw->userdata) free(tw->userdata);
            tw->userdata = NULL; 
            tw->active = false;
            e->items[i] = e->items[--e->count];
        }
    }
}

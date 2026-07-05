// Hanayo sprites — raw PNG decoded by LVGL lodepng
#ifndef HANAYO_SPRITES_H
#define HANAYO_SPRITES_H

#include <lvgl.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

extern const lv_image_dsc_t hanayo_neutral;
extern const lv_image_dsc_t hanayo_happy;
extern const lv_image_dsc_t hanayo_sad;
extern const lv_image_dsc_t hanayo_angry;
extern const lv_image_dsc_t hanayo_shocked;
extern const lv_image_dsc_t hanayo_sleepy;
extern const lv_image_dsc_t hanayo_laughing;
extern const lv_image_dsc_t hanayo_confused;
extern const lv_image_dsc_t hanayo_winking;
extern const lv_image_dsc_t hanayo_listen;
extern const lv_image_dsc_t hanayo_loving;
extern const lv_image_dsc_t hanayo_crying;

static inline const lv_image_dsc_t* hanayo_get_sprite(const char* emotion) {
    if (strcmp(emotion, "neutral") == 0) {
        return &hanayo_neutral;
    } else if (strcmp(emotion, "happy") == 0) {
        return &hanayo_happy;
    } else if (strcmp(emotion, "sad") == 0) {
        return &hanayo_sad;
    } else if (strcmp(emotion, "angry") == 0) {
        return &hanayo_angry;
    } else if (strcmp(emotion, "shocked") == 0) {
        return &hanayo_shocked;
    } else if (strcmp(emotion, "sleepy") == 0) {
        return &hanayo_sleepy;
    } else if (strcmp(emotion, "laughing") == 0) {
        return &hanayo_laughing;
    } else if (strcmp(emotion, "confused") == 0) {
        return &hanayo_confused;
    } else if (strcmp(emotion, "winking") == 0) {
        return &hanayo_winking;
    } else if (strcmp(emotion, "listen") == 0) {
        return &hanayo_listen;
    } else if (strcmp(emotion, "loving") == 0) {
        return &hanayo_loving;
    } else if (strcmp(emotion, "crying") == 0) {
        return &hanayo_crying;
    }
    return NULL;
}

#ifdef __cplusplus
}
#endif
#endif
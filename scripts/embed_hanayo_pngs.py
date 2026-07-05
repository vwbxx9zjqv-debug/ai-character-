#!/usr/bin/env python3
"""Embed raw Hanayo PNG files as C arrays — no conversion, LVGL decodes PNGs natively.

LVGL auto-detects PNG by magic bytes and decodes via lodepng (enabled in sdkconfig).
"""

import os
from pathlib import Path

PNG_DIR = os.path.expanduser("~/Desktop/hanayo_sprites_esp32")
OUTPUT_DIR = os.path.expanduser("~/xiaozhi-esp32/main/display/hanayo_sprites")

EMOTION_MAP = {
    "neutral":   "neutral.png",
    "happy":     "happy.png",
    "sad":       "sad.png",
    "angry":     "angry.png",
    "shocked":   "shocked.png",
    "sleepy":    "sleepy.png",
    "laughing":  "laughing.png",
    "confused":  "confused.png",
    "winking":   "winking.png",
    "listen":    "listen.png",
    "loving":    "loving.png",
    "crying":    "crying.png",
}


def generate_c(name, png_bytes):
    """Generate a .c file with raw PNG data as lv_image_dsc_t."""
    lines = []
    lines.append(f'// Hanayo sprite: {name} (raw PNG embedded)')
    lines.append('#include <lvgl.h>')
    lines.append('')
    lines.append('#ifndef LV_ATTRIBUTE_MEM_ALIGN')
    lines.append('#define LV_ATTRIBUTE_MEM_ALIGN')
    lines.append('#endif')
    lines.append('')
    # Embed raw PNG bytes
    lines.append('static const')
    lines.append('LV_ATTRIBUTE_MEM_ALIGN LV_ATTRIBUTE_LARGE_CONST')
    lines.append(f'uint8_t hanayo_{name}_png[] = {{')
    for i in range(0, len(png_bytes), 16):
        chunk = png_bytes[i:i + 16]
        hex_str = ', '.join(f'0x{b:02x}' for b in chunk)
        lines.append(f'    {hex_str},')
    lines.append('};')
    lines.append('')
    # lv_image_dsc_t pointing to raw PNG; LVGL decodes it on first use
    lines.append(f'const lv_image_dsc_t hanayo_{name} = {{')
    lines.append(f'  .header.cf = LV_COLOR_FORMAT_RAW,')
    lines.append(f'  .header.magic = LV_IMAGE_HEADER_MAGIC,')
    lines.append(f'  .data_size = sizeof(hanayo_{name}_png),')
    lines.append(f'  .data = hanayo_{name}_png,')
    lines.append(f'}};')
    lines.append('')
    return '\n'.join(lines)


def generate_header(emotions):
    """Generate header with declarations and lookup function."""
    lines = []
    lines.append('// Hanayo sprites — raw PNG decoded by LVGL lodepng')
    lines.append('#ifndef HANAYO_SPRITES_H')
    lines.append('#define HANAYO_SPRITES_H')
    lines.append('')
    lines.append('#include <lvgl.h>')
    lines.append('#include <string.h>')
    lines.append('')
    lines.append('#ifdef __cplusplus')
    lines.append('extern "C" {')
    lines.append('#endif')
    lines.append('')
    for name in emotions:
        lines.append(f'extern const lv_image_dsc_t hanayo_{name};')
    lines.append('')
    lines.append('static inline const lv_image_dsc_t* hanayo_get_sprite(const char* emotion) {')
    for i, name in enumerate(emotions):
        prefix = 'if' if i == 0 else '} else if'
        lines.append(f'    {prefix} (strcmp(emotion, "{name}") == 0) {{')
        lines.append(f'        return &hanayo_{name};')
    lines.append('    }')
    lines.append('    return NULL;')
    lines.append('}')
    lines.append('')
    lines.append('#ifdef __cplusplus')
    lines.append('}')
    lines.append('#endif')
    lines.append('#endif')
    return '\n'.join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    emotions_done = []
    total_size = 0

    for emotion, filename in EMOTION_MAP.items():
        png_path = os.path.join(PNG_DIR, filename)
        if not os.path.exists(png_path):
            print(f"  SKIP {emotion}: {png_path} not found")
            continue

        with open(png_path, 'rb') as f:
            png_bytes = f.read()

        c_source = generate_c(emotion, png_bytes)
        c_path = os.path.join(OUTPUT_DIR, f'hanayo_{emotion}.c')
        with open(c_path, 'w') as f:
            f.write(c_source)

        size_kb = len(png_bytes) / 1024
        print(f"  {emotion}: {len(png_bytes)} bytes ({size_kb:.1f} KB) → {c_path}")
        emotions_done.append(emotion)
        total_size += len(png_bytes)

    header = generate_header(emotions_done)
    header_path = os.path.join(OUTPUT_DIR, 'hanayo_sprites.h')
    with open(header_path, 'w') as f:
        f.write(header)

    print(f"\nDone: {len(emotions_done)} sprites, {total_size / 1024:.1f} KB total (raw PNG)")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()

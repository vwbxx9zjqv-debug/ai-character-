#!/usr/bin/env python3
"""Convert Hanayo PNG sprites to LVGL-compatible C source files.

Each sprite is 158x240 RGBA PNG → RGB565A8 format (3 bytes/pixel: 2B color + 1B alpha)
Output: a single .c file with lv_image_dsc_t for each emotion + a mapping header.
"""

import os
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Need Pillow: pip install Pillow")
    sys.exit(1)

PNG_DIR = os.path.expanduser("~/Desktop/hanayo_sprites_esp32")
OUTPUT_DIR = os.path.expanduser("~/xiaozhi-esp32/main/display/hanayo_sprites")

# Emotion name → filename mapping (matches the emotion strings sent by server)
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


def rgba_to_rgb565a8(r, g, b, a):
    """Convert RGBA pixel to RGB565 + alpha."""
    r5 = (r >> 3) & 0x1F
    g6 = (g >> 2) & 0x3F
    b5 = (b >> 3) & 0x1F
    color = (r5 << 11) | (g6 << 5) | b5
    return [color & 0xFF, (color >> 8) & 0xFF, a]


def convert_png(png_path):
    """Convert a single PNG to RGB565A8 byte array."""
    img = Image.open(png_path).convert("RGBA")
    w, h = img.size
    pixels = img.load()
    data = bytearray()
    for y in range(h):
        for x in range(w):
            r, g, b, a = pixels[x, y]
            data.extend(rgba_to_rgb565a8(r, g, b, a))
    return bytes(data), w, h


def generate_c_array(name, w, h, raw_data):
    """Generate C source string for a single image."""
    stride = w * 3  # RGB565A8 = 3 bytes/pixel
    lines = []
    lines.append(f'// Hanayo sprite: {name} ({w}x{h})')
    lines.append(f'#include "lvgl/lvgl.h"')
    lines.append('')
    lines.append(f'#ifndef LV_ATTRIBUTE_MEM_ALIGN')
    lines.append(f'#define LV_ATTRIBUTE_MEM_ALIGN')
    lines.append(f'#endif')
    lines.append('')
    lines.append(f'static const')
    lines.append(f'LV_ATTRIBUTE_MEM_ALIGN LV_ATTRIBUTE_LARGE_CONST')
    lines.append(f'uint8_t hanayo_{name}_map[] = {{')
    lines.append('')

    # Format as hex bytes, 16 per line
    for i in range(0, len(raw_data), 16):
        chunk = raw_data[i:i+16]
        hex_str = ', '.join(f'0x{b:02x}' for b in chunk)
        lines.append(f'    {hex_str},')

    lines.append('};')
    lines.append('')
    lines.append(f'const lv_image_dsc_t hanayo_{name} = {{')
    lines.append(f'  .header.magic = LV_IMAGE_HEADER_MAGIC,')
    lines.append(f'  .header.cf = LV_COLOR_FORMAT_RGB565A8,')
    lines.append(f'  .header.flags = 0,')
    lines.append(f'  .header.w = {w},')
    lines.append(f'  .header.h = {h},')
    lines.append(f'  .header.stride = {stride},')
    lines.append(f'  .data_size = sizeof(hanayo_{name}_map),')
    lines.append(f'  .data = hanayo_{name}_map,')
    lines.append(f'}};')
    lines.append('')
    return '\n'.join(lines)


def generate_header(emotions):
    """Generate the header file declaring all sprites."""
    lines = []
    lines.append('// Auto-generated Hanayo sprite declarations')
    lines.append('#ifndef HANAYO_SPRITES_H')
    lines.append('#define HANAYO_SPRITES_H')
    lines.append('')
    lines.append('#include "lvgl/lvgl.h"')
    lines.append('')
    lines.append('#ifdef __cplusplus')
    lines.append('extern "C" {')
    lines.append('#endif')
    lines.append('')
    for name in emotions:
        lines.append(f'extern const lv_image_dsc_t hanayo_{name};')
    lines.append('')
    lines.append('// Lookup function: returns NULL if emotion not found')
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
    lines.append('')
    lines.append('#endif // HANAYO_SPRITES_H')
    return '\n'.join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    emotions_done = []

    # Generate individual .c files for each sprite
    for emotion, filename in EMOTION_MAP.items():
        png_path = os.path.join(PNG_DIR, filename)
        if not os.path.exists(png_path):
            print(f"  SKIP {emotion}: {png_path} not found")
            continue

        print(f"  Converting {emotion} ({filename})...")
        raw_data, w, h = convert_png(png_path)
        c_source = generate_c_array(emotion, w, h, raw_data)

        c_path = os.path.join(OUTPUT_DIR, f'hanayo_{emotion}.c')
        with open(c_path, 'w') as f:
            f.write(c_source)

        size_kb = len(raw_data) / 1024
        print(f"    → {c_path} ({w}x{h}, {size_kb:.1f} KB)")
        emotions_done.append(emotion)

    # Generate the combined header
    header = generate_header(emotions_done)
    header_path = os.path.join(OUTPUT_DIR, 'hanayo_sprites.h')
    with open(header_path, 'w') as f:
        f.write(header)
    print(f"  Header → {header_path}")

    # Print summary
    total_files = len(emotions_done)
    total_size = sum(
        EMOTION_MAP[name].split('.')[0] for name in emotions_done
    ) if False else sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f'hanayo_{name}.c'))
        for name in emotions_done
    )
    print(f"\nDone! {total_files} sprites generated ({total_size / 1024:.1f} KB total)")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()

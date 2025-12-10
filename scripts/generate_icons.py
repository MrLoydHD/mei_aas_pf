#!/usr/bin/env python3
"""
Generate PNG icons for the Chrome extension from SVG.
Requires: cairosvg (pip install cairosvg)
"""

import os
import sys

def generate_icons():
    try:
        import cairosvg
    except ImportError:
        print("Installing cairosvg...")
        os.system(f"{sys.executable} -m pip install cairosvg")
        import cairosvg

    icons_dir = os.path.join(os.path.dirname(__file__), '..', 'extension', 'icons')

    sizes = [16, 48, 128]

    for size in sizes:
        svg_path = os.path.join(icons_dir, f'icon{size}.svg')
        png_path = os.path.join(icons_dir, f'icon{size}.png')

        if os.path.exists(svg_path):
            print(f"Converting icon{size}.svg -> icon{size}.png")
            cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=size, output_height=size)
        else:
            print(f"Warning: {svg_path} not found")

    print("Done! Icons generated in extension/icons/")

if __name__ == '__main__':
    generate_icons()

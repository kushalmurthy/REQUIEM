"""
Run this after downloading the MAD dataset to see its actual folder
structure and class names/counts. Don't assume structure -- check it.

Usage: python inspect_dataset.py
"""

import os

ROOT = "data/MAD"


def walk_and_report(root):
    if not os.path.exists(root):
        print(f"'{root}' does not exist yet -- did the download/unzip finish?")
        return

    print(f"Contents of {root}:\n")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.replace(root, "").count(os.sep)
        indent = "  " * depth
        audio_files = [f for f in filenames if f.lower().endswith((".wav", ".mp3", ".flac"))]
        label = f"{os.path.basename(dirpath) or root}"
        if audio_files:
            print(f"{indent}{label}/  -> {len(audio_files)} audio files")
        else:
            print(f"{indent}{label}/")
        # don't descend forever into huge trees, just show structure clearly
        if depth > 3:
            dirnames.clear()


if __name__ == "__main__":
    walk_and_report(ROOT)

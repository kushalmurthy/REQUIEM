"""
Loads noise files + labels from the MAD dataset's annotation CSVs
(datalabeltrain.csv / datalabeltes.csv), for plugging into train.py.

CLASS NAMES BELOW ARE CONFIRMED (0 and 2 confirmed by listening to samples;
1, 3, 4, 5, 6 inferred with high confidence from source-video titles --
see conversation for methodology).
"""

import os
import csv

MAD_CLASS_NAMES = {
    0: "communication",       # confirmed by listening -- radio/voice comms chatter
    1: "gunshot_range",        # high confidence
    2: "footsteps",            # confirmed by listening
    3: "artillery_shelling",   # high confidence
    4: "tank_vehicle",         # high confidence
    5: "helicopter",           # high confidence
    6: "aircraft_jet",         # high confidence
}


def load_mad_files(csv_path, audio_root):
    """
    csv_path: path to datalabeltrain.csv or datalabeltes.csv
    audio_root: directory containing the 'training/' or 'test/' folder
                referenced by the CSV's 'path' column
                e.g. if CSV says 'training/398/0.wav' and the real file is at
                'data/MAD/MAD_dataset/training/398/0.wav', pass
                audio_root='data/MAD/MAD_dataset'

    Returns: (files, labels) -- parallel lists, files are full paths,
             labels are ints 0-6.
    """
    files, labels = [], []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            full_path = os.path.join(audio_root, row['path'])
            files.append(full_path)
            labels.append(int(row['label']))
    return files, labels


def verify_files_exist(files, sample_size=20):
    """Quick check: do the first N referenced files actually exist on disk?
    Run this before training -- catches path/root mismatches early."""
    import random
    sample = random.sample(files, min(sample_size, len(files)))
    missing = [f for f in sample if not os.path.exists(f)]
    if missing:
        print(f"WARNING: {len(missing)}/{len(sample)} sampled files not found. Examples:")
        for m in missing[:5]:
            print(" ", m)
        print("Check that audio_root matches your actual folder structure.")
    else:
        print(f"OK: all {len(sample)} sampled files found on disk.")
    return len(missing) == 0


if __name__ == "__main__":
    # adjust audio_root to match your actual extracted folder structure
    AUDIO_ROOT = "data/MAD/MAD_dataset"

    train_files, train_labels = load_mad_files("datalabeltrain.csv", AUDIO_ROOT)
    print(f"Loaded {len(train_files)} training noise files across "
          f"{len(set(train_labels))} classes")

    verify_files_exist(train_files)

    from collections import Counter
    counts = Counter(train_labels)
    for label_id in sorted(counts):
        name = MAD_CLASS_NAMES.get(label_id, "unknown")
        print(f"  label {label_id} ({name}): {counts[label_id]} files")

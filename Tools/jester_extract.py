#!/usr/bin/env python3
"""Extract only selected Jester clips from a prefix of Qualcomm's TGZ stream.

The numbered archive parts are consecutive byte ranges of one gzip stream. A
prefix can therefore be decompressed up to its end without downloading all
22.8 GB. The final, potentially incomplete clip is ignored.
"""

import argparse
import csv
import json
import tarfile
from collections import Counter
from pathlib import Path


LABELS = (
    "Swiping Left",
    "Swiping Right",
    "Swiping Up",
    "Swiping Down",
    "Stop Sign",
    "Thumb Up",
    "No gesture",
    "Doing other things",
)


def load_labels(root: Path):
    labels = {}
    for split in ("train", "validation"):
        with (root / "jester_labels" / f"{split}.csv").open() as handle:
            for clip_id, label in csv.reader(handle, delimiter=";"):
                if label in LABELS:
                    labels[clip_id] = (split, label)
    return labels


def extract(archive: Path, root: Path, train_per_class: int, val_per_class: int):
    labels = load_labels(root)
    quotas = {"train": train_per_class, "validation": val_per_class}
    counts = Counter()
    chosen = {}
    frame_counts = Counter()
    clips_dir = root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    with archive.open("rb") as stream:
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            while True:
                try:
                    member = tar.next()
                except (EOFError, OSError, tarfile.ReadError):
                    break  # A partial download ends inside the gzip stream.
                if member is None:
                    break
                parts = Path(member.name).parts
                if len(parts) < 2 or parts[0] != "20bn-jester-v1":
                    continue
                clip_id = parts[1]
                info = labels.get(clip_id)
                if info is None:
                    continue
                split, label = info
                key = (split, label)
                if clip_id not in chosen:
                    if counts[key] >= quotas[split]:
                        continue
                    chosen[clip_id] = {"id": clip_id, "split": split, "label": label}
                    counts[key] += 1
                if (len(parts) != 3 or not member.isfile()
                        or not parts[2].endswith(".jpg")
                        or not parts[2][:-4].isdigit()):
                    continue
                target = clips_dir / clip_id / parts[2]
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    source = tar.extractfile(member)
                    if source is None:
                        continue
                    target.write_bytes(source.read())
                frame_counts[clip_id] += 1

    # The last clip of a truncated stream can be incomplete. Full Jester clips
    # are about 36 frames; 16 is the minimum needed by the feature extractor.
    selected = [item for clip_id, item in chosen.items() if frame_counts[clip_id] >= 16]
    selected.sort(key=lambda item: (item["split"], item["label"], int(item["id"])))
    manifest = root / f"manifest_{train_per_class}_{val_per_class}.json"
    manifest.write_text(json.dumps(selected, indent=2) + "\n")
    available = Counter((item["split"], item["label"]) for item in selected)
    for split in ("train", "validation"):
        for label in LABELS:
            print(f"{split:10s} {label:20s} {available[split, label]:4d} / {quotas[split]}")
    print(f"Manifest: {manifest}")
    if any(available[split, label] < quotas[split]
           for split in quotas for label in LABELS):
        raise SystemExit("Archive prefix does not contain enough complete clips")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--root", type=Path, default=Path("Data/jester"))
    parser.add_argument("--train-per-class", type=int, required=True)
    parser.add_argument("--val-per-class", type=int, default=30)
    args = parser.parse_args()
    extract(args.archive, args.root, args.train_per_class, args.val_per_class)

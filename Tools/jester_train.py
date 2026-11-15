#!/usr/bin/env python3
"""Train and evaluate an eight-class Jester gesture classifier.

The video encoder is an R3D-18 pretrained on Kinetics-400 (not Jester). Its
512-dimensional features are frozen, so the 1,000- and 5,000-clip experiments
use exactly the specified number of Jester training examples. Validation clips
come only from Qualcomm's separate validation.csv split.
"""

import argparse
import json
import pickle
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import R3D_18_Weights, r3d_18


LABELS = [
    "Swiping Left", "Swiping Right", "Swiping Up", "Swiping Down",
    "Stop Sign", "Thumb Up", "No gesture", "Doing other things",
]
COMMANDS = {
    "Swiping Left": "swipe_next", "Swiping Right": "swipe_previous",
    "Swiping Up": "volume_up", "Swiping Down": "volume_down",
    "Stop Sign": "stop", "Thumb Up": "play",
    "No gesture": "ignore", "Doing other things": "ignore",
}
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


class Clips(Dataset):
    def __init__(self, items, clips_dir):
        self.items = items
        self.clips_dir = clips_dir

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        frames = sorted((self.clips_dir / item["id"]).glob("*.jpg"))
        if len(frames) < 16:
            raise RuntimeError(f"Clip {item['id']} has only {len(frames)} frames")
        indices = np.rint(np.linspace(0, len(frames) - 1, 16)).astype(int)
        sampled = []
        for i in indices:
            with Image.open(frames[i]) as image:
                image = image.convert("RGB").resize((171, 128), Image.Resampling.BILINEAR)
                image = image.crop((29, 8, 141, 120))
                sampled.append(np.asarray(image, dtype=np.uint8))
        array = np.stack(sampled).astype(np.float32) / 255.0
        array = (array - MEAN) / STD
        array = np.ascontiguousarray(array.transpose(3, 0, 1, 2))
        return item["id"], torch.from_numpy(array)


def open_cache(path):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS features (id TEXT PRIMARY KEY, vector BLOB NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS motion (id TEXT PRIMARY KEY, vector BLOB NOT NULL)")
    db.commit()
    return db


def load_features(db, ids):
    found = {}
    for clip_id in ids:
        row = db.execute("SELECT vector FROM features WHERE id=?", (clip_id,)).fetchone()
        if row:
            found[clip_id] = np.frombuffer(row[0], dtype=np.float32).copy()
    return found


def encode_missing(items, root, db, batch_size):
    ids = [item["id"] for item in items]
    found = load_features(db, ids)
    missing = [item for item in items if item["id"] not in found]
    print(f"Cached {len(found)} features; encoding {len(missing)} clips", flush=True)
    if not missing:
        return found
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Encoder device: {device}", flush=True)
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    loader = DataLoader(Clips(missing, root / "clips"), batch_size=batch_size,
                        num_workers=2, shuffle=False)
    done = 0
    with torch.inference_mode():
        for batch_ids, videos in loader:
            outputs = model(videos.to(device)).cpu().numpy().astype(np.float32)
            rows = []
            for clip_id, vector in zip(batch_ids, outputs):
                found[clip_id] = vector
                rows.append((clip_id, vector.tobytes()))
            db.executemany("INSERT OR REPLACE INTO features VALUES (?, ?)", rows)
            db.commit()
            done += len(batch_ids)
            if done % 100 < len(batch_ids) or done == len(missing):
                print(f"Encoded {done}/{len(missing)}", flush=True)
    return found


def motion_for_clip(pair):
    """Summarize signed optical flow, retaining motion order and direction."""
    clip_id, clips_dir = pair
    cv2.setNumThreads(1)
    frames = sorted((Path(clips_dir) / clip_id).glob("*.jpg"))
    indices = np.rint(np.linspace(0, len(frames) - 1, 16)).astype(int)
    images = [cv2.resize(cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE),
                         (100, 100), interpolation=cv2.INTER_AREA) for i in indices]
    if any(image is None for image in images):
        raise RuntimeError(f"Could not read frames for {clip_id}")
    steps = []
    for before, after in zip(images[:-1], images[1:]):
        flow = cv2.calcOpticalFlowFarneback(before, after, None, 0.5, 2, 15, 2, 5, 1.2, 0)
        dx, dy = flow[:, :, 0], flow[:, :, 1]
        mag = cv2.magnitude(dx, dy)
        active = mag > max(0.3, float(np.percentile(mag, 75)))
        if not np.any(active):
            steps.append(np.zeros(9, dtype=np.float32))
            continue
        x, y, m = dx[active], dy[active], mag[active]
        weight = np.clip(m, 0, 8)
        steps.append(np.array([
            np.average(x, weights=weight), np.average(y, weights=weight),
            np.maximum(x, 0).mean(), np.maximum(-x, 0).mean(),
            np.maximum(y, 0).mean(), np.maximum(-y, 0).mean(),
            np.percentile(m, 90), active.mean(), m.mean(),
        ], dtype=np.float32))
    steps = np.stack(steps)
    # Aggregate early, middle, and late motion separately. The final two
    # values preserve net direction over the full gesture.
    vector = np.concatenate([
        steps[:5].mean(axis=0), steps[5:10].mean(axis=0),
        steps[10:].mean(axis=0),
        steps[:, :2].sum(axis=0),
        steps[:, 2:6].max(axis=0),
    ]).astype(np.float32)
    return clip_id, vector


def encode_motion(items, root, db):
    ids = [item["id"] for item in items]
    found = {}
    for clip_id in ids:
        row = db.execute("SELECT vector FROM motion WHERE id=?", (clip_id,)).fetchone()
        if row:
            found[clip_id] = np.frombuffer(row[0], dtype=np.float32).copy()
    missing = [clip_id for clip_id in ids if clip_id not in found]
    print(f"Cached {len(found)} motion vectors; computing {len(missing)}", flush=True)
    if not missing:
        return found
    with ThreadPoolExecutor(max_workers=4) as pool:
        pairs = ((clip_id, str(root / "clips")) for clip_id in missing)
        for index, (clip_id, vector) in enumerate(pool.map(motion_for_clip, pairs), 1):
            found[clip_id] = vector
            db.execute("INSERT OR REPLACE INTO motion VALUES (?, ?)",
                       (clip_id, vector.tobytes()))
            if index % 100 == 0 or index == len(missing):
                db.commit()
                print(f"Motion {index}/{len(missing)}", flush=True)
    return found


def train(manifest, root, batch_size):
    items = json.loads(manifest.read_text())
    counts = Counter((item["split"], item["label"]) for item in items)
    if len({counts["train", label] for label in LABELS}) != 1:
        raise SystemExit("Training manifest is not balanced")
    if len({counts["validation", label] for label in LABELS}) != 1:
        raise SystemExit("Validation manifest is not balanced")
    train_items = [item for item in items if item["split"] == "train"]
    val_items = [item for item in items if item["split"] == "validation"]
    if {item["id"] for item in train_items} & {item["id"] for item in val_items}:
        raise SystemExit("Training/validation ID overlap")
    db = open_cache(root / "r3d18_features.sqlite")
    features = encode_missing(items, root, db, batch_size)
    motion = encode_motion(items, root, db)
    def x_for(entries):
        return np.stack([np.concatenate((features[item["id"]], motion[item["id"]]))
                         for item in entries])
    x_train = x_for(train_items)
    y_train = [item["label"] for item in train_items]
    x_val = x_for(val_items)
    y_val = [item["label"] for item in val_items]

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_train[:, 512:] *= 4.0
    x_val[:, 512:] *= 4.0
    classifier = SVC(C=10, kernel="rbf", gamma="scale")
    classifier.fit(x_train, y_train)
    predicted = classifier.predict(x_val)
    accuracy = accuracy_score(y_val, predicted)
    action_indices = [i for i, label in enumerate(y_val) if COMMANDS[label] != "ignore"]
    negative_indices = [i for i, label in enumerate(y_val) if COMMANDS[label] == "ignore"]
    action_accuracy = float(np.mean([y_val[i] == predicted[i] for i in action_indices]))
    negative_rejection = float(np.mean([COMMANDS[predicted[i]] == "ignore"
                                        for i in negative_indices]))
    matrix = confusion_matrix(y_val, predicted, labels=LABELS)
    report = {
        "training_examples": len(train_items), "validation_examples": len(val_items),
        "train_per_class": counts["train", LABELS[0]],
        "validation_per_class": counts["validation", LABELS[0]],
        "encoder": "torchvision R3D-18 Kinetics-400 V1, frozen",
        "classifier": "R3D-18 features + optical flow, StandardScaler, motion weight 4, RBF SVC(C=10, gamma=scale)",
        "validation_accuracy": accuracy,
        "six_command_accuracy": action_accuracy,
        "negative_rejection_rate": negative_rejection,
        "labels": LABELS,
        "confusion_matrix": matrix.tolist(),
        "errors": [
            {"id": val_items[i]["id"], "actual": y_val[i], "predicted": predicted[i]}
            for i in range(len(val_items)) if y_val[i] != predicted[i]
        ],
    }
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)
    stem = f"jester_{len(train_items)}"
    (result_dir / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    with (result_dir / f"{stem}.pkl").open("wb") as handle:
        pickle.dump({"scaler": scaler, "classifier": classifier,
                     "labels": LABELS, "commands": COMMANDS}, handle)
    print(json.dumps({key: report[key] for key in (
        "training_examples", "validation_examples", "validation_accuracy",
        "six_command_accuracy", "negative_rejection_rate")}, indent=2))
    print(f"Report: {result_dir / (stem + '.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path("Data/jester"))
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    train(args.manifest, args.root, args.batch_size)

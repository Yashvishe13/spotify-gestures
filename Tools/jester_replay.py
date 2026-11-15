#!/usr/bin/env python3
"""Replay the user's short gesture clips through a Jester video checkpoint."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import r3d_18


MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
ACTION = {
    "Swiping Left": "swipe_next", "Swiping Right": "swipe_previous",
    "Swiping Up": "volume_up", "Swiping Down": "volume_down",
    "Stop Sign": "stop", "Thumb Up": "play",
    "No gesture": "ignore", "Doing other things": "ignore",
}


def load_video(path: Path, frame_count: int, mirror: bool):
    capture = cv2.VideoCapture(str(path))
    images = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(cv2.resize(image, (128, 96), interpolation=cv2.INTER_AREA))
    capture.release()
    if len(images) < frame_count:
        raise ValueError(f"Too few decoded frames in {path}: {len(images)}")
    indices = np.rint(np.linspace(0, len(images) - 1, frame_count)).astype(int)
    video = np.stack([images[i] for i in indices]).astype(np.float32) / 255.0
    if mirror:
        video = video[:, :, ::-1, :]
    video = (video - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(video.transpose(3, 0, 1, 2)))


def main(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.coreml_model:
        import coremltools as ct
        model = ct.models.MLModel(str(args.coreml_model))
        device = None
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = r3d_18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(saved["labels"]))
        model.load_state_dict(saved["model"])
        model.eval().to(device)
    records = []
    with torch.inference_mode():
        for path in sorted(args.clips.glob("*/*.mp4")):
            actual = path.parent.name
            video = load_video(path, saved["frames"], args.mirror).unsqueeze(0)
            if args.coreml_model:
                logits = model.predict({"video": video.numpy()})["logits"].reshape(-1)
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                index = int(probabilities.argmax())
                confidence = float(probabilities[index])
            else:
                probabilities = torch.softmax(model(video.to(device)).squeeze(0), dim=0)
                index = int(probabilities.argmax().item())
                confidence = float(probabilities[index].item())
            label = saved["labels"][index]
            prediction = ACTION[label]
            records.append({"file": str(path), "actual": actual, "predicted": prediction,
                            "dataset_label": label, "confidence": confidence})
    correct = sum(row["actual"] == row["predicted"] for row in records)
    result = {"count": len(records), "correct": correct,
              "accuracy": correct / len(records), "mirror": args.mirror, "records": records}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--clips", type=Path, default=Path("transformed_vids"))
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--coreml-model", type=Path)
    parser.add_argument("--output", type=Path)
    main(parser.parse_args())

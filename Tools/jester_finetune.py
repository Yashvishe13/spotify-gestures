#!/usr/bin/env python3
"""Fine-tune a temporal video network on the selected Jester clips.

Unlike jester_train.py, this updates the video network itself. It keeps the
entire webcam frame, uses consistent clip-level augmentation, and saves a
checkpoint after every epoch. The official Jester validation split is never
used for gradient updates.
"""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import R3D_18_Weights, r3d_18


LABELS = [
    "Swiping Left", "Swiping Right", "Swiping Up", "Swiping Down",
    "Stop Sign", "Thumb Up", "No gesture", "Doing other things",
]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}
FLIPPED = {0: 1, 1: 0}
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


class VideoClips(Dataset):
    def __init__(self, items, root: Path, training: bool, frames: int):
        self.items = items
        self.training = training
        self.frames = frames
        self.paths = []
        for item in items:
            paths = sorted((root / "clips" / item["id"]).glob("*.jpg"))
            if len(paths) < frames:
                raise ValueError(f"Clip {item['id']} has only {len(paths)} frames")
            self.paths.append(paths)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        paths = self.paths[index]
        positions = np.linspace(0, len(paths) - 1, self.frames)
        if self.training:
            positions += np.random.uniform(-0.65, 0.65, size=self.frames)
        positions = np.clip(np.rint(positions), 0, len(paths) - 1).astype(int)
        frames = []
        for i in positions:
            with Image.open(paths[i]) as image:
                # Use the whole frame; a center crop can remove swipe endpoints.
                image = image.convert("RGB").resize((128, 96), Image.Resampling.BILINEAR)
                frames.append(np.asarray(image, dtype=np.uint8))
        video = np.stack(frames).astype(np.float32) / 255.0
        label = LABEL_TO_INDEX[item["label"]]
        if self.training:
            gain = np.random.uniform(0.85, 1.15)
            offset = np.random.uniform(-0.06, 0.06)
            video = np.clip(video * gain + offset, 0, 1)
            if random.random() < 0.5:
                video = video[:, :, ::-1, :]
                label = FLIPPED.get(label, label)
        video = (video - MEAN) / STD
        video = np.ascontiguousarray(video.transpose(3, 0, 1, 2))
        return torch.from_numpy(video), label


def make_model(device):
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.layer4.requires_grad_(True)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    return model.to(device)


def freeze_batch_norm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.eval()


def evaluate(model, loader, device):
    model.eval()
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    loss_sum = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for videos, labels in loader:
            videos = videos.to(device)
            labels = labels.to(device)
            logits = model(videos)
            loss_sum += criterion(logits, labels).item() * len(labels)
            actual = labels.cpu().numpy()
            predicted = logits.argmax(dim=1).cpu().numpy()
            np.add.at(matrix, (actual, predicted), 1)
            total += len(labels)
    return loss_sum / total, float(np.trace(matrix) / total), matrix


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    items = json.loads(args.manifest.read_text())
    train_items = [item for item in items if item["split"] == "train"]
    val_items = [item for item in items if item["split"] == "validation"]
    if args.limit_train:
        train_items = train_items[:args.limit_train]
    if args.limit_val:
        val_items = val_items[:args.limit_val]
    if {item["id"] for item in train_items} & {item["id"] for item in val_items}:
        raise ValueError("Training and validation clip IDs overlap")
    print(f"Device: {device}; training: {len(train_items)}; validation: {len(val_items)}", flush=True)
    train_set = VideoClips(train_items, args.root, True, args.frames)
    val_set = VideoClips(val_items, args.root, False, args.frames)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, persistent_workers=args.workers > 0)
    model = make_model(device)
    optimizer = torch.optim.AdamW([
        {"params": model.layer4.parameters(), "lr": args.lr},
        {"params": model.fc.parameters(), "lr": args.lr * 5},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    output_dir = args.root / "finetune" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.json"
    history = []
    start_epoch = 0
    best_accuracy = -1.0
    if args.resume and last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = saved["epoch"] + 1
        best_accuracy = saved["best_accuracy"]
        history = saved["history"]
        print(f"Resuming after epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        started = time.monotonic()
        model.train()
        freeze_batch_norm(model)
        loss_sum = 0.0
        correct = 0
        seen = 0
        for step, (videos, labels) in enumerate(train_loader, 1):
            videos = videos.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(videos)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.layer4.parameters()) + list(model.fc.parameters()), 2.0)
            optimizer.step()
            count = len(labels)
            loss_sum += loss.item() * count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += count
            if step % args.log_steps == 0:
                print(f"Epoch {epoch + 1}/{args.epochs} step {step}/{len(train_loader)} "
                      f"train={correct / seen:.3f} loss={loss_sum / seen:.3f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        val_loss, val_accuracy, matrix = evaluate(model, val_loader, device)
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "training_examples": seen,
            "train_accuracy": correct / seen,
            "train_loss": loss_sum / seen,
            "validation_accuracy": val_accuracy,
            "validation_loss": val_loss,
            "seconds": round(time.monotonic() - started, 1),
            "confusion_matrix": matrix.tolist(),
        }
        history.append(record)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "best_accuracy": max(best_accuracy, val_accuracy),
                    "history": history}, last_path)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({"model": model.state_dict(), "labels": LABELS,
                        "frames": args.frames, "height": 96, "width": 128}, best_path)
        history_path.write_text(json.dumps({"labels": LABELS, "history": history}, indent=2) + "\n")
        print(f"Epoch {epoch + 1}: train={record['train_accuracy']:.3%} "
              f"validation={val_accuracy:.3%} best={best_accuracy:.3%} "
              f"time={record['seconds']}s", flush=True)
        if args.max_steps:
            break
        if args.early_stop and len(history) - 1 - max(
                range(len(history)), key=lambda i: history[i]["validation_accuracy"]) >= args.early_stop:
            print("Early stopping after no validation improvement", flush=True)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path("Data/jester"))
    parser.add_argument("--name", default="r3d18_full_frame")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-steps", type=int, default=100)
    parser.add_argument("--early-stop", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--max-steps", type=int)
    train(parser.parse_args())

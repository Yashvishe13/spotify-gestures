#!/usr/bin/env python3
"""Export the local research checkpoint to Core ML for the macOS app."""

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn
from torchvision.models.video import r3d_18


def main(checkpoint: Path, output: Path, float32: bool):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(saved["labels"]))
    model.load_state_dict(saved["model"])
    model.eval()
    shape = (1, 3, saved["frames"], saved["height"], saved["width"])
    sample = torch.randn(shape)
    with torch.inference_mode():
        features = model.layer4(model.layer3(model.layer2(model.layer1(model.stem(sample)))))
        # The deployed input has a fixed shape. A regular pool over the final
        # feature map is exactly the same calculation as adaptive pooling to 1.
        model.avgpool = nn.AvgPool3d(kernel_size=tuple(features.shape[-3:]))
        traced = torch.jit.trace(model, sample)
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name="video", shape=shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT32 if float32 else ct.precision.FLOAT16,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(output))
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--float32", action="store_true")
    args = parser.parse_args()
    main(args.checkpoint, args.output, args.float32)

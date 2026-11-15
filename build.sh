#!/bin/zsh
set -euo pipefail
cd "${0:A:h}"

if [[ "${1:-}" == "--train" || ! -f Resources/GestureModel.json ]]; then
  echo "Training gesture model from augmented_vids…"
  swiftc -O -target arm64-apple-macosx13.0 -framework Vision -framework AVFoundation \
    Sources/GestureCore.swift Sources/TrainModel.swift -o /tmp/spotify-gesture-trainer
  /tmp/spotify-gesture-trainer augmented_vids Resources/GestureModel.json
fi

app="dist/Spotify Gestures.app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
swiftc -O -target arm64-apple-macosx13.0 \
  -framework AppKit -framework AVFoundation -framework Vision -framework Carbon -framework ServiceManagement \
  -framework CoreML -framework CoreImage \
  Sources/GestureCore.swift Sources/VideoGestureModel.swift Sources/SpotifyGestureApp.swift \
  -o "$app/Contents/MacOS/SpotifyGestures"
cp Resources/Info.plist "$app/Contents/Info.plist"
cp Resources/GestureModel.json "$app/Contents/Resources/GestureModel.json"
video_model="Data/jester/finetune/r3d18_full_frame/JesterVideoFP32.mlpackage"
rm -rf "$app/Contents/Resources/JesterVideoFP32.mlmodelc"
if [[ -d "$video_model" ]]; then
  xcrun coremlc compile "$video_model" "$app/Contents/Resources"
fi
codesign --force --sign - "$app"
echo "Built $app"

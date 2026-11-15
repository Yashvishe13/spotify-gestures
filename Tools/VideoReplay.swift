import AVFoundation
import Foundation

func videoFrames(at url: URL, using model: VideoGestureModel) async throws -> (video: [VideoFrame], hand: [HandFrame]) {
    let asset = AVURLAsset(url: url)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else { return ([], []) }
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
    ])
    output.alwaysCopiesSampleData = false
    reader.add(output)
    reader.startReading()
    var frames: [VideoFrame] = []
    var handFrames: [HandFrame] = []
    var nextTime = 0.0
    while let buffer = output.copyNextSampleBuffer() {
        let time = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(buffer))
        guard time >= nextTime else { continue }
        nextTime = time + 0.10
        if let image = CMSampleBufferGetImageBuffer(buffer), let frame = model.capture(image) {
            frames.append(frame)
            if let hand = GestureCore.detect(image, time: time) { handFrames.append(hand) }
        }
    }
    return (frames, handFrames)
}

@main
struct VideoReplay {
    static func main() async throws {
        guard CommandLine.arguments.count == 4 else {
            fputs("Usage: video-replay compiled-model.mlmodelc GestureModel.json transformed_vids\n", stderr)
            exit(2)
        }
        let modelURL = URL(fileURLWithPath: CommandLine.arguments[1])
        guard let model = VideoGestureModel(modelURL: modelURL) else {
            fputs("Could not load compiled Core ML model\n", stderr)
            exit(1)
        }
        let handModel = try JSONDecoder().decode(GestureModel.self,
            from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2])))
        let root = URL(fileURLWithPath: CommandLine.arguments[3])
        let labels = ["swipe_next", "swipe_previous", "volume_up", "volume_down", "stop"]
        var correct = 0
        var total = 0
        for label in labels {
            let files = try FileManager.default.contentsOfDirectory(
                at: root.appendingPathComponent(label), includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "mp4" }.sorted { $0.path < $1.path }
            for file in files {
                let frames = try await videoFrames(at: file, using: model)
                let prediction = model.predict(frames.video, mirrored: true)
                let hand = GestureCore.classifyEpisode(frames.hand, model: handModel)
                let action = prediction?.action ?? "none"
                let confidence = prediction?.confidence ?? 0
                let fused = GestureFusion.resolve(video: prediction, hand: hand) ?? "none"
                print("\(file.lastPathComponent): video=\(action) hand=\(hand.action ?? "none") fused=\(fused) " +
                      "(\(frames.video.count) frames, \(String(format: "%.2f", confidence)))")
                if fused == label { correct += 1 }
                total += 1
            }
        }
        print("\(correct)/\(total) clips correct")
    }
}

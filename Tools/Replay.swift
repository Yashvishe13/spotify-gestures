import Foundation
import AVFoundation

func replay(_ url: URL, model: GestureModel) async throws -> (detections: Int, predictions: [String]) {
    let asset = AVURLAsset(url: url)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else { return (0, []) }
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
    ])
    reader.add(output)
    reader.startReading()
    var episode: [HandFrame] = []
    var predictions: [String] = []
    var nextTime = 0.0
    var lastHandTime = -1.0
    var detections = 0
    func finish() {
        let decision = GestureCore.classifyEpisode(episode, model: model)
        predictions.append(decision.action ?? "none")
        episode.removeAll()
    }
    while let buffer = output.copyNextSampleBuffer() {
        let time = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(buffer))
        guard time >= nextTime else { continue }
        nextTime = time + 0.10
        guard let image = CMSampleBufferGetImageBuffer(buffer) else { continue }
        if let hand = GestureCore.detect(image, time: time) {
            episode.append(hand)
            lastHandTime = time
            detections += 1
        } else if time - lastHandTime > 0.5 && !episode.isEmpty {
            finish()
        }
    }
    if !episode.isEmpty { finish() }
    return (detections, predictions)
}

@main
struct Replay {
    static func main() async throws {
        guard CommandLine.arguments.count == 3 else {
            fputs("Usage: replay model.json video_directory\n", stderr)
            exit(2)
        }
        let model = try JSONDecoder().decode(GestureModel.self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
        let root = URL(fileURLWithPath: CommandLine.arguments[2])
        let fm = FileManager.default
        var correct = 0
        var total = 0
        for label in GestureCore.labels {
            let files = try fm.contentsOfDirectory(at: root.appendingPathComponent(label), includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "mp4" }.sorted { $0.path < $1.path }
            for file in files {
                let result = try await replay(file, model: model)
                let first = result.predictions.first(where: { $0 != "none" }) ?? "none"
                print("\(file.lastPathComponent): \(first) (\(result.detections) hand frames)")
                if first == label { correct += 1 }
                total += 1
            }
        }
        print("\(correct)/\(total) clips triggered correctly")
    }
}

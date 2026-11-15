import Foundation
import AVFoundation

func frames(from url: URL) async throws -> [HandFrame] {
    let asset = AVURLAsset(url: url)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else { return [] }
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
    ])
    output.alwaysCopiesSampleData = false
    reader.add(output)
    reader.startReading()
    var result: [HandFrame] = []
    var nextTime = 0.0
    while let buffer = output.copyNextSampleBuffer() {
        let time = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(buffer))
        guard time >= nextTime else { continue }
        nextTime = time + 0.10
        if let image = CMSampleBufferGetImageBuffer(buffer),
           let hand = GestureCore.detect(image, time: time) { result.append(hand) }
    }
    return result
}

@main
struct TrainModel {
    static func main() async throws {
        guard CommandLine.arguments.count == 3 else {
            fputs("Usage: train-model augmented_vids output.json\n", stderr)
            exit(2)
        }
        let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        let output = URL(fileURLWithPath: CommandLine.arguments[2])
        let fm = FileManager.default
        var examples: [GestureExample] = []
        var episodes: [GestureExample] = []
        var processed = 0
        for label in GestureCore.labels {
            let directory = root.appendingPathComponent(label)
            let files = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension.lowercased() == "mp4" }.sorted { $0.path < $1.path }
            for file in files {
                let handFrames = try await frames(from: file)
                if let vector = GestureCore.episodeFeatures(handFrames) {
                    episodes.append(GestureExample(label: label, source: file.lastPathComponent, features: vector))
                }
                for window in GestureCore.trainingWindows(handFrames, label: label) {
                    if let vector = GestureCore.features(window) {
                        examples.append(GestureExample(label: label, source: file.lastPathComponent, features: vector))
                    }
                }
                processed += 1
                if processed % 20 == 0 { fputs("Processed \(processed) clips; \(examples.count) examples\n", stderr) }
            }
        }
        let model = GestureModel(version: 2, sampleCount: processed, examples: examples, episodeExamples: episodes)
        let data = try JSONEncoder().encode(model)
        try data.write(to: output, options: .atomic)
        print("Saved \(examples.count) windows and \(episodes.count) episodes from \(processed) videos to \(output.path)")
    }
}

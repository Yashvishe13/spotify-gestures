import CoreImage
import CoreML
import CoreVideo
import Foundation

struct VideoFrame {
    let bgra: Data
    let bytesPerRow: Int
}

struct VideoPrediction {
    let action: String
    let confidence: Double
}

final class VideoGestureModel {
    static let width = 128
    static let height = 96
    static let frameCount = 12
    private static let actions = [
        "swipe_next", "swipe_previous", "volume_up", "volume_down",
        "stop", "play", "ignore", "ignore"
    ]
    private static let mean: [Float] = [0.43216, 0.394666, 0.37645]
    private static let deviation: [Float] = [0.22803, 0.22145, 0.216989]

    private let model: MLModel
    private let context = CIContext(options: [.cacheIntermediates: false])

    init?(modelURL: URL? = nil) {
        guard let url = modelURL ?? Bundle.main.url(forResource: "JesterVideoFP32", withExtension: "mlmodelc") else {
            return nil
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        do {
            model = try MLModel(contentsOf: url, configuration: configuration)
        } catch {
            NSLog("Could not load video gesture model: %@", error.localizedDescription)
            return nil
        }
    }

    func capture(_ source: CVPixelBuffer) -> VideoFrame? {
        var destination: CVPixelBuffer?
        let attributes: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]]
        let result = CVPixelBufferCreate(kCFAllocatorDefault, Self.width, Self.height,
                                         kCVPixelFormatType_32BGRA, attributes as CFDictionary,
                                         &destination)
        guard result == kCVReturnSuccess, let destination else { return nil }
        let scale = CGAffineTransform(scaleX: CGFloat(Self.width) / CGFloat(CVPixelBufferGetWidth(source)),
                                      y: CGFloat(Self.height) / CGFloat(CVPixelBufferGetHeight(source)))
        let image = CIImage(cvPixelBuffer: source).transformed(by: scale)
        context.render(image, to: destination)
        CVPixelBufferLockBaseAddress(destination, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(destination, .readOnly) }
        guard let address = CVPixelBufferGetBaseAddress(destination) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(destination)
        return VideoFrame(bgra: Data(bytes: address, count: bytesPerRow * Self.height),
                          bytesPerRow: bytesPerRow)
    }

    func predict(_ frames: [VideoFrame], mirrored: Bool) -> VideoPrediction? {
        guard frames.count >= 4,
              let input = try? MLMultiArray(shape: [1, 3, Self.frameCount, Self.height, Self.width]
                                                  .map { NSNumber(value: $0) }, dataType: .float32) else {
            return nil
        }
        let values = input.dataPointer.bindMemory(to: Float.self,
                                                  capacity: 3 * Self.frameCount * Self.height * Self.width)
        for time in 0..<Self.frameCount {
            let sourceIndex = Int((Double(time) * Double(frames.count - 1) /
                                   Double(Self.frameCount - 1)).rounded())
            let frame = frames[sourceIndex]
            frame.bgra.withUnsafeBytes { raw in
                let bytes = raw.bindMemory(to: UInt8.self)
                for y in 0..<Self.height {
                    for x in 0..<Self.width {
                        let sourceX = mirrored ? Self.width - 1 - x : x
                        let offset = y * frame.bytesPerRow + sourceX * 4
                        let channels: [UInt8] = [bytes[offset + 2], bytes[offset + 1], bytes[offset]]
                        for channel in 0..<3 {
                            let index = (((channel * Self.frameCount + time) * Self.height + y) * Self.width + x)
                            values[index] = (Float(channels[channel]) / 255 - Self.mean[channel]) /
                                Self.deviation[channel]
                        }
                    }
                }
            }
        }
        do {
            let provider = try MLDictionaryFeatureProvider(dictionary: ["video": MLFeatureValue(multiArray: input)])
            let output = try model.prediction(from: provider)
            guard let logits = output.featureValue(for: "logits")?.multiArrayValue,
                  logits.count == Self.actions.count else { return nil }
            let scores = (0..<Self.actions.count).map { index in
                logits[[NSNumber(value: 0), NSNumber(value: index)]].doubleValue
            }
            guard let winner = scores.indices.max(by: { scores[$0] < scores[$1] }) else { return nil }
            let peak = scores[winner]
            let exponentials = scores.map { exp($0 - peak) }
            let confidence = exponentials[winner] / exponentials.reduce(0, +)
            return VideoPrediction(action: Self.actions[winner], confidence: confidence)
        } catch {
            NSLog("Video gesture prediction failed: %@", error.localizedDescription)
            return nil
        }
    }
}

enum GestureFusion {
    private static let motionActions: Set<String> = [
        "swipe_next", "swipe_previous", "volume_up", "volume_down"
    ]

    static func resolve(video: VideoPrediction?, hand: EpisodeDecision) -> String? {
        // The hand model was trained on this camera and is reliable when it
        // identifies a direction. The video model fills gaps in hand tracking.
        if let action = hand.action, motionActions.contains(action) { return action }
        if let video, motionActions.contains(video.action), video.confidence >= 0.35 {
            return video.action
        }
        // The user's open-palm Stop resembles Thumb Up in Jester at this hand
        // scale. Preserve the hand-pose Stop decision before trying Play.
        if hand.action == "stop" { return "stop" }
        if let video, video.action == "stop", video.confidence >= 0.4 { return "stop" }
        if let video, video.action == "play", video.confidence >= 0.6 { return "play" }
        return nil
    }
}

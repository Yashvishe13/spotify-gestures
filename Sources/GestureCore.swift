import Foundation
import Vision
import CoreVideo

struct HandFrame: Codable {
    let time: Double
    let x: Double
    let y: Double
    let openness: Double
    let confidence: Double
}

struct GestureExample: Codable {
    let label: String
    let source: String
    let features: [Double]
}

struct GestureModel: Codable {
    let version: Int
    let sampleCount: Int
    let examples: [GestureExample]
    let episodeExamples: [GestureExample]
}

struct EpisodeDecision {
    let action: String?
    let message: String
}

enum GestureCore {
    static let labels = ["swipe_next", "swipe_previous", "stop", "volume_up", "volume_down"]
    static let windowSeconds = 0.85

    static func detect(_ pixelBuffer: CVPixelBuffer, time: Double) -> HandFrame? {
        let request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = 1
        do {
            try VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
            guard let observation = request.results?.first,
                  let points = try? observation.recognizedPoints(.all),
                  let wrist = points[.wrist], wrist.confidence > 0.25,
                  let index = points[.indexMCP], index.confidence > 0.25,
                  let little = points[.littleMCP], little.confidence > 0.25 else { return nil }
            let centerX = (Double(wrist.location.x) + Double(index.location.x) + Double(little.location.x)) / 3
            let centerY = (Double(wrist.location.y) + Double(index.location.y) + Double(little.location.y)) / 3
            let palmWidth = max(0.025, hypot(Double(index.location.x - little.location.x), Double(index.location.y - little.location.y)))
            let tips: [VNHumanHandPoseObservation.JointName] = [.thumbTip, .indexTip, .middleTip, .ringTip, .littleTip]
            let reaches = tips.compactMap { name -> Double? in
                guard let p = points[name], p.confidence > 0.25 else { return nil }
                return hypot(Double(p.location.x - wrist.location.x), Double(p.location.y - wrist.location.y)) / palmWidth
            }
            let openness = reaches.isEmpty ? 0 : reaches.reduce(0, +) / Double(reaches.count)
            return HandFrame(time: time, x: centerX, y: centerY, openness: min(openness, 8), confidence: Double(observation.confidence))
        } catch { return nil }
    }

    static func features(_ frames: [HandFrame]) -> [Double]? {
        guard frames.count >= 4, let first = frames.first, let last = frames.last,
              last.time - first.time >= 0.28 else { return nil }
        let x = frames.map(\.x), y = frames.map(\.y)
        let dx = last.x - first.x, dy = last.y - first.y
        let rangeX = (x.max() ?? 0) - (x.min() ?? 0)
        let rangeY = (y.max() ?? 0) - (y.min() ?? 0)
        var pathX = 0.0, pathY = 0.0
        for i in 1..<frames.count {
            pathX += abs(frames[i].x - frames[i - 1].x)
            pathY += abs(frames[i].y - frames[i - 1].y)
        }
        let openness = frames.map(\.openness).reduce(0, +) / Double(frames.count)
        let mid = frames[frames.count / 2]
        return [dx, dy, rangeX, rangeY, pathX, pathY, openness / 4,
                first.x, first.y, last.x, last.y,
                mid.x - first.x, mid.y - first.y]
    }

    static func trainingWindows(_ frames: [HandFrame], label: String) -> [[HandFrame]] {
        guard frames.count >= 4 else { return [] }
        var candidates: [([HandFrame], Double)] = []
        for anchor in stride(from: 0, to: frames.count, by: 2) {
            let start = frames[anchor].time
            let window = frames.filter { $0.time >= start && $0.time <= start + windowSeconds }
            guard let f = features(window) else { continue }
            let score: Double
            switch label {
            case "swipe_next": score = -f[0] - 0.5 * abs(f[1])
            case "swipe_previous": score = f[0] - 0.5 * abs(f[1])
            case "volume_up": score = f[1] - 0.5 * abs(f[0])
            case "volume_down": score = -f[1] - 0.5 * abs(f[0])
            case "stop": score = f[6] - 2 * hypot(f[0], f[1]) - f[2] - f[3]
            default: score = -1
            }
            if label == "stop" || score > 0.06 { candidates.append((window, score)) }
        }
        return candidates.sorted { $0.1 > $1.1 }.prefix(2).map(\.0)
    }

    static func episodeFeatures(_ frames: [HandFrame]) -> [Double]? {
        guard frames.count >= 4, let first = frames.first, let last = frames.last,
              last.time - first.time >= 0.3 else { return nil }
        let xs = frames.map(\.x), ys = frames.map(\.y)
        let elapsed = last.time - first.time
        func index(_ values: [Double], max maximum: Bool) -> Double {
            let target = maximum ? (values.max() ?? 0) : (values.min() ?? 0)
            let i = values.firstIndex(of: target) ?? 0
            return (frames[i].time - first.time) / elapsed
        }
        var result = [last.x - first.x, last.y - first.y,
                      (xs.max() ?? 0) - (xs.min() ?? 0),
                      (ys.max() ?? 0) - (ys.min() ?? 0),
                      index(xs, max: true), index(xs, max: false),
                      index(ys, max: true), index(ys, max: false),
                      frames.map(\.openness).reduce(0, +) / Double(frames.count) / 4,
                      (frames.map(\.openness).max() ?? 0) / 4]
        for step in 0..<8 {
            let target = first.time + elapsed * Double(step) / 7
            let after = frames.firstIndex { $0.time >= target } ?? (frames.count - 1)
            let before = max(0, after - 1)
            let a = frames[before], b = frames[after]
            let fraction = b.time == a.time ? 0 : (target - a.time) / (b.time - a.time)
            result.append(a.x + fraction * (b.x - a.x) - first.x)
            result.append(a.y + fraction * (b.y - a.y) - first.y)
        }
        result += [first.x, first.y, min(elapsed, 3) / 3]
        return result
    }

    static func predict(_ features: [Double], model: GestureModel) -> (label: String, score: Double)? {
        guard model.version == 2, !model.examples.isEmpty else { return nil }
        let weights: [Double] = [3, 3, 1.5, 1.5, 0.8, 0.8, 0.08, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8]
        return vote(features, examples: model.examples, weights: weights)
    }

    static func predictEpisode(_ features: [Double], model: GestureModel) -> (label: String, score: Double)? {
        guard model.version == 2, !model.episodeExamples.isEmpty else { return nil }
        let weights: [Double] = [2, 2, 1, 1, 0.12, 0.12, 0.12, 0.12, 0.07, 0.07] +
            Array(repeating: 1.0, count: 16) + [0.1, 0.1, 0.05]
        return vote(features, examples: model.episodeExamples, weights: weights)
    }

    static func classifyEpisode(_ frames: [HandFrame], model: GestureModel) -> EpisodeDecision {
        guard let vector = episodeFeatures(frames),
              let result = predictEpisode(vector, model: model), result.score >= 0.55 else {
            return EpisodeDecision(action: nil, message: "No gesture matched · try a larger motion")
        }
        let action = result.label
        if action == "stop" && vector[2] > 0.14 && vector[2] > vector[3] * 1.1 {
            return EpisodeDecision(action: nil, message: "Sideways motion was unclear · try the swipe again")
        }
        let enoughMotion: Bool
        switch action {
        case "swipe_next", "swipe_previous": enoughMotion = vector[2] > 0.14
        case "volume_up", "volume_down": enoughMotion = vector[3] > 0.14
        case "stop": enoughMotion = vector[8] > 0.35
        default: enoughMotion = false
        }
        guard enoughMotion else {
            return EpisodeDecision(action: nil, message: "Gesture was too small · try again")
        }
        return EpisodeDecision(action: action,
                               message: "Recognized: \(action.replacingOccurrences(of: "_", with: " "))")
    }

    private static func vote(_ features: [Double], examples: [GestureExample], weights: [Double]) -> (label: String, score: Double)? {
        guard features.count == weights.count else { return nil }
        let neighbors = examples.compactMap { example -> (String, Double)? in
            guard example.features.count == features.count else { return nil }
            let distance = zip(zip(features, example.features), weights).reduce(0.0) { sum, pair in
                let delta = pair.0.0 - pair.0.1
                return sum + pair.1 * delta * delta
            }
            return (example.label, sqrt(distance))
        }.sorted { $0.1 < $1.1 }
        let nearest = Array(neighbors.prefix(9))
        guard !nearest.isEmpty else { return nil }
        var votes: [String: Double] = [:]
        for (label, distance) in nearest { votes[label, default: 0] += 1 / max(0.025, distance) }
        guard let winner = votes.max(by: { $0.value < $1.value }) else { return nil }
        let score = winner.value / votes.values.reduce(0, +)
        return (winner.key, score)
    }
}

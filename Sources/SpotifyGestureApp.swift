import AppKit
import AVFoundation
import Carbon
import ServiceManagement
import Vision

final class CameraView: NSView {
    let previewLayer = AVCaptureVideoPreviewLayer()
    let status = NSTextField(labelWithString: "Waiting for camera…")
    let detail = NSTextField(labelWithString: "Show a gesture, then move your hand out of view to send it")

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.backgroundColor = NSColor.black.cgColor
        previewLayer.videoGravity = .resizeAspectFill
        layer?.addSublayer(previewLayer)
        let panel = NSVisualEffectView()
        panel.material = .hudWindow
        panel.blendingMode = .withinWindow
        panel.state = .active
        panel.translatesAutoresizingMaskIntoConstraints = false
        addSubview(panel)
        status.font = .systemFont(ofSize: 18, weight: .semibold)
        status.textColor = .white
        detail.font = .systemFont(ofSize: 12)
        detail.textColor = .white
        for label in [status, detail] {
            label.translatesAutoresizingMaskIntoConstraints = false
            panel.addSubview(label)
        }
        NSLayoutConstraint.activate([
            panel.leadingAnchor.constraint(equalTo: leadingAnchor),
            panel.trailingAnchor.constraint(equalTo: trailingAnchor),
            panel.bottomAnchor.constraint(equalTo: bottomAnchor),
            panel.heightAnchor.constraint(equalToConstant: 78),
            status.leadingAnchor.constraint(equalTo: panel.leadingAnchor, constant: 20),
            status.topAnchor.constraint(equalTo: panel.topAnchor, constant: 13),
            detail.leadingAnchor.constraint(equalTo: status.leadingAnchor),
            detail.topAnchor.constraint(equalTo: status.bottomAnchor, constant: 6)
        ])
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
    override func layout() {
        super.layout()
        previewLayer.frame = bounds
    }
}

final class SpotifyController {
    private let queue = DispatchQueue(label: "spotify.command")

    func perform(_ action: String, completion: @escaping (String) -> Void) {
        let command: String
        switch action {
        case "swipe_next": command = "next track"
        case "swipe_previous": command = "try\nset player position to 0\ndelay 0.15\nend try\nprevious track"
        case "play": command = "play"
        case "stop": command = "pause"
        case "volume_up": command = "set v to sound volume\nif v > 90 then\nset sound volume to 100\nelse\nset sound volume to (v + 10)\nend if"
        case "volume_down": command = "set v to sound volume\nif v < 10 then\nset sound volume to 0\nelse\nset sound volume to (v - 10)\nend if"
        default: return
        }
        queue.async {
            let script = NSAppleScript(source: "tell application \"Spotify\"\n\(command)\nend tell")
            var error: NSDictionary?
            _ = script?.executeAndReturnError(&error)
            let message = error == nil ? "Spotify: \(action.replacingOccurrences(of: "_", with: " "))" :
                "Spotify error: \(error?[NSAppleScript.errorMessage] as? String ?? "unknown error")"
            DispatchQueue.main.async { completion(message) }
        }
    }
}

final class CameraController: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "gesture.camera")
    private let inferenceQueue = DispatchQueue(label: "gesture.video.inference")
    private let spotify = SpotifyController()
    private let model: GestureModel?
    private let videoModel = VideoGestureModel()
    private var configured = false
    private var wantsCamera = false
    private var classifying = false
    private var generation = 0
    private var episodeFrames: [HandFrame] = []
    private var recentVideoFrames: [VideoFrame] = []
    private var episodeVideoFrames: [VideoFrame] = []
    private var lastProcessTime = -1.0
    private var lastHandTime = -1.0
    private var lastActionTime = -10.0
    private var lastDiagnosticTime = -1.0
    private var mirrorGestures = UserDefaults.standard.bool(forKey: "MirrorGestures")
    var report: ((String) -> Void)?

    override init() {
        if let url = Bundle.main.url(forResource: "GestureModel", withExtension: "json"),
           let data = try? Data(contentsOf: url) {
            model = try? JSONDecoder().decode(GestureModel.self, from: data)
        } else { model = nil }
        super.init()
    }

    private func show(_ message: String) {
        DispatchQueue.main.async { [weak self] in self?.report?(message) }
    }

    private func diagnostic(_ message: String, at time: Double) {
        guard time - lastDiagnosticTime > 0.6, time - lastActionTime > 2 else { return }
        lastDiagnosticTime = time
        show(message)
    }

    func start() {
        queue.async {
            NSLog("Spotify Gestures camera start; video model: %@", self.videoModel == nil ? "missing" : "loaded")
            self.wantsCamera = true
            if self.model == nil { self.show("Gesture model missing. Run build.sh."); return }
            NSLog("Spotify Gestures camera authorization: %d", AVCaptureDevice.authorizationStatus(for: .video).rawValue)
            switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized: self.configureAndStart()
            case .notDetermined:
                AVCaptureDevice.requestAccess(for: .video) { granted in
                    self.queue.async {
                        if granted { self.configureAndStart() }
                        else { self.show("Camera access denied. Enable it in System Settings.") }
                    }
                }
            default: self.show("Camera access denied. Enable it in System Settings.")
            }
        }
    }

    func stop() {
        queue.async {
            self.wantsCamera = false
            if self.session.isRunning { self.session.stopRunning() }
            self.episodeFrames.removeAll()
            self.recentVideoFrames.removeAll()
            self.episodeVideoFrames.removeAll()
            self.classifying = false
            self.generation += 1
            self.lastProcessTime = -1
            self.lastHandTime = -1
        }
    }

    func setMirrored(_ enabled: Bool) {
        queue.async {
            self.mirrorGestures = enabled
            self.episodeFrames.removeAll()
            self.recentVideoFrames.removeAll()
            self.episodeVideoFrames.removeAll()
            self.generation += 1
        }
    }

    private func configureAndStart() {
        NSLog("Spotify Gestures configure camera; wanted: %@", wantsCamera ? "yes" : "no")
        guard wantsCamera else { return }
        if !configured {
            guard let device = AVCaptureDevice.default(for: .video) else {
                show("No camera found."); return
            }
            do {
                let input = try AVCaptureDeviceInput(device: device)
                session.beginConfiguration()
                session.sessionPreset = session.canSetSessionPreset(.hd1280x720) ? .hd1280x720 : .high
                guard session.canAddInput(input) else {
                    session.commitConfiguration()
                    show("Camera input unavailable.")
                    return
                }
                session.addInput(input)
                let output = AVCaptureVideoDataOutput()
                output.alwaysDiscardsLateVideoFrames = true
                output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
                output.setSampleBufferDelegate(self, queue: queue)
                guard session.canAddOutput(output) else {
                    session.commitConfiguration()
                    show("Camera frames unavailable.")
                    return
                }
                session.addOutput(output)
                session.commitConfiguration()
                configured = true
            } catch { show("Camera setup failed: \(error.localizedDescription)"); return }
        }
        if !session.isRunning { session.startRunning() }
        NSLog("Spotify Gestures camera running: %@", session.isRunning ? "yes" : "no")
        show("Camera on · Show a gesture")
    }

    private func send(_ action: String?, at time: Double) {
        guard let action else {
            show("No gesture matched · try again")
            return
        }
        lastActionTime = time
        show("Recognized: \(action.replacingOccurrences(of: "_", with: " "))")
        spotify.perform(action) { [weak self] message in self?.report?(message) }
    }

    private func finishEpisode(at time: Double) {
        let handFrames = episodeFrames
        let videoFrames = episodeVideoFrames
        episodeFrames.removeAll()
        episodeVideoFrames.removeAll()
        lastHandTime = -1
        lastDiagnosticTime = time
        guard time - lastActionTime > 1.2, let model else { return }
        let handDecision = GestureCore.classifyEpisode(handFrames, model: model)
        guard let videoModel, videoFrames.count >= 4 else {
            send(handDecision.action, at: time)
            return
        }
        classifying = true
        let currentGeneration = generation
        let mirrored = !mirrorGestures // Align Jester labels with the camera's mirrored clips.
        inferenceQueue.async { [weak self] in
            let prediction = videoModel.predict(videoFrames, mirrored: mirrored)
            self?.queue.async { [weak self] in
                guard let self else { return }
                self.classifying = false
                guard self.wantsCamera, self.generation == currentGeneration else { return }
                self.send(GestureFusion.resolve(video: prediction, hand: handDecision),
                          at: CACurrentMediaTime())
            }
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard wantsCamera, !classifying else { return }
        let now = CACurrentMediaTime()
        guard now - lastProcessTime >= 0.10 else { return }
        lastProcessTime = now
        guard let image = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        if let frame = videoModel?.capture(image) {
            recentVideoFrames.append(frame)
            if recentVideoFrames.count > 5 { recentVideoFrames.removeFirst() }
            if !episodeFrames.isEmpty {
                episodeVideoFrames.append(frame)
                if episodeVideoFrames.count > 60 { episodeVideoFrames.removeFirst() }
            }
        }
        if let hand = GestureCore.detect(image, time: now) {
            if episodeFrames.isEmpty { episodeVideoFrames = recentVideoFrames }
            let aligned = mirrorGestures ? HandFrame(time: hand.time, x: 1 - hand.x, y: hand.y,
                                                       openness: hand.openness, confidence: hand.confidence) : hand
            episodeFrames.append(aligned)
            lastHandTime = now
            diagnostic("Hand tracked · move it out of view to send", at: now)
        } else if now - lastHandTime > 0.5 {
            if !episodeFrames.isEmpty { finishEpisode(at: now) }
            else { diagnostic("Camera on · no hand tracked", at: now) }
            return
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    private let camera = CameraController()
    private var window: NSWindow?
    private var statusItem: NSStatusItem?
    private var loginItem: NSMenuItem?
    private var mirrorItem: NSMenuItem?
    private var openHotKey: EventHotKeyRef?
    private var closeHotKey: EventHotKeyRef?
    private var hotKeyHandler: EventHandlerRef?

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSLog("Spotify Gestures did finish launching")
        NSApp.setActivationPolicy(.accessory)
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        item.button?.image = NSImage(systemSymbolName: "hand.raised.fill", accessibilityDescription: "Spotify Gestures")
        let menu = NSMenu()
        menu.addItem(NSMenuItem(title: "Open Camera", action: #selector(openCamera), keyEquivalent: ""))
        menu.addItem(NSMenuItem(title: "Close Camera", action: #selector(closeCamera), keyEquivalent: ""))
        menu.addItem(.separator())
        let mirror = NSMenuItem(title: "Mirror Gesture Directions", action: #selector(toggleMirror), keyEquivalent: "")
        mirror.state = UserDefaults.standard.bool(forKey: "MirrorGestures") ? .on : .off
        menu.addItem(mirror)
        mirrorItem = mirror
        let login = NSMenuItem(title: "Launch at Login", action: #selector(toggleLogin), keyEquivalent: "")
        login.state = SMAppService.mainApp.status == .enabled ? .on : .off
        menu.addItem(login)
        loginItem = login
        menu.addItem(.separator())
        menu.addItem(NSMenuItem(title: "Quit Spotify Gestures", action: #selector(quit), keyEquivalent: ""))
        for entry in menu.items { entry.target = self }
        item.menu = menu
        statusItem = item
        var eventType = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        let handlerStatus = InstallEventHandler(GetApplicationEventTarget(), { _, event, userData in
            guard let event, let userData else { return noErr }
            var id = EventHotKeyID()
            GetEventParameter(event, UInt32(kEventParamDirectObject), UInt32(typeEventHotKeyID), nil,
                              MemoryLayout<EventHotKeyID>.size, nil, &id)
            let app = Unmanaged<AppDelegate>.fromOpaque(userData).takeUnretainedValue()
            DispatchQueue.main.async { id.id == 1 ? app.openCamera() : app.closeCamera() }
            return noErr
        }, 1, &eventType, UnsafeMutableRawPointer(Unmanaged.passUnretained(self).toOpaque()), &hotKeyHandler)
        let openStatus = RegisterEventHotKey(UInt32(kVK_ANSI_O), UInt32(cmdKey | shiftKey), EventHotKeyID(signature: 0x47535452, id: 1), GetApplicationEventTarget(), 0, &openHotKey)
        let closeStatus = RegisterEventHotKey(UInt32(kVK_ANSI_C), UInt32(cmdKey | shiftKey), EventHotKeyID(signature: 0x47535452, id: 2), GetApplicationEventTarget(), 0, &closeHotKey)
        NSLog("Spotify Gestures hotkeys: handler=%d open=%d close=%d", handlerStatus, openStatus, closeStatus)
        if ProcessInfo.processInfo.arguments.contains("--open-camera") { openCamera() }
    }

    @objc func openCamera() {
        NSLog("Spotify Gestures openCamera")
        if window == nil {
            let view = CameraView(frame: NSRect(x: 0, y: 0, width: 680, height: 480))
            let newWindow = NSWindow(contentRect: view.bounds, styleMask: [.titled, .closable, .miniaturizable, .resizable],
                                     backing: .buffered, defer: false)
            newWindow.title = "Spotify Gestures"
            newWindow.minSize = NSSize(width: 480, height: 360)
            newWindow.isReleasedWhenClosed = false
            newWindow.animationBehavior = .none
            newWindow.contentView = view
            newWindow.center()
            newWindow.delegate = self
            view.previewLayer.session = camera.session
            camera.report = { [weak view] message in view?.status.stringValue = message }
            window = newWindow
        }
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        camera.start()
    }

    @objc func closeCamera() { window?.close() }
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        openCamera()
        return true
    }
    @objc func toggleMirror() {
        let enabled = !(mirrorItem?.state == .on)
        UserDefaults.standard.set(enabled, forKey: "MirrorGestures")
        mirrorItem?.state = enabled ? .on : .off
        camera.setMirrored(enabled)
    }
    @objc func toggleLogin() {
        do {
            if SMAppService.mainApp.status == .enabled {
                try SMAppService.mainApp.unregister()
            } else {
                try SMAppService.mainApp.register()
            }
            loginItem?.state = SMAppService.mainApp.status == .enabled ? .on : .off
        } catch {
            let alert = NSAlert()
            alert.messageText = "Could not change Launch at Login"
            alert.informativeText = error.localizedDescription
            alert.runModal()
        }
    }
    func windowWillClose(_ notification: Notification) {
        camera.stop()
    }
    @objc func quit() { NSApp.terminate(nil) }
    func applicationWillTerminate(_ notification: Notification) {
        camera.stop()
        if let openHotKey { UnregisterEventHotKey(openHotKey) }
        if let closeHotKey { UnregisterEventHotKey(closeHotKey) }
        if let hotKeyHandler { RemoveEventHandler(hotKeyHandler) }
    }
}

@main
struct SpotifyGestureApp {
    private static let delegate = AppDelegate()

    static func main() {
        let app = NSApplication.shared
        app.delegate = delegate
        app.run()
    }
}

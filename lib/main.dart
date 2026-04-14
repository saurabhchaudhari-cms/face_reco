import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:camera/camera.dart';
import 'package:face_recognition/detector_painters.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_face_mesh/flutter_face_mesh.dart' as ffm;
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

import 'utils.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Logger — zero-cost in release builds.
// ─────────────────────────────────────────────────────────────────────────────
abstract final class _Log {
  static void d(String msg) {
    assert(() {
      debugPrint('[FaceRecog] $msg');
      return true;
    }());
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CameraImage container — plain data so it can cross an isolate boundary.
// ─────────────────────────────────────────────────────────────────────────────
class _RawCameraFrame {
  const _RawCameraFrame({
    required this.width,
    required this.height,
    required this.planesBytes,
    required this.planesBytesPerRow,
    required this.planesBytesPerPixel,
    required this.isFront,
    required this.isIOS,
  });

  final int width;
  final int height;
  final List<Uint8List> planesBytes;
  final List<int> planesBytesPerRow;
  final List<int?> planesBytesPerPixel;
  final bool isFront;
  final bool isIOS;

  factory _RawCameraFrame.fromCameraImage(
    CameraImage image,
    CameraLensDirection dir,
  ) {
    return _RawCameraFrame(
      width: image.width,
      height: image.height,
      planesBytes: image.planes.map((p) => Uint8List.fromList(p.bytes)).toList(),
      planesBytesPerRow: image.planes.map((p) => p.bytesPerRow).toList(),
      planesBytesPerPixel: image.planes.map((p) => p.bytesPerPixel).toList(),
      isFront: dir == CameraLensDirection.front,
      isIOS: Platform.isIOS,
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Isolate helper — wraps args for compute() so we don't need a global function
// with multiple parameters.
// ─────────────────────────────────────────────────────────────────────────────
class _ConvertArgs {
  const _ConvertArgs({required this.frame, required this.isFront});
  final _RawCameraFrame frame;
  final bool isFront;
}

// Top-level — required by compute().
img.Image _convertFrameIsolate(_ConvertArgs args) {
  final img.Image raw = args.frame.isIOS
      ? _decodeBGRA8888(args.frame)
      : _decodeYUV(args.frame);
  return args.isFront
      ? img.copyRotate(raw, angle: -90)
      : img.copyRotate(raw, angle: 90);
}

// ── Image decoders — top-level so the isolate can reach them ─────────────────

img.Image _decodeBGRA8888(_RawCameraFrame frame) {
  final result = img.Image(width: frame.width, height: frame.height);
  final bytes  = frame.planesBytes[0];
  final bpr    = frame.planesBytesPerRow[0];
  for (int y = 0; y < frame.height; y++) {
    for (int x = 0; x < frame.width; x++) {
      final o = y * bpr + x * 4;
      result.setPixelRgba(x, y, bytes[o + 2], bytes[o + 1], bytes[o], 255);
    }
  }
  return result;
}

img.Image _decodeYUV(_RawCameraFrame frame) {
  final result = img.Image(width: frame.width, height: frame.height);
  final w = frame.width;
  final h = frame.height;

  if (frame.planesBytes.length == 1) {
    final bytes = frame.planesBytes[0];
    final ySize = w * h;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final yp    = bytes[y * w + x] & 0xFF;
        final vuIdx = ySize + (y ~/ 2) * w + (x & ~1);
        final vp    = bytes[vuIdx] & 0xFF;
        final up    = bytes[vuIdx + 1] & 0xFF;
        result.setPixelRgba(x, y,
          (yp + (vp - 128) * 1436 / 1024).round().clamp(0, 255),
          (yp - (up - 128) * 46549 / 131072 - (vp - 128) * 93604 / 131072).round().clamp(0, 255),
          (yp + (up - 128) * 1814 / 1024).round().clamp(0, 255),
          255);
      }
    }
    return result;
  }

  final yBytes        = frame.planesBytes[0];
  final uBytes        = frame.planesBytes[1];
  final vBytes        = frame.planesBytes[2];
  final yRowStride    = frame.planesBytesPerRow[0];
  final uvRowStride   = frame.planesBytesPerRow[1];
  final uvPixelStride = frame.planesBytesPerPixel[1] ?? 1;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      final yp    = yBytes[y * yRowStride + x] & 0xFF;
      final uvIdx = uvPixelStride * (x ~/ 2) + uvRowStride * (y ~/ 2);
      final up    = uBytes[uvIdx] & 0xFF;
      final vp    = vBytes[uvIdx] & 0xFF;
      result.setPixelRgba(x, y,
        (yp + (vp - 128) * 1436 / 1024).round().clamp(0, 255),
        (yp - (up - 128) * 46549 / 131072 - (vp - 128) * 93604 / 131072).round().clamp(0, 255),
        (yp + (up - 128) * 1814 / 1024).round().clamp(0, 255),
        255);
    }
  }
  return result;
}

// ─────────────────────────────────────────────────────────────────────────────

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const FaceRecognitionApp());
}

enum Choice { view, delete }

class FaceRecognitionApp extends StatelessWidget {
  const FaceRecognitionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Face Recognition',
      theme: ThemeData(brightness: Brightness.light, useMaterial3: true),
      home: const FaceRecognitionPage(),
    );
  }
}

class FaceRecognitionPage extends StatefulWidget {
  const FaceRecognitionPage({super.key});

  @override
  State<FaceRecognitionPage> createState() => _FaceRecognitionPageState();
}

class _FaceRecognitionPageState extends State<FaceRecognitionPage> {
  static const Map<DeviceOrientation, int> _orientations = {
    DeviceOrientation.portraitUp: 0,
    DeviceOrientation.landscapeLeft: 90,
    DeviceOrientation.portraitDown: 180,
    DeviceOrientation.landscapeRight: 270,
  };

  // ── Removed: static const int _frameIntervalMs = 40
  //    The 40 ms artificial gate is gone. The _isDetecting bool already
  //    prevents pipeline overlap — no extra throttle needed.
  static const int _maxConsecutiveSkips = 3;

  // Recognition fires every N detection frames so labels update ~6×/sec
  // without blocking the fast bounding-box track.
  static const int _recognitionEveryNFrames = 5;

  CameraController? _cameraController;
  List<CameraDescription> _cameras = [];
  CameraLensDirection _direction = CameraLensDirection.front;

  ffm.FaceDetector? _faceDetector;
  tfl.Interpreter?  _interpreter;

  Directory? _appDir;
  File?      _jsonFile;

  final TextEditingController _nameController = TextEditingController();

  bool _isDetecting   = false;
  bool _isRecognizing = false;
  bool _faceFound     = false;
  bool _busy          = false;

  int _lastPipelineMs  = 0;
  int _skipsRemaining  = 0;
  int _framesSinceReco = 0; // counts detection frames between recognition runs

  // _scanResults drives the painter.
  // Bounding-box positions update every detection frame (Track 1).
  // Labels update every _recognitionEveryNFrames frames (Track 2).
  Map<String, List<ffm.Face>> _scanResults    = {};
  Map<String, List<double>>   _savedEmbeddings = {};
  List<double>?                _currentEmbedding;

  double threshold = 1.0;

  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
    _initialize();
  }

  Future<void> _initialize() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      _Log.d('Camera permission denied');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text('Camera permission is required.'),
          duration: Duration(seconds: 4),
        ));
      }
      return;
    }

    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) throw Exception('No cameras found.');

      _faceDetector = ffm.FaceDetector();
      await _faceDetector!.initialize(
        modelAsset: 'packages/flutter_face_mesh/assets/face_landmarker.task',
        maxFaces: 5,
      );

      try {
        await _loadModel();
      } catch (e) {
        _Log.d('Model load error (recognition disabled): $e');
      }

      _appDir   = await getApplicationDocumentsDirectory();
      _jsonFile = File('${_appDir!.path}/emb.json');
      await _loadSavedFaces();
      await _startCamera();
    } catch (e) {
      _Log.d('Initialization error: $e');
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _loadModel() async {
    _interpreter?.close();
    final options = tfl.InterpreterOptions();
    if (Platform.isAndroid) {
      try { options.addDelegate(tfl.XNNPackDelegate()); }
      catch (e) { _Log.d('XNNPack unavailable: $e'); }
    }
    _interpreter = await tfl.Interpreter.fromAsset(
      'assets/mobilefacenet.tflite',
      options: options,
    );
  }

  Future<void> _loadSavedFaces() async {
    if (_jsonFile == null || !_jsonFile!.existsSync()) { _savedEmbeddings = {}; return; }
    final raw = await _jsonFile!.readAsString();
    if (raw.trim().isEmpty) { _savedEmbeddings = {}; return; }
    final decoded = json.decode(raw) as Map<String, dynamic>;
    _savedEmbeddings = {
      for (final e in decoded.entries)
        if (e.value is List)
          e.key: (e.value as List).map((v) => (v as num).toDouble()).toList(),
    };
  }

  Future<void> _startCamera() async {
    if (_busy) return;
    _busy = true;
    try {
      final camera = _cameras.firstWhere(
        (c) => c.lensDirection == _direction,
        orElse: () => _cameras.first,
      );
      await _stopCamera();

      final ctrl = CameraController(
        camera,
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.nv21
            : ImageFormatGroup.bgra8888,
      );
      _cameraController = ctrl;
      await ctrl.initialize();
      await ctrl.startImageStream(_processCameraImage);
      if (mounted) setState(() {});
    } catch (e) {
      _Log.d('Camera start error: $e');
    } finally {
      _busy = false;
    }
  }

  Future<void> _stopCamera() async {
    final ctrl = _cameraController;
    if (ctrl == null) return;
    _cameraController = null;
    if (mounted) setState(() {});
    try { if (ctrl.value.isStreamingImages) await ctrl.stopImageStream(); } catch (_) {}
    try { await ctrl.dispose(); } catch (_) {}
  }

  // ── FRAME GATE ────────────────────────────────────────────────────────────
  //
  // Only guard: never overlap pipeline runs.
  // The old 40 ms time gate is gone — bounding boxes now update at full
  // camera framerate (~30 fps on most devices).
  //
  void _processCameraImage(CameraImage image) {
    if (_isDetecting || _faceDetector == null) return;
    if (_skipsRemaining > 0) { _skipsRemaining--; return; }

    final ctrl = _cameraController;
    if (ctrl == null) return;

    _isDetecting = true;
    _framesSinceReco++;

    final dir = _direction;
    _detectFaces(image, dir, ctrl).whenComplete(() {
      _isDetecting = false;
      // Adaptive skips: pause extra frames only if detection itself was slow.
      _skipsRemaining = _lastPipelineMs > 80
          ? ((_lastPipelineMs / 40).floor() - 1).clamp(0, _maxConsecutiveSkips)
          : 0;
    });
  }

  // ── TRACK 1: DETECTION (every frame, fast) ────────────────────────────────
  //
  // Sends bytes → MediaPipe → updates bounding boxes immediately.
  // Does NOT convert frames or run TFLite. Typically <15 ms.
  //
  Future<void> _detectFaces(
    CameraImage image,
    CameraLensDirection direction,
    CameraController controller,
  ) async {
    final sw = kDebugMode ? (Stopwatch()..start()) : null;

    final int     rotation = _calculateRotation(controller);
    final Uint8List bytes  = Platform.isAndroid
        ? _buildNV21Bytes(image)
        : image.planes.first.bytes;
    final String format    = Platform.isAndroid ? 'nv21' : 'jpeg';

    final ffm.FaceResult result = await _faceDetector!.detectFromBytes(
      bytes: bytes,
      width: image.width,
      height: image.height,
      rotation: rotation,
      format: format,
    );
    final List<ffm.Face> faces = result.faces;

    // ── No faces: clear everything ────────────────────────────────────────
    if (faces.isEmpty) {
      if (_faceFound || _scanResults.isNotEmpty) {
        if (mounted) setState(() {
          _faceFound        = false;
          _scanResults      = {};
          _currentEmbedding = null;
          _framesSinceReco  = 0;
        });
      }
      if (kDebugMode) _lastPipelineMs = sw!.elapsedMilliseconds;
      return;
    }

    // ── Faces found: update bounding boxes immediately ────────────────────
    //
    // Key insight: we update positions every frame but keep the previous
    // labels. The box jumps to the new position instantly; the label stays
    // stable until the next recognition run completes.
    //
    if (mounted) {
      setState(() {
        _faceFound = true;

        // Rebuild _scanResults preserving existing label → face associations
        // but with the new bounding-box positions from this frame.
        final oldLabels  = _scanResults.keys.toList();
        final newResults = <String, List<ffm.Face>>{};
        for (int i = 0; i < faces.length; i++) {
          // Reuse the label from the same index position if it exists,
          // otherwise show '...' until recognition assigns a real label.
          final label = (i < oldLabels.length) ? oldLabels[i] : '...';
          newResults.putIfAbsent(label, () => []).add(faces[i]);
        }
        _scanResults = newResults;
      });
    }

    // ── TRACK 2: kick off recognition every N frames ──────────────────────
    //
    // _isRecognizing prevents stacking multiple recognition jobs.
    // The frame + face list are snapshotted here before the async gap.
    //
    if (_framesSinceReco >= _recognitionEveryNFrames && !_isRecognizing) {
      _framesSinceReco = 0;
      _isRecognizing   = true;

      // Snapshot everything we need — these will be invalid after await.
      final frameSnap = _RawCameraFrame.fromCameraImage(image, direction);
      final facesSnap = List<ffm.Face>.from(faces);
      final embSnap   = Map<String, List<double>>.from(_savedEmbeddings);

      _runRecognition(frameSnap, facesSnap, embSnap, direction)
          .whenComplete(() => _isRecognizing = false);
    }

    if (kDebugMode) _lastPipelineMs = sw!.elapsedMilliseconds;
  }

  // ── TRACK 2: RECOGNITION ─────────────────────────────────────────────────
  //
  // Step A — frame conversion runs on a background isolate (heavy pixel work).
  // Step B — TFLite inference runs on the main isolate (Interpreter is not
  //           isolate-safe; XNNPack delegate is typically <5 ms per face).
  // Step C — setState updates labels only, bounding boxes are already live.
  //
  Future<void> _runRecognition(
    _RawCameraFrame frame,
    List<ffm.Face> faces,
    Map<String, List<double>> savedEmbeddings,
    CameraLensDirection direction,
  ) async {
    if (_interpreter == null) return;

    try {
      // Step A: convert on background isolate.
      final img.Image converted = await compute(
        _convertFrameIsolate,
        _ConvertArgs(
          frame: frame,
          isFront: direction == CameraLensDirection.front,
        ),
      );

      // Step B: crop + TFLite per face (main isolate, fast with XNNPack).
      final Map<String, List<ffm.Face>> labelledResults = {};
      List<double>? firstEmbedding;

      for (final face in faces) {
        final pixelRect = face.boundingBox.toRect(
          Size(converted.width.toDouble(), converted.height.toDouble()),
        );
        final safeRect = _expandedRect(pixelRect, converted.width, converted.height);

        final img.Image cropped = img.copyCrop(
          converted,
          x: safeRect.left.round(),
          y: safeRect.top.round(),
          width: safeRect.width.round(),
          height: safeRect.height.round(),
        );
        final img.Image resized = img.copyResize(cropped, width: 112, height: 112);

        final String label = _recognize(resized);
        firstEmbedding ??= _currentEmbedding;
        labelledResults.putIfAbsent(label, () => []).add(face);
      }

      // Step C: update labels — boxes are already on screen from Track 1.
      if (mounted) {
        setState(() {
          _scanResults      = labelledResults;
          _currentEmbedding = firstEmbedding;
        });
      }
    } catch (e) {
      _Log.d('Recognition error: $e');
    }
  }

  // ── TFLite inference (main isolate only) ─────────────────────────────────

  String _recognize(img.Image faceImage) {
    if (_interpreter == null) return 'Model not loaded';
    final input  = imageToByteListFloat32(faceImage, 112, 128, 128);
    final shaped = input.reshape([1, 112, 112, 3]);
    final output = [List<double>.filled(192, 0.0)];
    _interpreter!.run(shaped, output);
    final embedding   = List<double>.from(output.first);
    _currentEmbedding = embedding;
    return _compare(embedding).toUpperCase();
  }

  String _compare(List<double> curr) {
    if (_savedEmbeddings.isEmpty) return 'NO FACE SAVED';
    double minDist = 999.0;
    String result  = 'NOT RECOGNIZED';
    for (final entry in _savedEmbeddings.entries) {
      final dist = euclideanDistance(entry.value, curr);
      if (dist <= threshold && dist < minDist) {
        minDist = dist;
        result  = entry.key;
      }
    }
    return result;
  }

  // ── NV21 builder ─────────────────────────────────────────────────────────

  static Uint8List _buildNV21Bytes(CameraImage image) {
    if (image.planes.length == 1) return image.planes[0].bytes;
    if (image.planes.length == 2) {
      final y  = image.planes[0].bytes;
      final vu = image.planes[1].bytes;
      final out = Uint8List(y.length + vu.length);
      out.setRange(0, y.length, y);
      out.setRange(y.length, out.length, vu);
      return out;
    }
    final y  = image.planes[0].bytes;
    final u  = image.planes[1].bytes;
    final v  = image.planes[2].bytes;
    final uvLen = image.width * image.height ~/ 2;
    final out   = Uint8List(y.length + uvLen);
    out.setRange(0, y.length, y);
    final uvPixelStride = image.planes[1].bytesPerPixel ?? 1;
    final uvRowStride   = image.planes[1].bytesPerRow;
    int dst = y.length;
    for (int row = 0; row < image.height ~/ 2; row++) {
      for (int col = 0; col < image.width ~/ 2; col++) {
        final src = row * uvRowStride + col * uvPixelStride;
        out[dst++] = v[src];
        out[dst++] = u[src];
      }
    }
    return out;
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  int _calculateRotation(CameraController controller) {
    final camera            = controller.description;
    final sensorOrientation = camera.sensorOrientation;
    final comp              = _orientations[controller.value.deviceOrientation] ?? 0;
    return camera.lensDirection == CameraLensDirection.front
        ? (sensorOrientation + comp) % 360
        : (sensorOrientation - comp + 360) % 360;
  }

  Rect _expandedRect(Rect box, int imgW, int imgH) {
    const padding = 10.0;
    return Rect.fromLTRB(
      max(0, box.left - padding),
      max(0, box.top - padding),
      min(imgW.toDouble(), box.right + padding),
      min(imgH.toDouble(), box.bottom + padding),
    );
  }

  // ── Camera controls ───────────────────────────────────────────────────────

  Future<void> _toggleCameraDirection() async {
    if (mounted) setState(() {
      _direction = _direction == CameraLensDirection.back
          ? CameraLensDirection.front
          : CameraLensDirection.back;
    });
    await _startCamera();
  }

  Future<void> _resetFile() async {
    _savedEmbeddings  = {};
    _currentEmbedding = null;
    if (_jsonFile != null && _jsonFile!.existsSync()) await _jsonFile!.delete();
    if (mounted) setState(() {});
  }

  Future<void> _viewLabels() async {
    await _stopCamera();
    if (!mounted) return;
    await showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Saved Faces'),
        content: SizedBox(
          width: double.maxFinite,
          height: 320,
          child: _savedEmbeddings.isEmpty
              ? const Center(child: Text('No saved faces'))
              : ListView.separated(
                  itemCount: _savedEmbeddings.length,
                  separatorBuilder: (_, __) => const Divider(height: 1),
                  itemBuilder: (_, i) => ListTile(
                    dense: true,
                    title: Text(_savedEmbeddings.keys.elementAt(i)),
                  ),
                ),
        ),
        actions: [
          TextButton(
            onPressed: () async { Navigator.pop(ctx); await _startCamera(); },
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  Future<void> _addLabel() async {
    if (_currentEmbedding == null) return;
    await _stopCamera();
    if (!mounted) return;
    await showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Add Face'),
        content: Row(children: [
          Expanded(
            child: TextField(
              controller: _nameController,
              autofocus: true,
              decoration: const InputDecoration(
                labelText: 'Name',
                icon: Icon(Icons.face),
              ),
            ),
          ),
        ]),
        actions: [
          TextButton(
            onPressed: () async {
              final name = _nameController.text.trim().toUpperCase();
              if (name.isNotEmpty && _currentEmbedding != null) {
                _savedEmbeddings[name] = List<double>.from(_currentEmbedding!);
                await _saveEmbeddings();
              }
              _nameController.clear();
              if (mounted) Navigator.pop(ctx);
              await _startCamera();
            },
            child: const Text('Save'),
          ),
          TextButton(
            onPressed: () async {
              _nameController.clear();
              if (mounted) Navigator.pop(ctx);
              await _startCamera();
            },
            child: const Text('Cancel'),
          ),
        ],
      ),
    );
  }

  Future<void> _saveEmbeddings() async {
    if (_jsonFile == null) return;
    await _jsonFile!.writeAsString(json.encode(_savedEmbeddings));
    if (mounted) setState(() {});
  }

  // ── UI ────────────────────────────────────────────────────────────────────

  Widget _buildResults() {
    if (_scanResults.isEmpty ||
        _cameraController == null ||
        !_cameraController!.value.isInitialized) {
      return const SizedBox.shrink();
    }
    final previewSize = _cameraController!.value.previewSize!;
    final imageSize   = Size(previewSize.height, previewSize.width);
    return Positioned.fill(
      child: RepaintBoundary(
        child: CustomPaint(
          painter: FaceDetectorPainter(
            imageSize: imageSize,
            results: _scanResults,
            isFrontCamera: _direction == CameraLensDirection.front,
          ),
        ),
      ),
    );
  }

  Widget _buildCameraView() {
    final ctrl = _cameraController;
    if (ctrl == null || !ctrl.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }
    return RepaintBoundary(
      child: Stack(
        fit: StackFit.expand,
        children: [CameraPreview(ctrl), _buildResults()],
      ),
    );
  }

  @override
  void dispose() {
    _nameController.dispose();
    _stopCamera();
    _faceDetector?.dispose();
    _interpreter?.close();
    SystemChrome.setPreferredOrientations(DeviceOrientation.values);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Face Recognition'),
        actions: [
          PopupMenuButton<Choice>(
            onSelected: (choice) async {
              if (choice == Choice.delete) await _resetFile();
              else await _viewLabels();
            },
            itemBuilder: (_) => const [
              PopupMenuItem(value: Choice.view,   child: Text('View Saved Faces')),
              PopupMenuItem(value: Choice.delete, child: Text('Remove all faces')),
            ],
          ),
        ],
      ),
      body: _buildCameraView(),
      floatingActionButton: Column(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          FloatingActionButton(
            backgroundColor: _faceFound ? Colors.blue : Colors.blueGrey,
            heroTag: 'add_face',
            onPressed: _faceFound ? _addLabel : null,
            child: const Icon(Icons.add),
          ),
          const SizedBox(height: 12),
          FloatingActionButton(
            heroTag: 'switch_camera',
            onPressed: _toggleCameraDirection,
            child: Icon(
              _direction == CameraLensDirection.back
                  ? Icons.camera_front
                  : Icons.camera_rear,
            ),
          ),
        ],
      ),
    );
  }
}

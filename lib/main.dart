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
// Minimal logger — zero-cost in release builds.
// All calls are eliminated by the compiler when kDebugMode is false.
// ─────────────────────────────────────────────────────────────────────────────
abstract final class _Log {
  static void d(String msg) {
    assert(() {
      debugPrint('[FaceRecog] $msg');
      return true;
    }());
  }

  // Runs a Stopwatch only in debug builds; returns 0 in release.
}

// ─────────────────────────────────────────────────────────────────────────────
// Plain-data container for CameraImage so it can cross an Isolate boundary.
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
      // Copies bytes once up front so the isolate receives plain Uint8Lists.
      planesBytes: image.planes
          .map((p) => Uint8List.fromList(p.bytes))
          .toList(),
      planesBytesPerRow: image.planes.map((p) => p.bytesPerRow).toList(),
      planesBytesPerPixel: image.planes.map((p) => p.bytesPerPixel).toList(),
      isFront: dir == CameraLensDirection.front,
      isIOS: Platform.isIOS,
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Background isolate entry – must be a top-level function.
// ─────────────────────────────────────────────────────────────────────────────
img.Image convertFrameInBackground(_RawCameraFrame frame) {
  final img.Image raw = frame.isIOS
      ? _decodeBGRA8888(frame)
      : _decodeYUV(frame);
  return frame.isFront
      ? img.copyRotate(raw, angle: -90)
      : img.copyRotate(raw, angle: 90);
}

img.Image _decodeBGRA8888(_RawCameraFrame frame) {
  final result = img.Image(width: frame.width, height: frame.height);
  final bytes = frame.planesBytes[0];
  final bpr = frame.planesBytesPerRow[0];

  for (int y = 0; y < frame.height; y++) {
    for (int x = 0; x < frame.width; x++) {
      final offset = y * bpr + x * 4;
      result.setPixelRgba(
        x,
        y,
        bytes[offset + 2],
        bytes[offset + 1],
        bytes[offset],
        255,
      );
    }
  }
  return result;
}

img.Image _decodeYUV(_RawCameraFrame frame) {
  final result = img.Image(width: frame.width, height: frame.height);
  final w = frame.width;
  final h = frame.height;

  // ── Single-plane packed NV21 ──────────────────────────────────────────────
  if (frame.planesBytes.length == 1) {
    final bytes = frame.planesBytes[0];
    final ySize = w * h;

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final yp = bytes[y * w + x] & 0xFF;
        final vuIdx = ySize + (y ~/ 2) * w + (x & ~1);
        final vp = bytes[vuIdx] & 0xFF;
        final up = bytes[vuIdx + 1] & 0xFF;

        result.setPixelRgba(
          x,
          y,
          (yp + (vp - 128) * 1436 / 1024).round().clamp(0, 255),
          (yp - (up - 128) * 46549 / 131072 - (vp - 128) * 93604 / 131072)
              .round()
              .clamp(0, 255),
          (yp + (up - 128) * 1814 / 1024).round().clamp(0, 255),
          255,
        );
      }
    }
    return result;
  }

  // ── Standard 3-plane YUV420 ───────────────────────────────────────────────
  final yBytes = frame.planesBytes[0];
  final uBytes = frame.planesBytes[1];
  final vBytes = frame.planesBytes[2];
  final yRowStride = frame.planesBytesPerRow[0];
  final uvRowStride = frame.planesBytesPerRow[1];
  final uvPixelStride = frame.planesBytesPerPixel[1] ?? 1;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      final yp = yBytes[y * yRowStride + x] & 0xFF;
      final uvIdx = uvPixelStride * (x ~/ 2) + uvRowStride * (y ~/ 2);
      final up = uBytes[uvIdx] & 0xFF;
      final vp = vBytes[uvIdx] & 0xFF;

      result.setPixelRgba(
        x,
        y,
        (yp + (vp - 128) * 1436 / 1024).round().clamp(0, 255),
        (yp - (up - 128) * 46549 / 131072 - (vp - 128) * 93604 / 131072)
            .round()
            .clamp(0, 255),
        (yp + (up - 128) * 1814 / 1024).round().clamp(0, 255),
        255,
      );
    }
  }
  return result;
}

Uint8List _buildNV21Bytes(CameraImage image) {
  final w = image.width;
  final h = image.height;
  final ySize = w * h;
  final uvSize = (w * h / 2).floor();
  final nv21 = Uint8List(ySize + uvSize);

  // Plane 0 is always Y
  final yPlane = image.planes[0].bytes;
  final yRowStride = image.planes[0].bytesPerRow;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      nv21[y * w + x] = yPlane[y * yRowStride + x];
    }
  }

  // UV planes handling
  if (image.planes.length >= 3) {
    // Standard 3-plane YUV420_888
    final uPlane = image.planes[1].bytes;
    final vPlane = image.planes[2].bytes;
    final uvRowStride = image.planes[1].bytesPerRow;
    final uvPixelStride = image.planes[1].bytesPerPixel ?? 1;

    for (int y = 0; y < (h / 2).floor(); y++) {
      for (int x = 0; x < (w / 2).floor(); x++) {
        final uvIdx = y * uvRowStride + x * uvPixelStride;
        final outIdx = ySize + y * w + x * 2;
        nv21[outIdx] = vPlane[uvIdx];
        nv21[outIdx + 1] = uPlane[uvIdx];
      }
    }
  } else if (image.planes.length == 2) {
    // 2-plane (often Y + VU interleaved)
    final vuPlane = image.planes[1].bytes;
    final vuRowStride = image.planes[1].bytesPerRow;
    for (int y = 0; y < (h / 2).floor(); y++) {
      for (int x = 0; x < w; x++) {
        nv21[ySize + y * w + x] = vuPlane[y * vuRowStride + x];
      }
    }
  }

  return nv21;
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

  // ── 25 fps target (1000 / 25 = 40 ms) ─────────────────────────────────────
  static const int _frameIntervalMs = 40;

  // ── Adaptive frame-skip: if the pipeline ran over budget, skip this many
  //    extra frames before processing again (max 3 so recognition stays fresh).
  static const int _maxConsecutiveSkips = 3;

  CameraController? _cameraController;
  List<CameraDescription> _cameras = [];
  CameraLensDirection _direction = CameraLensDirection.front;

  ffm.FaceDetector? _faceDetector;
  tfl.Interpreter? _interpreter;

  Directory? _appDir;
  File? _jsonFile;

  final TextEditingController _nameController = TextEditingController();

  bool _isDetecting = false;
  bool _faceFound = false;
  bool _busy = false;

  // ── Frame-timing state ────────────────────────────────────────────────────
  int _lastFrameMs = 0; // epoch-ms of last processed frame
  int _lastPipelineMs = 0; // how long the last full pipeline took
  int _skipsRemaining = 0; // adaptive-skip countdown

  Map<String, List<ffm.Face>> _scanResults = {};
  Map<String, List<double>> _savedEmbeddings = {};
  List<double>? _currentEmbedding;

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
    try {
      if (Platform.isAndroid) {
        final status = await Permission.camera.request();
        if (!status.isGranted) throw Exception('Camera permission denied.');
      }

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

      _appDir = await getApplicationDocumentsDirectory();
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
      try {
        options.addDelegate(tfl.XNNPackDelegate());
      } catch (e) {
        _Log.d('XNNPack delegate unavailable, using CPU: $e');
      }
    }

    _interpreter = await tfl.Interpreter.fromAsset(
      'assets/mobilefacenet.tflite',
      options: options,
    );
  }

  Future<void> _loadSavedFaces() async {
    if (_jsonFile == null || !_jsonFile!.existsSync()) {
      _savedEmbeddings = {};
      return;
    }
    final raw = await _jsonFile!.readAsString();
    if (raw.trim().isEmpty) {
      _savedEmbeddings = {};
      return;
    }
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

      if (Platform.isAndroid) {
        final status = await Permission.camera.request();
        if (!status.isGranted) return;
      }

      await _stopCamera();

      final newController = CameraController(
        camera,
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.nv21
            : ImageFormatGroup.bgra8888,
      );

      _cameraController = newController;
      await newController.initialize();
      await newController.startImageStream(_processCameraImage);

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

    try {
      if (ctrl.value.isStreamingImages) await ctrl.stopImageStream();
    } catch (e) {
      _Log.d('Error stopping image stream: $e');
    }

    try {
      await ctrl.dispose();
    } catch (e) {
      _Log.d('Error disposing camera: $e');
    }
  }

  // ── Frame gate: throttle + adaptive skip ─────────────────────────────────
  //
  // Decision matrix:
  //  1. Pipeline still running?           → hard skip (never queue work)
  //  2. Adaptive skips remaining?         → decrement and skip
  //  3. Not yet at 25-fps interval?       → skip
  //  4. All clear                         → process + schedule adaptive skips
  // ─────────────────────────────────────────────────────────────────────────
  void _processCameraImage(CameraImage image) {
    // Guard: never overlap pipeline runs.
    if (_isDetecting || _faceDetector == null) return;

    // Adaptive skip: the previous pipeline was over-budget; drain skip count.
    if (_skipsRemaining > 0) {
      _skipsRemaining--;
      return;
    }

    // Time-based throttle for 25 fps.
    final nowMs = DateTime.now().millisecondsSinceEpoch;
    if (nowMs - _lastFrameMs < _frameIntervalMs) return;

    final ctrl = _cameraController;
    if (ctrl == null) return;

    _lastFrameMs = nowMs;
    _isDetecting = true;

    final directionAtCapture = _direction;
    _analyzeImage(image, directionAtCapture, ctrl).whenComplete(() {
      _isDetecting = false;

      // Adaptive logic: if the pipeline took > 1.5× the target interval,
      // schedule skips proportional to the overshoot (capped).
      if (_lastPipelineMs > _frameIntervalMs * 1.5) {
        _skipsRemaining = ((_lastPipelineMs / _frameIntervalMs).floor() - 1)
            .clamp(0, _maxConsecutiveSkips);
        _Log.d(
          'Pipeline over-budget (${_lastPipelineMs}ms) → '
          'skipping $_skipsRemaining frame(s)',
        );
      } else {
        _skipsRemaining = 0;
      }
    });
  }

  Future<void> _analyzeImage(
    CameraImage image,
    CameraLensDirection direction,
    CameraController controller,
  ) async {
    // Stopwatches are compiled out in release builds via _Log.measure().
    final totalSw = kDebugMode ? (Stopwatch()..start()) : null;

    final int rotation = _calculateRotation(controller);
    final String format = Platform.isAndroid ? 'nv21' : 'jpeg';

    final detectionSw = kDebugMode ? (Stopwatch()..start()) : null;

    // For ffm, we pass the correctly packed NV21 bytes
    final ffm.FaceResult faceResult = await _faceDetector!.detectFromBytes(
      bytes: Platform.isAndroid ? _buildNV21Bytes(image) : image.planes.first.bytes,
      width: image.width,
      height: image.height,
      rotation: rotation,
      format: format,
    );
    final faces = faceResult.faces;
    detectionSw?.stop();

    if (faces.isEmpty) {
      // Avoid setState if state has not actually changed.
      if (_faceFound || _scanResults.isNotEmpty) {
        if (mounted) {
          setState(() {
            _faceFound = false;
            _scanResults = {};
            _currentEmbedding = null;
          });
        }
      }
      if (kDebugMode) _lastPipelineMs = totalSw!.elapsedMilliseconds;
      return;
    }

    // Image conversion on a background isolate.
    final conversionSw = kDebugMode ? (Stopwatch()..start()) : null;
    final frame = _RawCameraFrame.fromCameraImage(image, direction);
    final convertedImage = await compute(convertFrameInBackground, frame);
    conversionSw?.stop();

    final recognitionSw = kDebugMode ? (Stopwatch()..start()) : null;
    final Map<String, List<ffm.Face>> finalResults = {};

    for (final face in faces) {
      // ffm bounding box is normalized [0, 1]. Convert to pixel space of the convertedImage.
      final pixelRect = face.boundingBox.toRect(
        Size(convertedImage.width.toDouble(), convertedImage.height.toDouble()),
      );

      final safeRect = _expandedRect(
        pixelRect,
        convertedImage.width,
        convertedImage.height,
      );

      final cropped = img.copyCrop(
        convertedImage,
        x: safeRect.left.round(),
        y: safeRect.top.round(),
        width: safeRect.width.round(),
        height: safeRect.height.round(),
      );

      final resized = img.copyResize(cropped, width: 112, height: 112);
      final label = _recognize(resized);
      finalResults.putIfAbsent(label, () => []).add(face);
    }
    recognitionSw?.stop();
    totalSw?.stop();

    if (kDebugMode) {
      _lastPipelineMs = totalSw!.elapsedMilliseconds;
      _Log.d(
        'Latency → detect: ${detectionSw!.elapsedMilliseconds}ms | '
        'convert: ${conversionSw!.elapsedMilliseconds}ms | '
        'recognize: ${recognitionSw!.elapsedMilliseconds}ms | '
        'total: ${_lastPipelineMs}ms',
      );
    }

    // Only call setState when the result set has meaningfully changed.
    if (mounted) {
      final didChange = !_mapsEqual(_scanResults, finalResults);
      if (!_faceFound || didChange) {
        setState(() {
          _faceFound = true;
          _scanResults = finalResults;
        });
      }
    }
  }

  /// Shallow equality check to avoid redundant repaints when faces/labels
  /// haven't changed between frames.
  bool _mapsEqual(
    Map<String, List<ffm.Face>> a,
    Map<String, List<ffm.Face>> b,
  ) {
    if (a.length != b.length) return false;
    for (final key in a.keys) {
      if (!b.containsKey(key)) return false;
      if (a[key]!.length != b[key]!.length) return false;
    }
    return true;
  }

  int _calculateRotation(CameraController controller) {
    final camera = controller.description;
    final sensorOrientation = camera.sensorOrientation;
    final comp = _orientations[controller.value.deviceOrientation] ?? 0;

    return camera.lensDirection == CameraLensDirection.front
        ? (sensorOrientation + comp) % 360
        : (sensorOrientation - comp + 360) % 360;
  }

  String _recognize(img.Image faceImage) {
    if (_interpreter == null) return 'Model not loaded';

    final input = imageToByteListFloat32(faceImage, 112, 128, 128);
    final inputReshaped = input.reshape([1, 112, 112, 3]);
    final output = [List<double>.filled(192, 0.0)];

    _interpreter!.run(inputReshaped, output);

    final embedding = List<double>.from(output.first);
    _currentEmbedding = embedding;
    return _compare(embedding).toUpperCase();
  }

  String _compare(List<double> curr) {
    if (_savedEmbeddings.isEmpty) return 'NO FACE SAVED';

    double minDist = 999.0;
    String result = 'NOT RECOGNIZED';

    for (final entry in _savedEmbeddings.entries) {
      final dist = euclideanDistance(entry.value, curr);
      if (dist <= threshold && dist < minDist) {
        minDist = dist;
        result = entry.key;
      }
    }
    return result;
  }

  Future<void> _toggleCameraDirection() async {
    if (mounted) {
      setState(() {
        _direction = _direction == CameraLensDirection.back
            ? CameraLensDirection.front
            : CameraLensDirection.back;
      });
    }
    await _startCamera();
  }

  Future<void> _resetFile() async {
    _savedEmbeddings = {};
    _currentEmbedding = null;
    if (_jsonFile != null && _jsonFile!.existsSync()) {
      await _jsonFile!.delete();
    }
    if (mounted) setState(() {});
  }

  Future<void> _viewLabels() async {
    await _stopCamera();
    if (!mounted) return;

    await showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Saved Faces'),
        content: SizedBox(
          width: double.maxFinite,
          height: 320,
          child: _savedEmbeddings.isEmpty
              ? const Center(child: Text('No saved faces'))
              : ListView.separated(
                  itemCount: _savedEmbeddings.length,
                  separatorBuilder: (_, _) => const Divider(height: 1),
                  itemBuilder: (_, index) => ListTile(
                    dense: true,
                    title: Text(_savedEmbeddings.keys.elementAt(index)),
                  ),
                ),
        ),
        actions: [
          TextButton(
            onPressed: () async {
              Navigator.pop(context);
              await _startCamera();
            },
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
      builder: (context) => AlertDialog(
        title: const Text('Add Face'),
        content: Row(
          children: [
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
          ],
        ),
        actions: [
          TextButton(
            onPressed: () async {
              final name = _nameController.text.trim().toUpperCase();
              if (name.isNotEmpty && _currentEmbedding != null) {
                _savedEmbeddings[name] = List<double>.from(_currentEmbedding!);
                await _saveEmbeddings();
              }
              _nameController.clear();
              if (mounted) Navigator.pop(context);
              await _startCamera();
            },
            child: const Text('Save'),
          ),
          TextButton(
            onPressed: () async {
              _nameController.clear();
              if (mounted) Navigator.pop(context);
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

  Rect _expandedRect(Rect box, int imgW, int imgH) {
    const padding = 10.0;
    return Rect.fromLTRB(
      max(0, box.left - padding),
      max(0, box.top - padding),
      min(imgW.toDouble(), box.right + padding),
      min(imgH.toDouble(), box.bottom + padding),
    );
  }

  // ── UI ────────────────────────────────────────────────────────────────────

  Widget _buildResults() {
    if (_scanResults.isEmpty ||
        _cameraController == null ||
        !_cameraController!.value.isInitialized) {
      return const SizedBox.shrink();
    }

    final previewSize = _cameraController!.value.previewSize!;
    final imageSize = Size(previewSize.height, previewSize.width);

    // RepaintBoundary isolates the overlay repaints from the camera preview.
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
    // RepaintBoundary prevents camera frames from triggering full-tree repaints.
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
              if (choice == Choice.delete) {
                await _resetFile();
              } else {
                await _viewLabels();
              }
            },
            itemBuilder: (_) => const [
              PopupMenuItem(
                value: Choice.view,
                child: Text('View Saved Faces'),
              ),
              PopupMenuItem(
                value: Choice.delete,
                child: Text('Remove all faces'),
              ),
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

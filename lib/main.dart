import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:face_recognition/detector_painters.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_commons/google_mlkit_commons.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

import 'utils.dart';

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
  static const Map<DeviceOrientation, int> _orientations =
      <DeviceOrientation, int>{
        DeviceOrientation.portraitUp: 0,
        DeviceOrientation.landscapeLeft: 90,
        DeviceOrientation.portraitDown: 180,
        DeviceOrientation.landscapeRight: 270,
      };

  CameraController? _cameraController;
  List<CameraDescription> _cameras = <CameraDescription>[];
  CameraLensDirection _direction = CameraLensDirection.front;

  FaceDetector? _faceDetector;
  tfl.Interpreter? _interpreter;

  Directory? _appDir;
  File? _jsonFile;

  final TextEditingController _nameController = TextEditingController();

  bool _isDetecting = false;
  bool _faceFound = false;
  bool _busy = false;

  Map<String, List<Face>> _scanResults = <String, List<Face>>{};
  Map<String, List<double>> _savedEmbeddings = <String, List<double>>{};
  List<double>? _currentEmbedding;

  double threshold = 1.0;

  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations(const <DeviceOrientation>[
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
    _initialize();
  }

  Future<void> _initialize() async {
    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        throw Exception('No camera available on this device.');
      }

      _faceDetector = FaceDetector(
        options: FaceDetectorOptions(
          performanceMode: FaceDetectorMode.accurate,
          enableLandmarks: false,
          enableClassification: false,
          enableContours: false,
          enableTracking: false,
        ),
      );

      await _loadModel();
      _appDir = await getApplicationDocumentsDirectory();
      _jsonFile = File('${_appDir!.path}/emb.json');
      await _loadSavedFaces();
      await _startCamera();
    } catch (e) {
      if (mounted) {
        setState(() {
          _busy = false;
        });
      }
      debugPrint('Initialization error: $e');
    }
  }

  Future<void> _loadModel() async {
    _interpreter?.close();
    _interpreter = await tfl.Interpreter.fromAsset(
      'assets/mobilefacenet.tflite',
    );
  }

  Future<void> _loadSavedFaces() async {
    if (_jsonFile == null || !_jsonFile!.existsSync()) {
      _savedEmbeddings = <String, List<double>>{};
      return;
    }

    final String raw = await _jsonFile!.readAsString();
    if (raw.trim().isEmpty) {
      _savedEmbeddings = <String, List<double>>{};
      return;
    }

    final Map<String, dynamic> decoded =
        json.decode(raw) as Map<String, dynamic>;
    final Map<String, List<double>> loaded = <String, List<double>>{};

    decoded.forEach((String key, dynamic value) {
      if (value is List) {
        loaded[key] = value.map((dynamic e) => (e as num).toDouble()).toList();
      }
    });

    _savedEmbeddings = loaded;
  }

  Future<void> _startCamera() async {
    if (_busy) return;
    _busy = true;

    try {
      final CameraDescription camera = _cameras.firstWhere(
        (CameraDescription c) => c.lensDirection == _direction,
      );

      await _stopCamera();

      _cameraController = CameraController(
        camera,
        ResolutionPreset.low,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.nv21
            : ImageFormatGroup.bgra8888,
      );

      await _cameraController!.initialize();
      await _cameraController!.startImageStream(_processCameraImage);

      if (mounted) {
        setState(() {});
      }
    } catch (e) {
      debugPrint('Camera start error: $e');
    } finally {
      _busy = false;
    }
  }

  Future<void> _stopCamera() async {
    final CameraController? controller = _cameraController;
    if (controller == null) return;

    try {
      if (controller.value.isStreamingImages) {
        await controller.stopImageStream();
      }
    } catch (_) {}

    try {
      await controller.dispose();
    } catch (_) {}

    _cameraController = null;
  }

  void _processCameraImage(CameraImage image) {
    if (_isDetecting || _faceDetector == null || _cameraController == null) {
      return;
    }
    _isDetecting = true;
    _analyzeImage(image).whenComplete(() {
      _isDetecting = false;
    });
  }

  Future<void> _analyzeImage(CameraImage image) async {
    final InputImage? inputImage = _inputImageFromCameraImage(image);
    if (inputImage == null) return;

    final List<Face> faces = await _faceDetector!.processImage(inputImage);

    if (faces.isEmpty) {
      if (mounted) {
        setState(() {
          _faceFound = false;
          _scanResults = <String, List<Face>>{};
          _currentEmbedding = null;
        });
      }
      return;
    }

    final img.Image convertedImage = _convertCameraImage(image, _direction);
    final Map<String, List<Face>> finalResults = <String, List<Face>>{};

    for (final Face face in faces) {
      final Rect safeRect = _expandedRect(
        face.boundingBox,
        convertedImage.width,
        convertedImage.height,
      );

      final img.Image cropped = img.copyCrop(
        convertedImage,
        x: safeRect.left.round(),
        y: safeRect.top.round(),
        width: safeRect.width.round(),
        height: safeRect.height.round(),
      );

      final img.Image resized = img.copyResize(
        cropped,
        width: 112,
        height: 112,
      );

      final String label = _recognize(resized);
      finalResults.putIfAbsent(label, () => <Face>[]).add(face);
    }

    if (mounted) {
      setState(() {
        _faceFound = true;
        _scanResults = finalResults;
      });
    }
  }

  InputImage? _inputImageFromCameraImage(CameraImage image) {
    final CameraController? controller = _cameraController;
    if (controller == null) return null;

    final CameraDescription camera = controller.description;
    final int sensorOrientation = camera.sensorOrientation;

    InputImageRotation? rotation;
    if (Platform.isIOS) {
      rotation = InputImageRotationValue.fromRawValue(sensorOrientation);
    } else if (Platform.isAndroid) {
      final int? rotationCompensation =
          _orientations[controller.value.deviceOrientation];
      if (rotationCompensation == null) return null;

      final int adjusted = camera.lensDirection == CameraLensDirection.front
          ? (sensorOrientation + rotationCompensation) % 360
          : (sensorOrientation - rotationCompensation + 360) % 360;

      rotation = InputImageRotationValue.fromRawValue(adjusted);
    }

    if (rotation == null) return null;

    final InputImageFormat? format = InputImageFormatValue.fromRawValue(
      image.format.raw,
    );

    if (format == null) return null;

    final WriteBuffer allBytes = WriteBuffer();
    for (final Plane plane in image.planes) {
      allBytes.putUint8List(plane.bytes);
    }
    final Uint8List bytes = allBytes.done().buffer.asUint8List();

    return InputImage.fromBytes(
      bytes: bytes,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: format,
        bytesPerRow: image.planes.first.bytesPerRow,
      ),
    );
  }

  // FIX: Split into platform-specific converters. The original code:
  //   1. Accessed planes[1]/[2] on iOS where BGRA8888 has only one plane → crash
  //   2. Used cameraImage.data![index] = ... which is invalid in image v4.x
  //   3. Computed the Y-plane index as y*width+x, ignoring bytesPerRow stride
  img.Image _convertCameraImage(CameraImage image, CameraLensDirection dir) {
    final img.Image converted = Platform.isIOS
        ? _convertBGRA8888(image)
        : _convertYUV420(image);

    return dir == CameraLensDirection.front
        ? img.copyRotate(converted, angle: -90)
        : img.copyRotate(converted, angle: 90);
  }

  /// iOS: single-plane BGRA8888 interleaved buffer.
  img.Image _convertBGRA8888(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    final img.Image result = img.Image(width: width, height: height);
    final Uint8List bytes = image.planes[0].bytes;
    final int bytesPerRow = image.planes[0].bytesPerRow;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int offset = y * bytesPerRow + x * 4;
        final int b = bytes[offset];
        final int g = bytes[offset + 1];
        final int r = bytes[offset + 2];
        // offset + 3 is alpha — ignored
        result.setPixelRgba(x, y, r, g, b, 255);
      }
    }
    return result;
  }

  img.Image _convertYUV420(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    final img.Image result = img.Image(width: width, height: height);

    // Some Android OEMs (e.g. Motorola) deliver NV21 as a single
    // interleaved plane instead of 3 separate planes.
    if (image.planes.length == 1) {
      // Single-plane NV21: [Y*width*height bytes][VU interleaved bytes]
      final Uint8List bytes = image.planes[0].bytes;
      final int ySize = width * height;

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final int yIndex = y * width + x;

          // VU bytes start right after Y block; V comes before U in NV21
          final int vuIndex = ySize + (y ~/ 2) * width + (x & ~1);

          final int yp = bytes[yIndex] & 0xFF;
          final int vp = bytes[vuIndex] & 0xFF;
          final int up = bytes[vuIndex + 1] & 0xFF;

          final int r = (yp + (vp - 128) * 1436 / 1024).round().clamp(0, 255);
          final int g =
              (yp - (up - 128) * 46549 / 131072 - (vp - 128) * 93604 / 131072)
                  .round()
                  .clamp(0, 255);
          final int b = (yp + (up - 128) * 1814 / 1024).round().clamp(0, 255);

          result.setPixelRgba(x, y, r, g, b, 255);
        }
      }
      return result;
    }

    // Standard 3-plane YUV420 path (planes[0]=Y, [1]=U, [2]=V)
    final int yRowStride = image.planes[0].bytesPerRow;
    final int uvRowStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel ?? 1;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int yIndex = y * yRowStride + x;
        final int uvIndex = uvPixelStride * (x ~/ 2) + uvRowStride * (y ~/ 2);

        final int yp = image.planes[0].bytes[yIndex] & 0xFF;
        final int up = image.planes[1].bytes[uvIndex] & 0xFF;
        final int vp = image.planes[2].bytes[uvIndex] & 0xFF;

        final int r = (yp + (vp - 128) * 1436 / 1024).round().clamp(0, 255);
        final int g =
            (yp - (up - 128) * 46549 / 131072 - (vp - 128) * 93604 / 131072)
                .round()
                .clamp(0, 255);
        final int b = (yp + (up - 128) * 1814 / 1024).round().clamp(0, 255);

        result.setPixelRgba(x, y, r, g, b, 255);
      }
    }
    return result;
  }

  String _recognize(img.Image faceImage) {
    if (_interpreter == null) return 'Model not loaded';

    final Float32List input = imageToByteListFloat32(faceImage, 112, 128, 128);

    final List inputReshaped = input.reshape([1, 112, 112, 3]);
    final List<List<double>> output = <List<double>>[
      List<double>.filled(192, 0.0),
    ];

    _interpreter!.run(inputReshaped, output);

    final List<double> embedding = List<double>.from(output.first);
    _currentEmbedding = embedding;

    return _compare(embedding).toUpperCase();
  }

  String _compare(List<double> currEmb) {
    if (_savedEmbeddings.isEmpty) return 'NO FACE SAVED';

    double minDist = 999.0;
    String predRes = 'NOT RECOGNIZED';

    for (final String label in _savedEmbeddings.keys) {
      final double dist = euclideanDistance(_savedEmbeddings[label]!, currEmb);
      if (dist <= threshold && dist < minDist) {
        minDist = dist;
        predRes = label;
      }
    }

    return predRes;
  }

  Future<void> _toggleCameraDirection() async {
    _direction = _direction == CameraLensDirection.back
        ? CameraLensDirection.front
        : CameraLensDirection.back;

    await _startCamera();
  }

  Future<void> _resetFile() async {
    _savedEmbeddings = <String, List<double>>{};
    _currentEmbedding = null;

    if (_jsonFile != null && _jsonFile!.existsSync()) {
      await _jsonFile!.delete();
    }

    if (mounted) {
      setState(() {});
    }
  }

  Future<void> _viewLabels() async {
    if (_cameraController != null) {
      await _stopCamera();
    }

    if (!mounted) return;

    await showDialog<void>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Saved Faces'),
          content: SizedBox(
            width: double.maxFinite,
            height: 320,
            child: _savedEmbeddings.isEmpty
                ? const Center(child: Text('No saved faces'))
                : ListView.separated(
                    itemCount: _savedEmbeddings.length,
                    separatorBuilder: (_, __) => const Divider(height: 1),
                    itemBuilder: (BuildContext context, int index) {
                      final String name = _savedEmbeddings.keys.elementAt(
                        index,
                      );
                      return ListTile(dense: true, title: Text(name));
                    },
                  ),
          ),
          actions: <Widget>[
            TextButton(
              onPressed: () async {
                Navigator.pop(context);
                await _startCamera();
              },
              child: const Text('OK'),
            ),
          ],
        );
      },
    );
  }

  Future<void> _addLabel() async {
    if (_currentEmbedding == null) return;

    if (_cameraController != null) {
      await _stopCamera();
    }

    if (!mounted) return;

    await showDialog<void>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Add Face'),
          content: Row(
            children: <Widget>[
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
          actions: <Widget>[
            TextButton(
              onPressed: () async {
                final String name = _nameController.text.trim().toUpperCase();
                if (name.isNotEmpty && _currentEmbedding != null) {
                  _savedEmbeddings[name] = List<double>.from(
                    _currentEmbedding!,
                  );
                  await _saveEmbeddings();
                }
                _nameController.clear();
                if (mounted) {
                  Navigator.pop(context);
                }
                await _startCamera();
              },
              child: const Text('Save'),
            ),
            TextButton(
              onPressed: () async {
                _nameController.clear();
                if (mounted) {
                  Navigator.pop(context);
                }
                await _startCamera();
              },
              child: const Text('Cancel'),
            ),
          ],
        );
      },
    );
  }

  Future<void> _saveEmbeddings() async {
    if (_jsonFile == null) return;
    await _jsonFile!.writeAsString(json.encode(_savedEmbeddings));
    if (mounted) {
      setState(() {});
    }
  }

  Rect _expandedRect(Rect box, int imageWidth, int imageHeight) {
    const double padding = 10.0;

    final double left = max(0, box.left - padding);
    final double top = max(0, box.top - padding);
    final double right = min(imageWidth.toDouble(), box.right + padding);
    final double bottom = min(imageHeight.toDouble(), box.bottom + padding);

    return Rect.fromLTRB(left, top, right, bottom);
  }

  Widget _buildResults() {
    if (_scanResults.isEmpty || _cameraController == null) {
      return const SizedBox.shrink();
    }

    if (!_cameraController!.value.isInitialized) {
      return const SizedBox.shrink();
    }

    final Size previewSize = _cameraController!.value.previewSize!;
    final Size imageSize = Size(previewSize.height, previewSize.width);

    return Positioned.fill(
      child: CustomPaint(
        painter: FaceDetectorPainter(
          imageSize: imageSize,
          results: _scanResults,
          isFrontCamera: _direction == CameraLensDirection.front,
        ),
      ),
    );
  }

  Widget _buildCameraView() {
    final CameraController? controller = _cameraController;

    if (controller == null || !controller.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }

    return Stack(
      fit: StackFit.expand,
      children: <Widget>[CameraPreview(controller), _buildResults()],
    );
  }

  @override
  void dispose() {
    _nameController.dispose();
    _stopCamera(); // async, but dispose() must be synchronous
    _faceDetector?.close();
    _interpreter?.close();
    SystemChrome.setPreferredOrientations(DeviceOrientation.values);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Face recognition'),
        actions: <Widget>[
          PopupMenuButton<Choice>(
            onSelected: (Choice choice) async {
              if (choice == Choice.delete) {
                await _resetFile();
              } else {
                await _viewLabels();
              }
            },
            itemBuilder: (BuildContext context) => <PopupMenuEntry<Choice>>[
              const PopupMenuItem<Choice>(
                value: Choice.view,
                child: Text('View Saved Faces'),
              ),
              const PopupMenuItem<Choice>(
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
        children: <Widget>[
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

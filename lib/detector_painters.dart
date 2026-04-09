import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class FaceDetectorPainter extends CustomPainter {
  FaceDetectorPainter({
    required this.imageSize,
    required this.results,
    required this.isFrontCamera,
  });

  final Size imageSize;
  final Map<String, List<Face>> results;
  final bool isFrontCamera;

  @override
  void paint(Canvas canvas, Size size) {
    if (imageSize.width == 0 || imageSize.height == 0) return;

    final Paint paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.greenAccent;

    final double scaleX = size.width / imageSize.width;
    final double scaleY = size.height / imageSize.height;

    results.forEach((String label, List<Face> faces) {
      for (final Face face in faces) {
        final Rect rect = face.boundingBox;

        final double left = isFrontCamera
            ? size.width - (rect.right * scaleX)
            : rect.left * scaleX;

        final double right = isFrontCamera
            ? size.width - (rect.left * scaleX)
            : rect.right * scaleX;

        final double top = rect.top * scaleY;
        final double bottom = rect.bottom * scaleY;

        final RRect rrect = RRect.fromLTRBR(
          left,
          top,
          right,
          bottom,
          const Radius.circular(10),
        );

        canvas.drawRRect(rrect, paint);

        final TextSpan span = TextSpan(
          style: TextStyle(
            color: Colors.orange.shade300,
            fontSize: 15,
            fontWeight: FontWeight.w600,
          ),
          text: label,
        );

        final TextPainter textPainter = TextPainter(
          text: span,
          textDirection: TextDirection.ltr,
        )..layout();

        final double textX = isFrontCamera
            ? size.width - ((rect.right * scaleX) + 60)
            : rect.left * scaleX;

        final double textY = (rect.top * scaleY) - 18;

        textPainter.paint(
          canvas,
          Offset(
            textX.clamp(0.0, size.width - 1),
            textY.clamp(0.0, size.height - 1),
          ),
        );
      }
    });
  }

  @override
  bool shouldRepaint(covariant FaceDetectorPainter oldDelegate) {
    return oldDelegate.imageSize != imageSize ||
        oldDelegate.results != results ||
        oldDelegate.isFrontCamera != isFrontCamera;
  }
}

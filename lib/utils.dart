import 'dart:math';
import 'dart:typed_data';

import 'package:image/image.dart' as img;

// FIX: removed unused `buffer` variable and the redundant
//      `.buffer.asFloat32List()` round-trip on the return.
Float32List imageToByteListFloat32(
  img.Image image,
  int inputSize,
  double mean,
  double std,
) {
  final Float32List convertedBytes = Float32List(1 * inputSize * inputSize * 3);
  int pixelIndex = 0;

  for (int i = 0; i < inputSize; i++) {
    for (int x = 0; x < inputSize; x++) {
      final img.Pixel pixel = image.getPixel(x, i);
      convertedBytes[pixelIndex++] =
          (pixel.getChannel(img.Channel.red) - mean) / std;
      convertedBytes[pixelIndex++] =
          (pixel.getChannel(img.Channel.green) - mean) / std;
      convertedBytes[pixelIndex++] =
          (pixel.getChannel(img.Channel.blue) - mean) / std;
    }
  }

  return convertedBytes;
}

double euclideanDistance(List<double> e1, List<double> e2) {
  double sum = 0.0;
  for (int i = 0; i < e1.length; i++) {
    final double diff = e1[i] - e2[i];
    sum += diff * diff; // FIX: avoid pow() overhead for squaring
  }
  return sqrt(sum);
}

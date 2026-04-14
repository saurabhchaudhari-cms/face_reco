# ─── Flutter / Dart ───────────────────────────────────────────────────────────
-keep class io.flutter.** { *; }
-keep class io.flutter.plugins.** { *; }

# ─── Custom Plugin (Face Mesh) ────────────────────────────────────────────────
-keep class com.yourpkg.flutter_face_mesh.** { *; }

# ─── MediaPipe ────────────────────────────────────────────────────────────────
-keep class com.google.mediapipe.** { *; }
-keep interface com.google.mediapipe.** { *; }
-keep enum com.google.mediapipe.** { *; }

# ─── TFLite ───────────────────────────────────────────────────────────────────
-keep class org.tensorflow.** { *; }
-keep class org.tensorflow.lite.** { *; }

# ─── CameraX ──────────────────────────────────────────────────────────────────
-keep class androidx.camera.** { *; }
-keep interface androidx.camera.** { *; }
-keep enum androidx.camera.** { *; }

# ─── Android Lifecycle ────────────────────────────────────────────────────────
-keep class androidx.lifecycle.** { *; }

# ─── R8 missing-class suppressions ───────────────────────────────────────────
-dontwarn javax.annotation.processing.AbstractProcessor
-dontwarn javax.annotation.processing.ProcessingEnvironment
-dontwarn javax.annotation.processing.Processor
-dontwarn javax.annotation.processing.RoundEnvironment
-dontwarn javax.annotation.processing.SupportedAnnotationTypes
-dontwarn javax.annotation.processing.SupportedSourceVersion
-dontwarn javax.lang.model.**
-dontwarn autovalue.shaded.**
-dontwarn com.google.auto.value.processor.**
-dontwarn com.google.auto.value.extension.**
-dontwarn com.google.android.play.core.splitcompat.**
-dontwarn com.google.android.play.core.splitinstall.**
-dontwarn com.google.android.play.core.tasks.**

# Ignore MediaPipe internal proto warnings
-dontwarn com.google.mediapipe.proto.**

# Ignore Java annotation warnings
-dontwarn javax.annotation.**
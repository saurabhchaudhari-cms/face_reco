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

# ── Critical: prevent R8 from stripping attributes MediaPipe stack-walks ──────
-keepattributes EnclosingMethod
-keepattributes InnerClasses
-keepattributes Signature
-keepattributes *Annotation*

# ── Google Flogger (used internally by MediaPipe) ─────────────────────────────
-keep class com.google.common.flogger.** { *; }
-keep interface com.google.common.flogger.** { *; }
-keepclassmembers class com.google.common.flogger.** { *; }

# ── Google Guava (Flogger depends on this) ────────────────────────────────────
-keep class com.google.common.** { *; }
-keepclassmembers class com.google.common.** { *; }

# ── Prevent inlining ONLY for classes Flogger/MediaPipe stack-walks ───────────
# This replaces -dontoptimize (which slows the whole app)
-keepclassmembers,allowshrinking,allowobfuscation class * {
    @com.google.common.flogger.* <methods>;
}
-keep,allowshrinking,allowobfuscation class * extends com.google.common.flogger.AbstractLogger {
    <methods>;
}
# Tell R8 not to inline across MediaPipe's graph loading boundary
-keepclasseswithmembernames class com.google.mediapipe.framework.Graph {
    *;
}

# ── Protobuf (used internally by MediaPipe) ───────────────────────────────────
-keep class com.google.protobuf.** { *; }
-keep interface com.google.protobuf.** { *; }
-keep enum com.google.protobuf.** { *; }
-keepclassmembers class * extends com.google.protobuf.GeneratedMessageLite {
    <fields>;
}
-keepclassmembers class * extends com.google.protobuf.GeneratedMessageV3 {
    <fields>;
}
# Prevent R8 from renaming protobuf fields accessed by reflection
-keepclassmembers class com.google.protobuf.Any {
    private <fields>;
    public <fields>;
}
-keepclassmembers class * implements com.google.protobuf.MessageLite {
    private <fields>;
}
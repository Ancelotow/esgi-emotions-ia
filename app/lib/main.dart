import 'package:camera/camera.dart';
import 'package:emotion_detector_app/presentation/models/camera_session.dart';
import 'package:emotion_detector_app/presentation/views/face_detector_view.dart';
import 'package:flutter/material.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await CameraSession.initialize();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Emotion Detector',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: FaceDetectorView(
        camera: CameraSession.instance.firstCamera,
      ),
    );
  }
}

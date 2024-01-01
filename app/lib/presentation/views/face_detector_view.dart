import 'dart:async';
import 'package:permission_handler/permission_handler.dart' as handler;
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

class FaceDetectorView extends StatefulWidget {
  final CameraDescription camera;

  const FaceDetectorView({
    super.key,
    required this.camera,
  });

  @override
  State<FaceDetectorView> createState() => _FaceDetectorViewState();
}

class _FaceDetectorViewState extends State<FaceDetectorView> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;

  @override
  void initState() {
    super.initState();
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.medium,
    );
    _initializeControllerFuture = _controller.initialize();
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<void>(
      future: _initializeControllerFuture,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.done) {
          Timer.periodic(const Duration(seconds: 10), (Timer t) => _takePicture(context));
          return CameraPreview(_controller);
        } else {
          return const Center(child: CircularProgressIndicator());
        }
      },
    );
  }

  Future<void> requestCameraPermissions() async {
    var status = await handler.Permission.camera.status;
    if (!status.isGranted) {
      status = await handler.Permission.camera.request();
      if (!status.isGranted) {
        throw Exception('Camera permissions not granted');
      }
    }
  }

  Future<void> _takePicture(BuildContext context) async {
    try {
      await _initializeControllerFuture;
      await requestCameraPermissions();
      final image = await _controller.takePicture();
      print('Path to the image: ${image.path}');
    } catch (e) {
      print(e);
    }
  }
}


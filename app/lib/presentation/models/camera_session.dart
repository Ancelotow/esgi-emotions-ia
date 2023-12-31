import 'package:camera/camera.dart';

class CameraSession {

  static CameraSession? _instance;
  final List<CameraDescription> _cameras;

  CameraSession._(this._cameras);

  static Future<void> initialize() async {
    final cameras = await availableCameras();
    _instance = CameraSession._(cameras);
  }

  static CameraSession get instance {
    if (_instance == null) {
      throw Exception('CameraSession is not initialized');
    }
    return _instance!;
  }

  CameraDescription get firstCamera => _cameras.first;

}
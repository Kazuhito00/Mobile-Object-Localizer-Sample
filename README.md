# Mobile-Object-Localizer-Sample
[Google MobileObjectLocalizer](https://tfhub.dev/google/object_detection/mobile_object_localizer_v1/1)のPythonでの動作サンプルです。<br>
ONNXに変換したモデルも同梱しています。<br>変換自体を試したい方は[MobileObjectLocalizer_tf2onnx.ipynb](MobileObjectLocalizer_tf2onnx.ipynb)を使用ください。<br>

<!-- ![smjqx-4ndt8](https://user-images.githubusercontent.com/37477845/133912917-768d2e2e-be8b-4474-8349-b31f56402798.gif) -->
https://user-images.githubusercontent.com/37477845/133913149-fa296996-3f24-46cd-b52d-cb3d0ea6bedf.mp4

2021/09/19時点でTensorFlow Hubで提供されている以下モデルを使用しています。
* [object_detection/mobile_object_localizer_v1](https://tfhub.dev/google/object_detection/mobile_object_localizer_v1/1)

# Overview
MobileObjectLocalizerはクラスに依存しないオブジェクト検出器です。<br>
オブジェクト分類は無く、検出されるクラスは1種類(Entity)のみです。<br>

### Caution
このモデルは、画像内の最も目立つ物体を検出するのに適しています。<br>
自動運転のための障害物や人間の検出などのミッションクリティカルな使用には適していません。

# Requirement 
* TensorFlow 2.6.0 or later
* tensorflow-hub 0.12.0 or later
* OpenCV 3.4.2 or later
* onnxruntime 1.5.2 or later ※ONNX推論を使用する場合のみ

# Demo
デモの実行方法は以下です。
### ONNXデモ
```bash
python demo_onnx.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model<br>
使用モデルのパス指定<br>
デフォルト：'model/object_detection_mobile_object_localizer_v1_1_default_1.onnx'
* --score<br>
キーポイント表示の閾値<br>
デフォルト：0.2

### TensorFlow Liteデモ
```bash
python demo_tflite.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model<br>
使用モデルのパス指定<br>
デフォルト：'model/object_detection_mobile_object_localizer_v1_1_default_1.tflite'
* --score<br>
キーポイント表示の閾値<br>
デフォルト：0.2


# Reference
* [TensorFlow Hub:mobile_object_localizer_v1](https://tfhub.dev/google/object_detection/mobile_object_localizer_v1/1)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
Mobile-Object-Localizer-Sample is under [Apache-2.0 License](LICENSE).

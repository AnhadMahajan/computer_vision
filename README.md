# Computer Vision Projects

A collection of Python scripts and experiments for various computer vision tasks.

## 📁 Folder Structure

- `2.py`, `3.py`, ...: Individual experiment scripts
- `controler.py`: Main controller for running demos
- `eye_blink_detection.py`: Detects eye blinks using webcam
- `fruit_ninja.py`: Fruit Ninja game using hand tracking
- `gender.py`: Gender detection/classification
- `hand_detection.py`: Hand detection using OpenCV/MediaPipe
- `keyboard.py`: Virtual keyboard using hand gestures
- `live_drawing.py`: Draw on screen with hand gestures
- `object_movement.py`: Object tracking and movement detection
- `Sign Language Recognition.py`: Recognize sign language gestures
- `text_extracter.py`: Extract text from images (OCR)
- `train.py`, `training.py`: Training scripts for models
- `yolov8n.pt`: YOLOv8 model weights
- `sign_data.pkl`, `typed_data.json`: Data files for sign language and keyboard
- `shopping.webp`, `tracking_output_*.mp4`: Sample images and output videos

## 🛠️ Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- PyTorch
- Other dependencies as needed (see your scripts)

Install dependencies:
```sh
pip install -r requirements.txt
```
*(Create a `requirements.txt` if you don’t have one yet.)*

## 🚀 Usage

Run any script directly, for example:
```sh
python hand_detection.py
```

## 📄 Notes

- Some scripts require a webcam.
- Model files (e.g., `yolov8n.pt`) may need to be downloaded separately.
- Data files are included for sign language and keyboard recognition.

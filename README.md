This script implements a real-time emotion detection system using a pre-trained deep learning model and OpenCV. It captures video input from a webcam, detects faces, and predicts the corresponding emotion based on facial expressions. The predictions are displayed in real-time on the video feed.

Face Detection: Utilizes the Haar Cascade Classifier for detecting faces in real-time.
Emotion Recognition: Employs a pre-trained CNN model (emotiondetector.json and emotiondetector.h5) to classify emotions into categories like 'angry', 'happy', 'sad', etc.
Real-Time Display: Overlays bounding boxes and predicted emotions on the video feed.

Python Libraries: OpenCV for image processing and real-time video capture, NumPy for data manipulation, and Keras for loading and running the deep learning model.
Machine Learning: A CNN model trained for emotion detection.

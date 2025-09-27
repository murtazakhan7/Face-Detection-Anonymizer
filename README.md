# Face-Detection-Anonymizer
Smart Face Detection: Uses MediaPipe's ML model for accurate face detection
Privacy-Focused: Blurs faces while preserving image quality
Progress Tracking: Shows progress for video processing
Interactive Webcam: Real-time processing with 'q' to quit, 's' to save frames
Auto-Generated Filenames: Creates output files automatically if not specified
Error Handling: Comprehensive error checking and user feedback
Flexible Blur Intensity: Adjustable blur levels (default: 51)

The tool is production-ready and handles edge cases like missing files, webcam access issues, and proper resource cleanup. A privacy-focused tool to detect and anonymize faces in images, videos, and live webcam feeds using OpenCV and Media Pipe.
# Process an image
python face_anonymizer.py --image photo.jpg

# Process a video with custom output
python face_anonymizer.py --video video.mp4 --output anonymized_video.mp4

# Live webcam processing
python face_anonymizer.py --webcam

# Adjust blur intensity
python face_anonymizer.py --image photo.jpg --blur 75

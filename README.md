# Face-Detection-Anonymizer
Developed a privacy-focused tool to detect and anonymize faces in images, videos, and live webcam feeds using OpenCV and Media Pipe.
# Process an image
python face_anonymizer.py --image photo.jpg

# Process a video with custom output
python face_anonymizer.py --video video.mp4 --output anonymized_video.mp4

# Live webcam processing
python face_anonymizer.py --webcam

# Adjust blur intensity
python face_anonymizer.py --image photo.jpg --blur 75

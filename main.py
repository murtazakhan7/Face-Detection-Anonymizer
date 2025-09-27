
"""
Face Detection Anonymizer
A privacy-focused tool to detect and anonymize faces in images, videos, and live webcam feeds.
Uses MediaPipe for face detection and OpenCV for image processing and blurring.
"""

import cv2
import mediapipe as mp
import argparse
import sys
import os
from pathlib import Path


class FaceAnonymizer:
    def __init__(self, blur_intensity=51):
        """
        Initialize the Face Anonymizer with MediaPipe face detection.
        
        Args:
            blur_intensity (int): Intensity of blur effect (must be odd number)
        """
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        # Ensure blur intensity is odd
        self.blur_intensity = blur_intensity if blur_intensity % 2 == 1 else blur_intensity + 1
        
    def detect_and_blur_faces(self, image):
        """
        Detect faces in image and apply blur effect.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Processed image with blurred faces
        """
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image and detect faces
        results = self.face_detection.process(rgb_image)
        
        if results.detections:
            h, w, _ = image.shape
            
            for detection in results.detections:
                # Get bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                       int(bboxC.width * w), int(bboxC.height * h)
                
                x, y, width, height = bbox
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                # Extract face region
                face_region = image[y:y+height, x:x+width]
                
                if face_region.size > 0:
                    # Apply Gaussian blur to face region
                    blurred_face = cv2.GaussianBlur(face_region, 
                                                  (self.blur_intensity, self.blur_intensity), 0)
                    
                    # Replace original face region with blurred version
                    image[y:y+height, x:x+width] = blurred_face
        
        return image
    
    def process_image(self, input_path, output_path=None):
        """
        Process a single image file.
        
        Args:
            input_path (str): Path to input image
            output_path (str): Path to save output image (optional)
        """
        if not os.path.exists(input_path):
            print(f"Error: Input file '{input_path}' not found.")
            return False
            
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not read image from '{input_path}'.")
            return False
        
        print(f"Processing image: {input_path}")
        
        # Process image
        processed_image = self.detect_and_blur_faces(image)
        
        # Determine output path
        if output_path is None:
            path_obj = Path(input_path)
            output_path = str(path_obj.parent / f"{path_obj.stem}_anonymized{path_obj.suffix}")
        
        # Save processed image
        cv2.imwrite(output_path, processed_image)
        print(f"Anonymized image saved to: {output_path}")
        
        return True
    
    def process_video(self, input_path, output_path=None):
        """
        Process a video file.
        
        Args:
            input_path (str): Path to input video
            output_path (str): Path to save output video (optional)
        """
        if not os.path.exists(input_path):
            print(f"Error: Input file '{input_path}' not found.")
            return False
        
        # Open video capture
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video '{input_path}'.")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine output path
        if output_path is None:
            path_obj = Path(input_path)
            output_path = str(path_obj.parent / f"{path_obj.stem}_anonymized{path_obj.suffix}")
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {input_path}")
        print(f"Total frames: {total_frames}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.detect_and_blur_faces(frame)
                
                # Write processed frame
                out.write(processed_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:  # Progress update every 30 frames
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        except KeyboardInterrupt:
            print("\nVideo processing interrupted by user.")
        
        finally:
            # Release everything
            cap.release()
            out.release()
            
        print(f"Anonymized video saved to: {output_path}")
        return True
    
    def process_webcam(self):
        """
        Process live webcam feed.
        """
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access webcam.")
            return False
        
        print("Starting live webcam feed...")
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read from webcam.")
                    break
                
                # Process frame
                processed_frame = self.detect_and_blur_faces(frame)
                
                # Display processed frame
                cv2.imshow('Face Anonymizer - Live Feed', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f"anonymized_frame_{frame_count:04d}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"Frame saved as: {filename}")
                    frame_count += 1
        
        except KeyboardInterrupt:
            print("\nWebcam session interrupted by user.")
        
        finally:
            # Release everything
            cap.release()
            cv2.destroyAllWindows()
        
        print("Webcam session ended.")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Face Detection Anonymizer - Privacy-focused tool for face anonymization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python face_anonymizer.py --image input.jpg
  python face_anonymizer.py --image input.jpg --output anonymized.jpg
  python face_anonymizer.py --video input.mp4
  python face_anonymizer.py --video input.mp4 --output anonymized.mp4
  python face_anonymizer.py --webcam
  python face_anonymizer.py --image input.jpg --blur 75
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', '-i', type=str, 
                           help='Path to input image file')
    input_group.add_argument('--video', '-v', type=str,
                           help='Path to input video file')
    input_group.add_argument('--webcam', '-w', action='store_true',
                           help='Use live webcam feed')
    
    # Output options
    parser.add_argument('--output', '-o', type=str,
                       help='Path to output file (optional, auto-generated if not specified)')
    
    # Processing options
    parser.add_argument('--blur', '-b', type=int, default=51,
                       help='Blur intensity (odd number, default: 51)')
    
    args = parser.parse_args()
    
    # Validate blur intensity
    if args.blur < 1:
        print("Error: Blur intensity must be positive.")
        sys.exit(1)
    
    # Initialize face anonymizer
    try:
        anonymizer = FaceAnonymizer(blur_intensity=args.blur)
    except Exception as e:
        print(f"Error initializing Face Anonymizer: {e}")
        sys.exit(1)
    
    # Process based on input type
    success = False
    
    if args.image:
        success = anonymizer.process_image(args.image, args.output)
    elif args.video:
        success = anonymizer.process_video(args.video, args.output)
    elif args.webcam:
        if args.output:
            print("Warning: Output parameter ignored for webcam mode.")
        success = anonymizer.process_webcam()
    
    if not success:
        sys.exit(1)
    
    print("Processing completed successfully!")


if __name__ == "__main__":
    main()

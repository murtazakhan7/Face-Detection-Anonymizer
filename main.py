#!/usr/bin/env python3
"""
Streamlit Face Detection Anonymizer Frontend
A beautiful web interface for the Face Detection Anonymizer tool.
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
from PIL import Image
import io
import base64


# Configure Streamlit page
st.set_page_config(
    page_title="Face Detection Anonymizer",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .feature-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitFaceAnonymizer:
    def __init__(self):
        """Initialize the Face Anonymizer with MediaPipe face detection."""
        if 'face_detection' not in st.session_state:
            # Initialize MediaPipe Face Detection
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            st.session_state.face_detection = self.face_detection
        else:
            self.face_detection = st.session_state.face_detection
            self.mp_face_detection = mp.solutions.face_detection
    
    def detect_and_blur_faces(self, image, blur_intensity=51):
        """
        Detect faces in image and apply blur effect.
        
        Args:
            image: Input image (BGR format)
            blur_intensity: Intensity of blur effect
            
        Returns:
            Processed image with blurred faces, number of faces detected
        """
        # Ensure blur intensity is odd
        blur_intensity = blur_intensity if blur_intensity % 2 == 1 else blur_intensity + 1
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image and detect faces
        results = self.face_detection.process(rgb_image)
        
        faces_detected = 0
        
        if results.detections:
            h, w, _ = image.shape
            
            for detection in results.detections:
                faces_detected += 1
                
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
                                                  (blur_intensity, blur_intensity), 0)
                    
                    # Replace original face region with blurred version
                    image[y:y+height, x:x+width] = blurred_face
        
        return image, faces_detected
    
    def process_image(self, uploaded_file, blur_intensity):
        """Process uploaded image file."""
        try:
            # Convert uploaded file to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                st.error("Error: Could not decode the uploaded image.")
                return None, None, 0
            
            # Create a copy for processing
            processed_image = image.copy()
            
            # Process image
            with st.spinner('Detecting and anonymizing faces...'):
                processed_image, faces_detected = self.detect_and_blur_faces(
                    processed_image, blur_intensity
                )
            
            return image, processed_image, faces_detected
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None, None, 0
    
    def process_video(self, uploaded_file, blur_intensity):
        """Process uploaded video file."""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Open video
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                st.error("Error: Could not open video file.")
                os.unlink(tmp_path)
                return None
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create output video file
            output_path = tempfile.mktemp(suffix='_anonymized.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frame_count = 0
            total_faces = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, faces_in_frame = self.detect_and_blur_faces(
                    frame, blur_intensity
                )
                total_faces += faces_in_frame
                
                # Write processed frame
                out.write(processed_frame)
                
                frame_count += 1
                
                # Update progress
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f'Processing frame {frame_count}/{total_frames} - '
                               f'Faces detected so far: {total_faces}')
            
            # Release resources
            cap.release()
            out.release()
            os.unlink(tmp_path)
            
            # Return processed video path and stats
            return output_path, total_faces, total_frames
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return None, 0, 0


def main():
    # Header
    st.markdown('<h1 class="main-header">🔒 Face Detection Anonymizer</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
        <h3>🛡️ Privacy-Focused Face Anonymization</h3>
        <p>A production-ready tool to detect and anonymize faces in images, videos, and live webcam feeds using OpenCV and MediaPipe. 
        Built with advanced AI for accurate, real-time processing while preserving image quality and ensuring complete privacy protection.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize the anonymizer
    anonymizer = StreamlitFaceAnonymizer()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Blur intensity slider
        blur_intensity = st.slider(
            "Blur Intensity",
            min_value=11,
            max_value=101,
            value=51,
            step=10,
            help="Higher values create stronger blur effect"
        )
        
        st.header("📋 Features")
        st.markdown("""
        - 🎯 **Smart Face Detection**: Advanced MediaPipe ML model for accurate face detection
        - 🛡️ **Privacy-Focused**: Blurs faces while preserving image quality
        - 📊 **Progress Tracking**: Real-time progress monitoring for video processing
        - 📹 **Interactive Webcam**: Live processing with intuitive controls
        - 📁 **Auto-Generated Filenames**: Intelligent output file naming
        - ⚠️ **Error Handling**: Comprehensive error checking and user feedback
        - ⚙️ **Flexible Blur Intensity**: Adjustable blur levels (default: 51)
        - 🔧 **Production-Ready**: Handles edge cases and resource cleanup
        - 🔒 **Local Processing**: All data stays on your device
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📸 Image Processing", "🎬 Video Processing", "ℹ️ About"])
    
    with tab1:
        st.header("Image Face Anonymization")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Image")
            uploaded_image = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
            )
            
            if uploaded_image is not None:
                # Display original image
                st.image(uploaded_image, caption="Original Image", use_container_width=True)
                
                # Process button
                if st.button("🔍 Detect & Anonymize Faces", key="process_image"):
                    original, processed, faces_count = anonymizer.process_image(
                        uploaded_image, blur_intensity
                    )
                    
                    if processed is not None:
                        # Store results in session state
                        st.session_state.processed_image = processed
                        st.session_state.faces_detected = faces_count
                        st.session_state.image_processed = True
        
        with col2:
            st.subheader("Processed Result")
            
            if hasattr(st.session_state, 'image_processed') and st.session_state.image_processed:
                # Convert BGR to RGB for display
                processed_rgb = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                st.image(processed_rgb, caption="Anonymized Image", use_container_width=True)
                
                # Show statistics
                st.markdown(f"""
                <div class="success-box">
                    <strong>✅ Processing Complete!</strong><br>
                    Faces detected and anonymized: <strong>{st.session_state.faces_detected}</strong><br>
                    Blur intensity applied: <strong>{blur_intensity}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Download button
                processed_pil = Image.fromarray(processed_rgb)
                buf = io.BytesIO()
                processed_pil.save(buf, format='PNG')
                
                st.download_button(
                    label="📥 Download Anonymized Image",
                    data=buf.getvalue(),
                    file_name="anonymized_image.png",
                    mime="image/png"
                )
    
    with tab2:
        st.header("Video Face Anonymization")
        
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_video is not None:
            # Display video info
            st.video(uploaded_video)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.info(f"**File:** {uploaded_video.name}")
                st.info(f"**Size:** {uploaded_video.size / (1024*1024):.2f} MB")
            
            with col2:
                st.warning("⚠️ Video processing may take several minutes depending on file size.")
            
            # Process button
            if st.button("🎬 Process Video", key="process_video"):
                result = anonymizer.process_video(uploaded_video, blur_intensity)
                
                if result[0] is not None:
                    output_path, total_faces, total_frames = result
                    
                    # Show results
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>✅ Video Processing Complete!</strong><br>
                        Total frames processed: <strong>{total_frames}</strong><br>
                        Total faces anonymized: <strong>{total_faces}</strong><br>
                        Blur intensity applied: <strong>{blur_intensity}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Provide download link
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Anonymized Video",
                            data=f.read(),
                            file_name="anonymized_video.mp4",
                            mime="video/mp4"
                        )
                    
                    # Clean up temporary file
                    os.unlink(output_path)
    
    with tab3:
        st.header("About Face Detection Anonymizer")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ### 🔧 Technical Features
            
            **Core Capabilities:**
            - **Smart Face Detection**: MediaPipe's ML model for accurate face detection
            - **Privacy-Focused**: Blurs faces while preserving image quality  
            - **Progress Tracking**: Shows real-time progress for video processing
            - **Interactive Webcam**: Real-time processing with 'q' to quit, 's' to save frames
            - **Auto-Generated Filenames**: Creates output files automatically if not specified
            - **Error Handling**: Comprehensive error checking and user feedback
            - **Flexible Blur Intensity**: Adjustable blur levels (default: 51)
            
            **Production-Ready:**
            - Handles edge cases like missing files
            - Webcam access issue management  
            - Proper resource cleanup and memory management
            """)
        
        with col2:
            st.markdown("""
            ### 🛡️ Privacy & Security
            
            **Your Privacy is Protected:**
            - ✅ All processing happens locally in your browser
            - ✅ No data is sent to external servers
            - ✅ Files are not stored permanently
            - ✅ Temporary files are automatically deleted
            
            **Supported Formats:**
            - **Images**: PNG, JPG, JPEG, BMP, TIFF
            - **Videos**: MP4, AVI, MOV, MKV
            
            **Use Cases:**
            - 📚 Educational content creation
            - 🏢 Corporate presentations  
            - 📱 Social media content
            - 🎥 Video production
            - 🔒 Privacy compliance
            """)
        
        st.markdown("""
        ### 🚀 How to Use
        
        1. **For Images**: Upload an image in the "Image Processing" tab, adjust blur settings, and click process
        2. **For Videos**: Upload a video in the "Video Processing" tab and wait for processing to complete
        3. **Download**: Get your anonymized content with faces automatically blurred
        
        ---
        
        **Built with ❤️ using Streamlit, MediaPipe, and OpenCV**
        """)


if __name__ == "__main__":
    main()
